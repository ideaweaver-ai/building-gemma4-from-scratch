"""
Build Gemma 4 ~95M Parameter SLM from Scratch
==============================================
Implements all key Gemma 4 text-decoder innovations over Gemma 3:

  1. Dual head dimensions   – head_dim=64 (sliding) vs global_head_dim=128 (full)
  2. Proportional RoPE      – partial_rotary_factor=0.25 for global layers (only 25% of dims rotated)
  3. Per-Layer Embeddings    – per-layer token conditioning via a second embedding table (PLE)
  4. Shared KV Cache         – later layers reuse K/V from earlier donor layers
  5. K=V Attention           – global layers share key/value projections (no separate V)
  6. Value Normalization     – RMSNorm (without learned scale) applied to values
  7. Logit Softcapping       – tanh-based capping at 30.0 to prevent extreme logits
  8. Embedding Weight Tying  – input embedding and output head share weights
  9. QK Normalization        – attention scaling=1.0 (QK norm controls magnitude)
 10. Zero-centered RMSNorm  – weights initialized to 0, applied as (1 + w)

Architecture reference: google/gemma-4-E2B, google/gemma-4-E4B, google/gemma-4-31B configs
Training on TinyStories dataset with GPT-2 tokenizer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from tqdm.auto import tqdm
from contextlib import nullcontext
import tiktoken


# ============================================================================
# Step 1: Tokenizer and Data Preparation
# ============================================================================

enc = tiktoken.get_encoding("gpt2")


def process(example):
    ids = enc.encode_ordinary(example['text'])
    return {'ids': ids, 'len': len(ids)}


def prepare_data():
    """Download TinyStories, tokenize with GPT-2, and write binary files."""
    from datasets import load_dataset
    ds = load_dataset("roneneldan/TinyStories")
    if os.path.exists("train.bin"):
        print("train.bin already exists — skipping data preparation.")
        return
    tokenized = ds.map(process, remove_columns=['text'], desc="tokenizing", num_proc=8)
    for split, dset in tokenized.items():
        arr_len = np.sum(dset['len'], dtype=np.uint64)
        filename = f'{split}.bin'
        arr = np.memmap(filename, dtype=np.uint16, mode='w+', shape=(arr_len,))
        idx = 0
        for batch_idx in tqdm(range(1024), desc=f'writing {filename}'):
            batch = dset.shard(num_shards=1024, index=batch_idx, contiguous=True).with_format('numpy')
            arr_batch = np.concatenate(batch['ids'])
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)
        arr.flush()


# Uncomment the next line to download and prepare data (run once):
# prepare_data()


# ============================================================================
# Step 2: Model Configuration (~95M parameters)
# ============================================================================

GEMMA4_CONFIG_95M = {
    "vocab_size": 50257,         # GPT-2 tokenizer vocabulary
    "context_length": 2048,
    "emb_dim": 512,
    "n_heads": 8,
    "n_layers": 18,
    "hidden_dim": 1536,          # GeGLU intermediate size

    # --- Gemma 4 new: dual head dimensions ---
    "head_dim": 64,              # Sliding attention layers
    "global_head_dim": 128,      # Full attention layers (2x sliding)

    # --- Gemma 4 new: separate KV head counts ---
    "n_kv_heads": 2,             # KV heads for sliding (GQA group_size = 8/2 = 4)
    "n_global_kv_heads": 1,      # KV heads for global  (GQA group_size = 8/1 = 8)

    "qk_norm": True,

    # --- Gemma 4 new: K=V for global layers ---
    "attention_k_eq_v": True,    # Global layers: V reuses K projection (no separate V weight)

    # --- Gemma 4 new: proportional RoPE with partial rotation ---
    "rope_local_base": 10_000.0,        # Standard theta for sliding layers
    "rope_base": 1_000_000.0,           # Higher theta for global layers (longer context)
    "partial_rotary_factor": 0.25,      # Only 25% of global_head_dim gets RoPE

    "sliding_window": 512,

    # --- Gemma 4 new: shared KV cache ---
    "num_kv_shared_layers": 6,   # Last 6 layers reuse KV from donor layers

    # --- Gemma 4 new: Per-Layer Embeddings ---
    "ple_dim": 16,               # PLE conditioning dimension (set to 0 to disable)

    # --- Gemma 4 new: logit softcapping ---
    "final_logit_softcapping": 30.0,

    "dtype": torch.bfloat16,
    # 3 groups of (5 sliding + 1 full) = 18 layers
    "layer_types": [
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
        "sliding_attention", "sliding_attention", "sliding_attention",
        "sliding_attention", "sliding_attention", "full_attention",
    ],
}


# ============================================================================
# Step 3: Rotary Position Embeddings
#   - Standard RoPE for sliding layers (all dims rotated, theta=10K)
#   - Proportional RoPE for global layers (25% dims rotated, theta=1M)
# ============================================================================

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096,
                        partial_rotary_factor=1.0, dtype=torch.float32):
    """Precompute cos/sin tables. partial_rotary_factor < 1.0 leaves some dims unrotated."""
    rotary_dim = max(2, int(head_dim * partial_rotary_factor) // 2 * 2)  # ensure even
    inv_freq = 1.0 / (theta_base ** (torch.arange(0, rotary_dim, 2, dtype=dtype) / rotary_dim))
    positions = torch.arange(context_length, dtype=dtype)
    angles = positions[:, None] * inv_freq[None, :]   # (context_length, rotary_dim // 2)
    return torch.cos(angles), torch.sin(angles)


def apply_rope(x, cos, sin):
    """Apply RoPE with partial rotation: rotary dims get rotation, rest passes through."""
    _, _, seq_len, _ = x.shape
    rotary_dim = cos.shape[-1] * 2

    # Cast cos/sin to input dtype to avoid float32 upcasting during mixed-precision
    cos = cos[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)   # (1, 1, seq_len, rotary_dim//2)
    sin = sin[:seq_len].unsqueeze(0).unsqueeze(0).to(x.dtype)

    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    x1 = x_rot[..., : rotary_dim // 2]
    x2 = x_rot[..., rotary_dim // 2 :]
    rotated = torch.cat((-x2, x1), dim=-1)

    cos_full = torch.cat([cos, cos], dim=-1)            # (1, 1, seq_len, rotary_dim)
    sin_full = torch.cat([sin, sin], dim=-1)
    x_rot = x_rot * cos_full + rotated * sin_full

    return torch.cat([x_rot, x_pass], dim=-1)


# ============================================================================
# Step 4: RMSNorm — zero-centered weights, optional learned scale
# ============================================================================

class RMSNorm(nn.Module):
    """
    Gemma 4 RMSNorm: weight stored as zeros, applied as (1 + weight).
    Set with_scale=False for value normalization (pure normalization, no learned params).
    """
    def __init__(self, dim, eps=1e-6, with_scale=True):
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.scale = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.with_scale:
            x = x * (1.0 + self.scale.float())
        return x.to(dtype)


# ============================================================================
# Step 5: Grouped Query Attention
#   - Different head_dim per layer type (sliding=64, global=128)
#   - K=V for global layers (V reuses K projection output)
#   - Shared KV: later layers retrieve K/V from donor layers
#   - QK norm → scaling=1.0 (no 1/sqrt(d) scaling needed)
#   - V norm (RMSNorm without learned scale)
# ============================================================================

class GroupedQueryAttention(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        layer_type = cfg["layer_types"][layer_idx]
        self.is_sliding = (layer_type == "sliding_attention")
        self.num_heads = cfg["n_heads"]

        # Head dimension and KV head count depend on layer type
        if self.is_sliding:
            self.head_dim = cfg["head_dim"]
            self.num_kv_heads = cfg["n_kv_heads"]
            self.use_k_eq_v = False
        else:
            self.head_dim = cfg["global_head_dim"]
            self.num_kv_heads = cfg["n_global_kv_heads"]
            self.use_k_eq_v = cfg.get("attention_k_eq_v", False)

        self.group_size = self.num_heads // self.num_kv_heads
        self.d_out = self.num_heads * self.head_dim

        # --- Shared KV logic ---
        n_layers = cfg["n_layers"]
        n_shared = cfg.get("num_kv_shared_layers", 0)
        first_shared = n_layers - n_shared
        self.is_kv_shared = layer_idx >= first_shared > 0
        self.is_donor = False
        self.kv_donor_idx = None

        if self.is_kv_shared:
            # Find the last non-shared layer of the same attention type
            non_shared_types = cfg["layer_types"][:first_shared]
            self.kv_donor_idx = len(non_shared_types) - 1 - non_shared_types[::-1].index(layer_type)
        elif n_shared > 0:
            non_shared_types = cfg["layer_types"][:first_shared]
            last_of_type = len(non_shared_types) - 1 - non_shared_types[::-1].index(layer_type)
            self.is_donor = (layer_idx == last_of_type)

        d_in = cfg["emb_dim"]
        dtype = cfg.get("dtype")

        # Q projection + norm (always present)
        self.W_query = nn.Linear(d_in, self.d_out, bias=False, dtype=dtype)
        self.q_norm = RMSNorm(self.head_dim)

        # K/V projections + norms (only for non-shared layers)
        if not self.is_kv_shared:
            self.W_key = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)
            self.k_norm = RMSNorm(self.head_dim)
            self.v_norm = RMSNorm(self.head_dim, with_scale=False)   # V norm: no learned scale
            if not self.use_k_eq_v:
                self.W_value = nn.Linear(d_in, self.num_kv_heads * self.head_dim, bias=False, dtype=dtype)

        self.out_proj = nn.Linear(self.d_out, d_in, bias=False, dtype=dtype)

    def forward(self, x, mask, cos, sin, shared_kv_states):
        b, seq_len, _ = x.shape

        # Q: project → reshape → norm → RoPE
        queries = self.W_query(x).view(b, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.q_norm(queries)
        queries = apply_rope(queries, cos, sin)

        if self.is_kv_shared:
            # Retrieve pre-computed K/V from the donor layer
            keys, values = shared_kv_states[self.kv_donor_idx]
        else:
            # K: project → reshape
            k_raw = self.W_key(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            # V: either separate projection or reuse K projection (K=V)
            #   When K=V, V gets the raw projection output (before k_norm and RoPE)
            #   K gets: k_norm → RoPE
            #   V gets: v_norm (no RoPE, no learned scale)
            if self.use_k_eq_v:
                v_raw = k_raw
            else:
                v_raw = self.W_value(x).view(b, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

            keys = apply_rope(self.k_norm(k_raw), cos, sin)
            values = self.v_norm(v_raw)

            if self.is_donor:
                shared_kv_states[self.layer_idx] = (keys, values)

        # GQA: expand KV heads to match Q heads
        keys = keys.repeat_interleave(self.group_size, dim=1)
        values = values.repeat_interleave(self.group_size, dim=1)

        # Attention with scaling=1.0 (QK norm handles magnitude)
        attn_scores = queries @ keys.transpose(2, 3)
        attn_scores = attn_scores.masked_fill(mask, -torch.inf)
        # Softmax in float32 for numerical stability, then cast back
        attn_weights = torch.softmax(attn_scores.float(), dim=-1).to(queries.dtype)

        context = (attn_weights @ values).transpose(1, 2).reshape(b, seq_len, self.d_out)
        return self.out_proj(context)


# ============================================================================
# Step 6: Feed Forward — GeGLU (Gated GeLU)
# ============================================================================

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        dtype = cfg.get("dtype")
        self.gate = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=dtype)
        self.up   = nn.Linear(cfg["emb_dim"], cfg["hidden_dim"], bias=False, dtype=dtype)
        self.down = nn.Linear(cfg["hidden_dim"], cfg["emb_dim"], bias=False, dtype=dtype)

    def forward(self, x):
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


# ============================================================================
# Step 7: Transformer Block — with PLE conditioning and layer scalar
# ============================================================================

class TransformerBlock(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.attn_type = cfg["layer_types"][layer_idx]
        emb_dim = cfg["emb_dim"]

        self.att = GroupedQueryAttention(cfg, layer_idx)
        self.ff = FeedForward(cfg)

        self.input_layernorm = RMSNorm(emb_dim)
        self.post_attention_layernorm = RMSNorm(emb_dim)
        self.pre_feedforward_layernorm = RMSNorm(emb_dim)
        self.post_feedforward_layernorm = RMSNorm(emb_dim)

        # Layer scalar: fixed buffer (meaningful only in pretrained checkpoints)
        self.register_buffer("layer_scalar", torch.ones(1, dtype=cfg.get("dtype", torch.float32)))

        # Per-Layer Embedding conditioning block
        self.ple_dim = cfg.get("ple_dim", 0)
        if self.ple_dim > 0:
            dtype = cfg.get("dtype")
            self.ple_gate = nn.Linear(emb_dim, self.ple_dim, bias=False, dtype=dtype)
            self.ple_proj = nn.Linear(self.ple_dim, emb_dim, bias=False, dtype=dtype)
            self.post_ple_norm = RMSNorm(emb_dim)

    def forward(self, x, mask, cos, sin, shared_kv_states, per_layer_input=None):
        input_dtype = x.dtype

        # Pre-norm attention with residual
        shortcut = x
        x = self.input_layernorm(x)
        x = self.att(x, mask, cos, sin, shared_kv_states)
        x = self.post_attention_layernorm(x)
        x = shortcut + x

        # Pre-norm FFN with residual
        shortcut = x
        x = self.pre_feedforward_layernorm(x)
        x = self.ff(x)
        x = self.post_feedforward_layernorm(x)
        x = shortcut + x

        # PLE: gate(hidden) * per_layer_signal → project back → residual
        if self.ple_dim > 0 and per_layer_input is not None:
            shortcut = x
            h = F.gelu(self.ple_gate(x), approximate="tanh") * per_layer_input
            h = self.ple_proj(h)
            h = self.post_ple_norm(h)
            x = shortcut + h

        return (x * self.layer_scalar).to(input_dtype)


# ============================================================================
# Step 8: Gemma 4 Model
# ============================================================================

class Gemma4Model(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        n_layers = cfg["n_layers"]
        emb_dim = cfg["emb_dim"]
        ple_dim = cfg.get("ple_dim", 0)
        dtype = cfg.get("dtype")

        # --- Token embedding (scaled by sqrt(emb_dim)) ---
        self.tok_emb = nn.Embedding(cfg["vocab_size"], emb_dim, dtype=dtype)

        # --- Output head (weight-tied with tok_emb) ---
        self.out_head = nn.Linear(emb_dim, cfg["vocab_size"], bias=False, dtype=dtype)
        self.out_head.weight = self.tok_emb.weight   # Weight tying

        # --- Per-Layer Embeddings (PLE) ---
        self.ple_dim = ple_dim
        if ple_dim > 0:
            # Second embedding table: produces a unique small vector per layer per token
            self.tok_emb_per_layer = nn.Embedding(cfg["vocab_size"], n_layers * ple_dim, dtype=dtype)
            # Context-aware projection from main embeddings
            self.ple_model_proj = nn.Linear(emb_dim, n_layers * ple_dim, bias=False, dtype=dtype)
            self.ple_proj_norm = RMSNorm(ple_dim)
            self.ple_embed_scale = ple_dim ** 0.5
            self.ple_model_proj_scale = emb_dim ** -0.5
            self.ple_combine_scale = 2.0 ** -0.5

        # --- Transformer blocks ---
        self.blocks = nn.ModuleList([TransformerBlock(cfg, i) for i in range(n_layers)])
        self.final_norm = RMSNorm(emb_dim)

        # --- Precompute RoPE tables for both layer types ---
        # Sliding: standard RoPE on all head_dim=64 dims
        cos_local, sin_local = compute_rope_params(
            head_dim=cfg["head_dim"],
            theta_base=cfg["rope_local_base"],
            context_length=cfg["context_length"],
            partial_rotary_factor=1.0,
        )
        # Global: proportional RoPE on 25% of global_head_dim=128 → 32 dims rotated
        cos_global, sin_global = compute_rope_params(
            head_dim=cfg["global_head_dim"],
            theta_base=cfg["rope_base"],
            context_length=cfg["context_length"],
            partial_rotary_factor=cfg["partial_rotary_factor"],
        )
        self.register_buffer("cos_local", cos_local, persistent=False)
        self.register_buffer("sin_local", sin_local, persistent=False)
        self.register_buffer("cos_global", cos_global, persistent=False)
        self.register_buffer("sin_global", sin_global, persistent=False)

    def _create_masks(self, seq_len, device):
        """Create causal (global) and sliding-window (local) attention masks."""
        ones = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device)
        # Global: standard causal — mask future positions (j > i)
        mask_global = torch.triu(ones, diagonal=1)
        # Local: causal + block positions beyond sliding_window
        far_past = torch.triu(ones, diagonal=self.cfg["sliding_window"]).T
        mask_local = mask_global | far_past
        return mask_global, mask_local

    def forward(self, input_ids, targets=None):
        b, seq_len = input_ids.shape

        # Scale embedding by sqrt(emb_dim) — standard Gemma convention
        x = self.tok_emb(input_ids) * (self.cfg["emb_dim"] ** 0.5)

        # Compute PLE signals (before transformer blocks)
        per_layer_inputs = None
        if self.ple_dim > 0:
            n_layers = self.cfg["n_layers"]
            # Token-identity signal: unique per-layer embedding per token
            ple_token = self.tok_emb_per_layer(input_ids) * self.ple_embed_scale
            ple_token = ple_token.view(b, seq_len, n_layers, self.ple_dim)
            # Context-aware signal: project main embeddings
            ple_ctx = self.ple_model_proj(x) * self.ple_model_proj_scale
            ple_ctx = ple_ctx.view(b, seq_len, n_layers, self.ple_dim)
            ple_ctx = self.ple_proj_norm(ple_ctx)
            # Combine both signals
            per_layer_inputs = (ple_token + ple_ctx) * self.ple_combine_scale

        mask_global, mask_local = self._create_masks(seq_len, x.device)

        # Dict populated by donor layers, read by shared layers
        shared_kv_states = {}

        for i, block in enumerate(self.blocks):
            if block.attn_type == "sliding_attention":
                mask, cos, sin = mask_local, self.cos_local, self.sin_local
            else:
                mask, cos, sin = mask_global, self.cos_global, self.sin_global

            ple_input = per_layer_inputs[:, :, i, :] if per_layer_inputs is not None else None
            x = block(x, mask, cos, sin, shared_kv_states, per_layer_input=ple_input)

        x = self.final_norm(x)

        # Output projection with logit softcapping
        logits = self.out_head(x.to(self.cfg.get("dtype", torch.float32)))
        softcap = self.cfg.get("final_logit_softcapping")
        if softcap:
            logits = torch.tanh(logits / softcap) * softcap

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            ctx_len = self.cfg["context_length"]
            idx_cond = idx if idx.size(1) <= ctx_len else idx[:, -ctx_len:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ============================================================================
# Step 9: Instantiate Model and Print Parameter Count
# ============================================================================

torch.manual_seed(42)
model = Gemma4Model(GEMMA4_CONFIG_95M)

total_params = sum(p.numel() for p in model.parameters())
print(f"\nGemma 4 Model — Total parameters: {total_params:,} ({total_params / 1e6:.1f}M)")
print(f"  Embedding (tied):     {model.tok_emb.weight.numel():>12,}")
if model.ple_dim > 0:
    print(f"  PLE embedding:        {model.tok_emb_per_layer.weight.numel():>12,}")
    print(f"  PLE model projection: {model.ple_model_proj.weight.numel():>12,}")
block_params = sum(p.numel() for p in model.blocks.parameters())
print(f"  Transformer blocks:   {block_params:>12,}")
print(f"  Final norm:           {model.final_norm.scale.numel():>12,}")


# ============================================================================
# Step 10: Define the Loss Estimation Function
# ============================================================================

def estimate_loss():
    out = {}
    model.eval()
    with torch.inference_mode():
        for split in ['train', 'validation']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                with ctx:
                    _, loss = model(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
    model.train()
    return out


# ============================================================================
# Step 11: Training Configuration
# ============================================================================

learning_rate = 3e-4
max_iters = 150_000
warmup_steps = 1000
min_lr = 1e-5
eval_iters = 500
batch_size = 16
block_size = 512       # >= sliding_window so local mask differs from global mask
gradient_accumulation_steps = 8

device = "cuda" if torch.cuda.is_available() else "cpu"
device_type = 'cuda' if 'cuda' in device else 'cpu'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

model = model.to(device)


# ============================================================================
# Step 12: Batch Loading
# ============================================================================

def get_batch(split):
    data_file = 'train.bin' if split == 'train' else 'validation.bin'
    data = np.memmap(data_file, dtype=np.uint16, mode='r')
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


# ============================================================================
# Step 13: Optimizer, Scheduler, Scaler
# ============================================================================

from torch.optim.lr_scheduler import LinearLR, SequentialLR, CosineAnnealingLR

optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate,
    betas=(0.9, 0.95), weight_decay=0.1, eps=1e-9,
)

scheduler_warmup = LinearLR(optimizer, total_iters=warmup_steps)
scheduler_decay = CosineAnnealingLR(
    optimizer, T_max=max_iters - warmup_steps, eta_min=min_lr,
)
scheduler = SequentialLR(
    optimizer,
    schedulers=[scheduler_warmup, scheduler_decay],
    milestones=[warmup_steps],
)

scaler = torch.amp.GradScaler(enabled=(dtype == 'float16'))


# ============================================================================
# Step 14: Training Loop
# ============================================================================

best_val_loss = float('inf')
best_model_path = "best_gemma4_model.pt"
train_loss_list, val_loss_list = [], []

model.train()
for step in tqdm(range(max_iters)):
    # Periodic evaluation
    if step % eval_iters == 0 and step > 0:
        losses = estimate_loss()
        print(f"Step {step}: train={losses['train']:.4f}  val={losses['validation']:.4f}  "
              f"lr={optimizer.param_groups[0]['lr']:.6f}")
        train_loss_list.append(losses['train'])
        val_loss_list.append(losses['validation'])

        if losses['validation'] < best_val_loss:
            best_val_loss = losses['validation']
            torch.save(model.state_dict(), best_model_path)

    # Forward + backward
    X, y = get_batch("train")
    with ctx:
        _, loss = model(X, y)
        loss = loss / gradient_accumulation_steps

    scaler.scale(loss).backward()

    # Optimizer step every gradient_accumulation_steps
    if ((step + 1) % gradient_accumulation_steps == 0) or (step + 1 == max_iters):
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    scheduler.step()


# ============================================================================
# Step 15: Plot the Loss Curves
# ============================================================================

import matplotlib.pyplot as plt

train_cpu = [t.cpu().item() if torch.is_tensor(t) else float(t) for t in train_loss_list]
val_cpu = [v.cpu().item() if torch.is_tensor(v) else float(v) for v in val_loss_list]

plt.plot(train_cpu, 'g', label='train_loss')
plt.plot(val_cpu, 'r', label='val_loss')
plt.xlabel(f"Evaluation checkpoints (every {eval_iters} steps)")
plt.ylabel("Loss")
plt.legend()
plt.title("Gemma 4 ~95M Training on TinyStories")
plt.show()


# ============================================================================
# Step 16: Run Inference on the Trained Model
# ============================================================================

model = Gemma4Model(GEMMA4_CONFIG_95M)
model.load_state_dict(torch.load(best_model_path, map_location=torch.device(device)))
model = model.to(device)
model.eval()

prompts = [
    "Once upon a time there was a pumpkin.",
    "A little girl went to the woods",
    "Grandmother was telling the kids story about a unicorn",
]

for sentence in prompts:
    context = torch.tensor(enc.encode_ordinary(sentence)).unsqueeze(0).to(device)
    y = model.generate(context, 200)
    print(f"\n{'='*60}")
    print(f"Prompt: {sentence}")
    print(f"{'='*60}")
    print(enc.decode(y.squeeze().tolist()))
