import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_rope_freqs(head_dim, max_len, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    t = torch.arange(max_len, dtype=torch.float)
    cos = torch.cos(t[:, None] * freqs[None, :])
    sin = torch.sin(t[:, None] * freqs[None, :])
    return cos, sin


def apply_rotary_emb(x, cos, sin, position_offset=0):
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    pos = position_offset
    sl = x.shape[1]
    cos = cos[pos:pos + sl].unsqueeze(0).unsqueeze(2)
    sin = sin[pos:pos + sl].unsqueeze(0).unsqueeze(2)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class GroupedQueryAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads=None, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads or num_heads
        self.head_dim = embed_dim // num_heads
        self.num_key_value_groups = num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(embed_dim, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(num_heads * self.head_dim, embed_dim, bias=False)
        self.attn_drop = dropout

    def forward(self, x, cos=None, sin=None, past_key_value=None, use_cache=False, position_offset=0):
        B, N, _ = x.shape
        q = self.q_proj(x).reshape(B, N, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).reshape(B, N, self.num_kv_heads, self.head_dim)

        if cos is not None:
            q = apply_rotary_emb(q, cos, sin, position_offset)
            k = apply_rotary_emb(k, cos, sin, position_offset)

        if past_key_value is not None:
            k = torch.cat([past_key_value[0], k], dim=1)
            v = torch.cat([past_key_value[1], v], dim=1)

        present = (k, v) if use_cache else None

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=2)
            v = v.repeat_interleave(self.num_key_value_groups, dim=2)

        x = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=not use_cache or past_key_value is None
        ).transpose(1, 2).reshape(B, N, -1)

        return self.o_proj(x), present


class SwiGLU(nn.Module):
    def __init__(self, embed_dim, ff_hidden_dim, dropout=0.0):
        super().__init__()
        self.gate = nn.Linear(embed_dim, ff_hidden_dim, bias=False)
        self.up = nn.Linear(embed_dim, ff_hidden_dim, bias=False)
        self.down = nn.Linear(ff_hidden_dim, embed_dim, bias=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.drop(self.down(F.silu(self.gate(x)) * self.up(x)))


class RefreshGate(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.gate = nn.Linear(embed_dim, embed_dim, bias=False)

    def forward(self, x, embedding):
        g = torch.sigmoid(self.gate(x))
        return x + g * embedding


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, num_kv_heads, ff_hidden_dim, dropout=0.0,
                 use_refresh=False):
        super().__init__()
        self.use_refresh = use_refresh
        self.norm1 = RMSNorm(embed_dim)
        self.attn = GroupedQueryAttention(embed_dim, num_heads, num_kv_heads, dropout)
        self.norm2 = RMSNorm(embed_dim)
        self.ffn = SwiGLU(embed_dim, ff_hidden_dim, dropout)
        if use_refresh:
            self.refresh = RefreshGate(embed_dim)

    def forward(self, x, cos=None, sin=None, use_checkpoint=False, embedding=None,
                past_key_value=None, use_cache=False, position_offset=0):
        if use_checkpoint and self.training:
            x = x + checkpoint.checkpoint(
                lambda x, cos, sin: self.attn(self.norm1(x), cos, sin)[0],
                x, cos, sin, use_reentrant=False
            )
            x = x + checkpoint.checkpoint(
                lambda x: self.ffn(self.norm2(x)),
                x, use_reentrant=False
            )
            present = None
        else:
            attn_out, present = self.attn(self.norm1(x), cos, sin,
                                           past_key_value=past_key_value,
                                           use_cache=use_cache,
                                           position_offset=position_offset)
            x = x + attn_out
            x = x + self.ffn(self.norm2(x))
        if self.use_refresh and embedding is not None:
            x = self.refresh(x, embedding)
        return x, present


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_hidden_dim, num_layers,
                 max_len=512, dropout=0.0, num_kv_heads=None,
                 tied_embeddings=True, rope_theta=2500.0,
                 refresh_layers=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.tied_embeddings = tied_embeddings
        self.token_emb = nn.Embedding(vocab_size, embed_dim)

        head_dim = embed_dim // num_heads
        cos, sin = precompute_rope_freqs(head_dim, max_len, theta=rope_theta)
        self.register_buffer('rope_cos', cos)
        self.register_buffer('rope_sin', sin)

        refresh_set = set(refresh_layers or [])
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, num_kv_heads, ff_hidden_dim, dropout,
                             use_refresh=(i in refresh_set))
            for i in range(num_layers)
        ])
        self.norm = RMSNorm(embed_dim)
        if tied_embeddings:
            self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        else:
            self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.apply(self._init_weights)

        if tied_embeddings:
            self.head.weight = self.token_emb.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, RMSNorm):
            if module.weight is not None:
                torch.nn.init.ones_(module.weight)

    def forward(self, x, use_checkpoint=False, past_key_values=None, use_cache=False):
        emb = self.token_emb(x) * (self.embed_dim ** 0.5)
        if use_cache and past_key_values is not None:
            position_offset = past_key_values[0][0].shape[1]
        else:
            position_offset = 0
        new_past = [] if use_cache else None
        if self.tied_embeddings:
            h = emb
            for i, blk in enumerate(self.blocks):
                pkv = past_key_values[i] if past_key_values is not None else None
                h, present = blk(h, self.rope_cos, self.rope_sin, use_checkpoint,
                                 embedding=emb, past_key_value=pkv,
                                 use_cache=use_cache, position_offset=position_offset)
                if use_cache:
                    new_past.append(present)
        else:
            h = emb
            for i, blk in enumerate(self.blocks):
                pkv = past_key_values[i] if past_key_values is not None else None
                h, present = blk(h, self.rope_cos, self.rope_sin, use_checkpoint,
                                 past_key_value=pkv,
                                 use_cache=use_cache, position_offset=position_offset)
                if use_cache:
                    new_past.append(present)
        h = self.norm(h)
        if use_cache:
            return self.head(h), new_past
        return self.head(h)
