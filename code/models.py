"""Neural verifier models: independent ResMLP, DeepSets, and set-attention.

All models map per-candidate inputs to a scalar logit and are permutation
equivariant across the four candidates (no positional information).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, dim, expansion=2, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        hidden = dim * expansion
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        h = self.norm(x)
        h = self.fc2(self.drop(F.gelu(self.fc1(h))))
        return x + self.drop(h)


class ResMLPScorer(nn.Module):
    """Independent per-candidate scoring: x_i -> logit s_i."""

    def __init__(self, in_dim, width=256, blocks=2, dropout=0.1):
        super().__init__()
        self.inp = nn.Linear(in_dim, width)
        self.blocks = nn.ModuleList(ResBlock(width, 2, dropout) for _ in range(blocks))
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x):
        # x: [B, 4, D]
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        return self.head(self.norm(h)).squeeze(-1)  # [B, 4]


class DeepSetsScorer(nn.Module):
    """h_i = phi(x_i); c = mean_j h_j; s_i = rho(h_i, c, h_i-c, h_i*c)."""

    def __init__(self, in_dim, width=128, dropout=0.1):
        super().__init__()
        self.phi = nn.Sequential(
            nn.Linear(in_dim, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width), nn.GELU(),
        )
        self.rho = nn.Sequential(
            nn.Linear(width * 4, width), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(width, width // 2), nn.GELU(),
            nn.Linear(width // 2, 1),
        )

    def forward(self, x):
        h = self.phi(x)                     # [B, 4, W]
        c = h.mean(dim=1, keepdim=True).expand_as(h)
        z = torch.cat([h, c, h - c, h * c], dim=-1)
        return self.rho(z).squeeze(-1)      # [B, 4]


class SetAttentionScorer(nn.Module):
    """Bidirectional self-attention over the 4 candidates (permutation equivariant).

    With ``q_dim`` set, a question vector modulates every candidate through
    FiLM (feature-wise linear modulation) before the set-attention blocks.
    The modulation head is zero-initialised, so training starts exactly at the
    unconditioned model.
    """

    def __init__(self, in_dim, width=128, layers=2, heads=4, ffn=256, dropout=0.1,
                 q_dim=None, q_mode='film'):
        super().__init__()
        self.proj = nn.Linear(in_dim, width)
        self.q_mode = q_mode if q_dim else None
        if q_dim:
            self.q_proj = nn.Sequential(
                nn.Linear(q_dim, width), nn.GELU(), nn.Linear(width, 2 * width))
            nn.init.zeros_(self.q_proj[-1].weight)
            nn.init.zeros_(self.q_proj[-1].bias)
        else:
            self.q_proj = None
        self.norms1 = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.attn = nn.ModuleList(
            nn.MultiheadAttention(width, heads, dropout=dropout, batch_first=True)
            for _ in range(layers))
        self.norms2 = nn.ModuleList(nn.LayerNorm(width) for _ in range(layers))
        self.ffn = nn.ModuleList(nn.Sequential(
            nn.Linear(width, ffn), nn.GELU(), nn.Dropout(dropout), nn.Linear(ffn, width))
            for _ in range(layers))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, 1)

    def forward(self, x, q=None):
        h = self.proj(x)                    # [B, 4, W]
        if q is not None and self.q_mode == 'film':
            gamma, beta = self.q_proj(q).chunk(2, dim=-1)
            h = h * (1.0 + gamma[:, None, :]) + beta[:, None, :]
        for n1, at, n2, ff in zip(self.norms1, self.attn, self.norms2, self.ffn):
            u = n1(h)
            a, _ = at(u, u, u, need_weights=False)
            h = h + self.drop(a)
            h = h + self.drop(ff(n2(h)))
        return self.head(self.norm(h)).squeeze(-1)  # [B, 4]


def build_model(name, in_dim, **kw):
    if name == 'resmlp':
        return ResMLPScorer(in_dim, **kw)
    if name == 'deepsets':
        return DeepSetsScorer(in_dim, **kw)
    if name == 'setattn':
        return SetAttentionScorer(in_dim, **kw)
    raise ValueError(name)
