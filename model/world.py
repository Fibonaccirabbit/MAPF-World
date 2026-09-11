"""MAPF-World: shared backbone with a direct fast head and a world branch."""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

# GPT components adapted from MAPF-GPT and MAPF-GPT-DDG (MIT License).
# https://github.com/CognitiveAISystems/MAPF-GPT/blob/main/gpt/model.py
# https://github.com/Cognitive-AI-Systems/MAPF-GPT-DDG/blob/main/gpt/model.py
# Their GPT building blocks follow nanoGPT: https://github.com/karpathy/nanoGPT.
# Copyright (c) 2024 Anton Andreychuk, Alexey Skrynnik.
# Copyright (c) 2022 Andrej Karpathy (nanoGPT).
# See the repository LICENSE for the MIT terms.


try:
    from loguru import logger
except ImportError:
    import logging

    logger = logging.getLogger(__name__)


class LayerNorm(nn.Module):
    """Layer normalization with an optional bias."""

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)


class NonCausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.capture_attention = False
        self._attn_weights = None
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            logger.warning("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # Mask future tokens.
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size)).view(
                    1, 1, config.block_size, config.block_size
                ),
            )

        self.sparse_attn = bool(getattr(config, "sparse_attn", False))
        self.sparse_attn_window = int(getattr(config, "sparse_attn_window", 0))
        if self.sparse_attn and self.sparse_attn_window > 0:
            pos = torch.arange(config.block_size)
            dist = (pos[:, None] - pos[None, :]).abs()
            mask = dist <= self.sparse_attn_window
            self.register_buffer("_sparse_mask", mask, persistent=False)
        else:
            self.register_buffer("_sparse_mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # Split projections into attention heads.
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if (
            self.flash
            and not self.capture_attention
            and not (self.sparse_attn and self.sparse_attn_window > 0)
        ):
            y = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=False,
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            if self.sparse_attn and self.sparse_attn_window > 0 and T <= self._sparse_mask.size(0):
                m = self._sparse_mask[:T, :T].to(att.device)
                att = att.masked_fill(~m.view(1, 1, T, T), float("-inf"))
            att = F.softmax(att, dim=-1)
            if self.capture_attention:
                self._attn_weights = att.detach()
            att = self.attn_dropout(att)
            y = att @ v
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        y = self.resid_dropout(self.c_proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = NonCausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


@dataclass
class GPTConfig:
    block_size: int = 161
    vocab_size: int = 67
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    sparse_attn: bool = False
    sparse_attn_window: int = 0


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(
            dict(
                wte=nn.Embedding(config.vocab_size, config.n_embd),
                wpe=nn.Embedding(config.block_size, config.n_embd),
                drop=nn.Dropout(config.dropout),
                h=nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
                ln_f=LayerNorm(config.n_embd, bias=config.bias),
            )
        )
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = (
            self.lm_head.weight
        )  # https://paperswithcode.com/method/weight-tying

        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, (
            f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        )
        pos = torch.arange(0, t, dtype=torch.long, device=device)  # shape (t)
        tok_emb = self.transformer.wte(idx)  # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos)  # position embeddings of shape (t, n_embd)
        x = self.transformer.drop(tok_emb + pos_emb)
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        else:
            # Project only the final token during inference.
            logits = self.lm_head(x[:, [-1], :])  # note: using list [-1] to preserve the time dim
            loss = None

        return logits, loss

    @torch.no_grad()
    def act(self, idx, do_sample=True, return_probs=False):
        logits, _ = self(idx)
        logits = logits[:, -1, :]

        mask = torch.ones_like(logits) * float("-inf")
        mask[:, :5] = logits[:, :5]
        masked_logits = mask

        probs = F.softmax(masked_logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        if return_probs:
            return idx_next.squeeze(), probs[:, :5]
        return idx_next.squeeze()


# Positional encoding adapted from MAPF-World world/Rop.py.
# The vocabulary layout follows the MAPF-GPT token convention:
# https://github.com/CognitiveAISystems/MAPF-GPT/tree/main/tokenizer


coord_range = list(range(-20, 20 + 1)) + [-20 * 4, -20 * 2, 20 * 2]
actions_range = ["n", "w", "u", "d", "l", "r"]
next_action_range = [format(i, "04b") for i in range(16)]  # 0000 to 1111
vocab = {
    token: idx for idx, token in enumerate(coord_range + actions_range + next_action_range + ["!"])
}  # '!' is a trash symbol
inverse_vocab = {idx: token for token, idx in vocab.items()}


class RotationalPositionEmbedding(nn.Module):
    def __init__(self, d_model=128, device="cuda"):
        super().__init__()
        self.d_model = d_model
        self.costmap_size = 11
        self.costmap_tokens = self.costmap_size**2  # 121
        self.num_agents = 13
        self.agent_token_len = 10  # Tokens per agent.
        self.total_len = 256
        self.device = device
        self.inverse_vocab = inverse_vocab
        self.linear = nn.Linear(3, d_model)
        self.register_buffer("vocab_tensor", self.build_vocab_tensor(inverse_vocab))

    def build_vocab_tensor(self, inverse_vocab):
        max_idx = max(inverse_vocab.keys())
        vocab_tensor = torch.full(
            (max_idx + 1,), float("nan"), dtype=torch.float32, device=self.device
        )
        for idx, token in inverse_vocab.items():
            try:
                vocab_tensor[idx] = float(token)
            except (ValueError, TypeError):
                vocab_tensor[idx] = float("nan")
        return vocab_tensor

    def _get_polar(self, x, y):
        r = torch.sqrt(x**2 + y**2)
        theta = torch.atan2(y, x)
        return torch.stack([r, torch.sin(theta), torch.cos(theta)], dim=-1)

    def forward(self, token_ids: torch.Tensor):
        """
        token_ids: (B, 256) discrete vocabulary indices.
        Returns:
            rot_pos_emb: (B, 256, d_model)
        """
        B = token_ids.shape[0]
        D = self.d_model
        device = token_ids.device

        # Encode spatial locations for the first 121 costmap tokens.
        coords = torch.stack(
            torch.meshgrid(torch.arange(-5, 6), torch.arange(-5, 6), indexing="ij"), dim=-1
        )  # (11,11,2)
        coords = coords.view(-1, 2).to(torch.float32).to(device)  # (121, 2)
        costmap_pos = self._get_polar(coords[:, 0], coords[:, 1])  # (121, 3)
        costmap_pos = self.linear(costmap_pos)  # (121, D)
        costmap_pos = costmap_pos.unsqueeze(0).expand(B, -1, -1)  # (B, 121, D)

        agent_base = (
            self.costmap_tokens
            + torch.arange(self.num_agents, device=device).unsqueeze(1) * self.agent_token_len
        )  # (13,1)
        offsets = torch.tensor([0, 1, 2, 3], device=device).unsqueeze(0)  # (1,4)
        token_indices = agent_base + offsets  # (13,4)
        token_indices = token_indices.view(-1)  # (52,)
        token_values = token_ids[:, token_indices]  # (B, 52)

        max_valid_id = self.vocab_tensor.shape[0] - 1
        token_values = torch.clamp(token_values, 0, max_valid_id)
        coord_values = self.vocab_tensor[token_values]  # (B, 52)
        coord_values = torch.nan_to_num(coord_values, nan=0.0, posinf=0.0, neginf=0.0)

        # Shape (B, 13, 4): x_start, y_start, x_goal, y_goal.
        coords = coord_values.view(B, self.num_agents, 4)
        rel_start = coords[:, :, 0:2]
        rel_goal = coords[:, :, 2:4]

        rpe_start = self._get_polar(rel_start[..., 0], rel_start[..., 1])  # (B, 13, 3)
        rpe_goal = self._get_polar(rel_goal[..., 0], rel_goal[..., 1])  # (B, 13, 3)
        rpe_start_embed = self.linear(rpe_start)  # (B, 13, D)
        rpe_goal_embed = self.linear(rpe_goal)  # (B, 13, D)

        rpe_start_pos = rpe_start_embed.unsqueeze(2).expand(
            -1, -1, 2, -1
        )  # offsets 0-1: agent position
        rpe_goal_pos = rpe_goal_embed.unsqueeze(2).expand(-1, -1, 2, -1)  # offsets 2-3: agent goal
        rpe_start_act = rpe_start_embed.unsqueeze(2).expand(
            -1, -1, 6, -1
        )  # offsets 4-9: spatial association

        rpe_embed = torch.cat([rpe_start_pos, rpe_goal_pos, rpe_start_act], dim=-2)
        rpe_agents = rpe_embed.reshape(B, 130, D)

        pad = torch.zeros(B, 5, D, device=device)
        rot_pos_emb = torch.cat([costmap_pos, rpe_agents, pad], dim=1)  # (B, 256, D)

        if torch.isnan(rot_pos_emb).any():
            print("⚠️ NaN detected in rot_pos_emb!")
            print("coord_values:", coord_values)
            print("token_values:", token_values)
            raise ValueError("NaN in rotational position embedding")

        return rot_pos_emb


# Adapted from MAPF-World world/world.py and world/Rop.py.


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).add(self.eps).sqrt()
        x = x / rms
        return self.weight * x


class MultiHeadAttentionWithRoPE(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        dropout,
        *,
        block_size: int,
        sparse_attn: bool = False,
        sparse_attn_window: int = 0,
        sparse_attn_mode: str = "sliding",
    ):
        super().__init__()
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        assert self.head_dim % 2 == 0, "head_dim must be even for RoPE"

        self.qkv_proj = nn.Linear(n_embd, 3 * n_embd)
        self.out_proj = nn.Linear(n_embd, n_embd)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        self.capture_attention = False
        self._attn_weights = None
        self.sparse_attn = bool(sparse_attn)
        self.sparse_attn_window = int(sparse_attn_window)
        self.sparse_attn_mode = str(sparse_attn_mode)
        self.block_size = int(block_size)

        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.head_dim, 2).float() / self.head_dim))
        self.register_buffer("inv_freq", inv_freq)

        mask = self._build_sparse_mask()
        self.register_buffer("_sparse_mask", mask, persistent=False)

    def _build_sparse_mask(self):
        if not self.sparse_attn:
            return torch.empty(0, dtype=torch.bool)

        if self.sparse_attn_mode == "structured":
            if self.block_size != 256:
                pos = torch.arange(self.block_size)
                dist = (pos[:, None] - pos[None, :]).abs()
                if self.sparse_attn_window > 0:
                    return dist <= self.sparse_attn_window
                return torch.empty(0, dtype=torch.bool)

            map_h = 11
            map_w = 11
            map_tokens = map_h * map_w
            num_agents = 13
            tokens_per_agent = 10
            pad_tokens = 5
            radius = max(0, int(self.sparse_attn_window))

            mask = torch.zeros((self.block_size, self.block_size), dtype=torch.bool)
            diag = torch.arange(self.block_size)
            mask[diag, diag] = True

            for i in range(map_tokens):
                ri, ci = divmod(i, map_w)
                for j in range(map_tokens):
                    rj, cj = divmod(j, map_w)
                    if abs(ri - rj) + abs(ci - cj) <= radius:
                        mask[i, j] = True

            agent_base = map_tokens
            agent_total = num_agents * tokens_per_agent
            agent_end = agent_base + agent_total

            agent_node_indices = [agent_base + k * tokens_per_agent for k in range(num_agents)]

            for k in range(num_agents):
                start = agent_base + k * tokens_per_agent
                end = start + tokens_per_agent
                mask[start:end, start:end] = True
                mask[start:end, agent_node_indices[k]] = True

            node_idx_tensor = torch.tensor(agent_node_indices, dtype=torch.long)
            mask[node_idx_tensor[:, None], node_idx_tensor[None, :]] = True
            mask[agent_base:agent_end, node_idx_tensor] = True

            map_idx = torch.arange(map_tokens, dtype=torch.long)
            mask[map_idx[:, None], node_idx_tensor[None, :]] = True
            mask[node_idx_tensor[:, None], map_idx[None, :]] = True

            if pad_tokens > 0:
                pad_start = self.block_size - pad_tokens
                pad_idx = torch.arange(pad_start, self.block_size, dtype=torch.long)
                mask[pad_idx[:, None], :] = False
                mask[:, pad_idx[None, :]] = False
                mask[pad_idx, pad_idx] = True

            return mask

        if self.sparse_attn_window <= 0:
            return torch.empty(0, dtype=torch.bool)

        pos = torch.arange(self.block_size)
        dist = (pos[:, None] - pos[None, :]).abs()
        return dist <= self.sparse_attn_window

    def _get_rope_sin_cos(self, seq_len, device, dtype):
        t = torch.arange(seq_len, device=device, dtype=dtype)  # (seq_len,)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # (seq_len, head_dim//2)
        emb = freqs.to(dtype)
        return emb.sin(), emb.cos()

    def apply_rope(self, x, sin, cos):
        # x: [B, H, T, D]
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        x_rot = torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)
        return x_rot.flatten(-2)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv_proj(x).view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # [B, T, H, D]
        q = q.transpose(1, 2)  # [B, H, T, D]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        sin, cos = self._get_rope_sin_cos(T, x.device, x.dtype)
        q = self.apply_rope(q, sin, cos)
        k = self.apply_rope(k, sin, cos)

        # Use float32 for numerical stability in dot-product
        att = (q.float() @ k.float().transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.sparse_attn and T <= self.block_size and self._sparse_mask.numel() > 0:
            m = self._sparse_mask[:T, :T].to(att.device)
            att = att.masked_fill(~m.view(1, 1, T, T), float("-inf"))
        att = att - att.max(dim=-1, keepdim=True).values  # softmax stabilize
        att = F.softmax(att, dim=-1).type_as(q)  # back to original dtype
        if self.capture_attention:
            self._attn_weights = att.detach()
        att = self.attn_dropout(att)

        out = att @ v  # [B, H, T, D]
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.resid_dropout(self.out_proj(out))
        return out


class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout):
        super().__init__()
        self.feed = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.feed(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        n_embd,
        n_head,
        dropout,
        *,
        block_size: int,
        sparse_attn: bool = False,
        sparse_attn_window: int = 0,
        sparse_attn_mode: str = "sliding",
    ):
        super().__init__()
        self.ln1 = RMSNorm(n_embd)
        self.attn = MultiHeadAttentionWithRoPE(
            n_embd,
            n_head,
            dropout,
            block_size=block_size,
            sparse_attn=sparse_attn,
            sparse_attn_window=sparse_attn_window,
            sparse_attn_mode=sparse_attn_mode,
        )
        self.ln2 = RMSNorm(n_embd)
        self.ff = FeedForward(n_embd, dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


@dataclass
class ModelConfig:
    vocab_size: int = 67
    block_size: int = 256
    n_layer: int = 8
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.1
    bias: bool = False
    sparse_attn: bool = False
    sparse_attn_window: int = 0
    sparse_attn_mode: str = "sliding"


class FastAndSlow(nn.Module):
    """Shared-backbone action and world prediction model."""

    def __init__(
        self,
        config: ModelConfig,
        action_config: GPTConfig,
    ):
        super().__init__()
        self.config = config
        # Reuse GPT initialization, including tied embeddings and scaled residuals.
        # Optimizer state depends on this parameter registration order.
        initialized_gpt = GPT(action_config)
        self.fast_token_embedding = initialized_gpt.transformer.wte
        self.fast_position_embedding = initialized_gpt.transformer.wpe
        self.fast_dropout = initialized_gpt.transformer.drop
        self.shared_backbone = nn.Sequential(*initialized_gpt.transformer.h)
        self.fast_norm = initialized_gpt.transformer.ln_f
        self.lm_head = initialized_gpt.lm_head
        self.world_token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.world_dropout = nn.Dropout(config.dropout)
        self.world_position_encoding = RotationalPositionEmbedding(config.n_embd, "cpu")
        self.world_branch = nn.Sequential(
            *[
                TransformerBlock(
                    config.n_embd,
                    config.n_head,
                    config.dropout,
                    block_size=config.block_size,
                    sparse_attn=config.sparse_attn,
                    sparse_attn_window=config.sparse_attn_window,
                    sparse_attn_mode=config.sparse_attn_mode,
                )
                for _ in range(config.n_layer)
            ]
        )
        self.world_norm = RMSNorm(config.n_embd)

        self.world_head = nn.Linear(config.n_embd, config.vocab_size)

    def _encode_fast(self, idx):
        assert idx.size(1) <= self.fast_position_embedding.num_embeddings, (
            "Input exceeds the fast position embedding context size"
        )
        pos = torch.arange(idx.size(1), dtype=torch.long, device=idx.device)
        return self.fast_dropout(self.fast_token_embedding(idx) + self.fast_position_embedding(pos))

    def _encode_world(self, idx):
        return self.world_dropout(
            self.world_token_embedding(idx) + self.world_position_encoding(idx)
        )

    def _decode_world(self, hidden):
        return self.world_head(self.world_norm(self.world_branch(hidden)))

    def forward(self, idx, targets=None, *, joint=False):
        # Route joint training through forward so DDP/compile keep their hooks.
        if joint:
            return self.hybrid_forward(idx, targets)
        hidden = self.shared_backbone(self._encode_world(idx))
        return self._decode_world(hidden)

    def hybrid_forward(self, idx, targets=None):
        # Batch the two encodings through one shared set of backbone parameters.
        encoded = torch.cat([self._encode_world(idx), self._encode_fast(idx)], dim=0)
        world_hidden, fast_hidden = self.shared_backbone(encoded).split(idx.size(0), dim=0)
        world_logits = self._decode_world(world_hidden)
        action_logits = self.lm_head(self.fast_norm(fast_hidden))
        return world_logits, action_logits

    def fast_inference(self, idx, do_sample=True, return_probs=False):
        hidden = self.shared_backbone(self._encode_fast(idx))
        logits = self.lm_head(self.fast_norm(hidden)[:, [-1], :])[:, -1, :]

        mask = torch.ones_like(logits) * float("-inf")
        mask[:, :5] = logits[:, :5]
        masked_logits = mask

        probs = F.softmax(masked_logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        if return_probs:
            return idx_next.squeeze(-1), probs[:, :5]
        return idx_next.squeeze(-1)

    def slow_inference(self, idx, do_sample=True):
        logits = self(idx)
        logits = logits[:, 129, :]

        mask = torch.ones_like(logits) * float("-inf")
        mask[:, 45:50] = logits[:, 45:50]
        masked_logits = mask

        probs = F.softmax(masked_logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)
        return idx_next.squeeze(-1) - 45

    def half_dream(self, idx, pre_agents, do_sample=True, return_probs=False):
        if pre_agents is not None:
            positions = [130 + i * 10 for i in range(13)]
            mask = idx[:, positions] != 66
            idx_masked = idx.clone()
            idx_masked[:, positions] = torch.where(mask, pre_agents, idx[:, positions])
            idx = idx_masked
        else:
            idx = idx.clone()

        next_obs, actions = self.hybrid_forward(idx)
        next_obs, pre_greedy = self.sample_dream(next_obs, idx=idx, do_sample=do_sample)

        actions_logits = actions[:, -1, :]

        mask = torch.ones_like(actions_logits) * float("-inf")
        mask[:, :5] = actions_logits[:, :5]
        masked_logits = mask

        probs = F.softmax(masked_logits, dim=-1)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        actions = idx_next.squeeze(-1)

        if return_probs:
            return pre_greedy, actions, probs[:, :5], next_obs
        return pre_greedy, actions

    def sample_dream(self, logits, idx=None, do_sample=True):
        """Decode each token region by sampling or deterministic argmax."""
        B, T, D = logits.shape

        def select(probs):
            if do_sample:
                return torch.multinomial(probs, num_samples=1)
            return probs.argmax(dim=-1, keepdim=True)

        results = torch.full((B, T), 66, dtype=torch.long, device=logits.device)

        # costmap
        costmap_mask = logits[:, :121, :44]
        costmap_probs = F.softmax(costmap_mask, dim=-1)
        costmap_pred = select(costmap_probs.view(B * 121, -1))
        results[:, :121] = costmap_pred.view(B, 121)

        base = 121 + torch.arange(13).unsqueeze(1) * 10

        if idx is not None:
            # position: delta-based (5 classes)
            pos_indices = (torch.tensor([0, 1]).unsqueeze(0) + base).flatten().tolist()
            pos_logits = logits[:, pos_indices, :5]
            pos_probs = F.softmax(pos_logits, dim=-1)
            pos_delta = select(pos_probs.view(B * 26, -1)).view(B, 26) - 2
            current_pos = idx[:, pos_indices]
            next_pos = current_pos + pos_delta
            next_pos = next_pos.clamp(0, 43)
            next_pos[current_pos == 66] = 66
            results[:, pos_indices] = next_pos

            # goal: absolute (vocab 15-25 + padding)
            goal_indices = (torch.tensor([2, 3]).unsqueeze(0) + base).flatten().tolist()
            goal_mask = torch.cat(
                [logits[:, goal_indices, 15:26], logits[:, goal_indices, -1:]], dim=-1
            )
            goal_probs = F.softmax(goal_mask, dim=-1)
            goal_pred = select(goal_probs.view(B * 26, -1)) + 15
            goal_pred[goal_pred == 26] = 66
            results[:, goal_indices] = goal_pred.view(B, 26)
        else:
            # Absolute-coordinate sampling for start and goal tokens.
            start_goal_indices = (torch.tensor([0, 1, 2, 3]).unsqueeze(0) + base).flatten().tolist()
            start_goal_mask = torch.cat(
                [logits[:, start_goal_indices, 15:26], logits[:, start_goal_indices, -1:]], dim=-1
            )
            start_goal_probs = F.softmax(start_goal_mask, dim=-1)
            start_goal_pred = select(start_goal_probs.view(B * 52, -1)) + 15
            start_goal_pred[start_goal_pred == 26] = 66
            results[:, start_goal_indices] = start_goal_pred.view(B, 52)

        # history_actions
        history_actions_indices = (
            (torch.tensor([4, 5, 6, 7, 8]).unsqueeze(0) + base).flatten().tolist()
        )
        history_actions_mask = logits[:, history_actions_indices, 45:50]
        history_actions_probs = F.softmax(history_actions_mask, dim=-1)
        history_actions_pred = select(history_actions_probs.view(B * 65, -1)) + 45
        results[:, history_actions_indices] = history_actions_pred.reshape(B, 65)

        # greedy_action
        greedy_action_index = (torch.tensor([9]).unsqueeze(0) + base).flatten().tolist()
        greedy_action_mask = logits[:, greedy_action_index, 50:]
        greedy_action_probs = F.softmax(greedy_action_mask, dim=-1)
        greedy_action_pred = select(greedy_action_probs.view(B * 13, -1)) + 50
        results[:, greedy_action_index] = greedy_action_pred.reshape(B, 13)

        return results, greedy_action_pred.reshape(B, 13)

    def dreamer(self, idx, max_steps=2, do_sample=True, return_dream=False):
        results = []
        next_obs = idx
        dreams = []
        for i in tqdm(range(max_steps)):
            current_tokens = next_obs
            next_obs, actions = self.hybrid_forward(current_tokens)
            next_obs, _ = self.sample_dream(next_obs, idx=current_tokens, do_sample=do_sample)
            dreams.append(next_obs)
            actions_logits = actions[:, -1, :]

            mask = torch.ones_like(actions_logits) * float("-inf")
            mask[:, :5] = actions_logits[:, :5]
            masked_logits = mask

            probs = F.softmax(masked_logits, dim=-1)

            if do_sample:
                idx_next = torch.multinomial(probs, num_samples=1)

            else:
                _, idx_next = torch.topk(probs, k=1, dim=-1)
            actions = idx_next.squeeze(-1)
            results.append(actions.detach().cpu().numpy().tolist())

        if return_dream:
            return results, dreams
        return results

    def hybrid_inference(self, idx, do_sample=False):
        """
        Select an action using confidence and entropy from the two heads.
        Use the fast head when its batch-mean confidence exceeds 0.7.
        Otherwise, select the lower-entropy head separately for each sample.
        """
        logits_slow, logits_fast = self.hybrid_forward(idx)

        logits_slow = logits_slow[:, 129, 45:50]  # [B, C]
        logits_fast = logits_fast[:, -1, 0:5]  # [B, C]

        probs_fast = F.softmax(logits_fast, dim=-1)  # [B, C]
        confidence_fast = probs_fast.max(dim=-1).values  # [B]
        avg_confidence = confidence_fast.mean()

        if avg_confidence > 0.7:
            if do_sample:
                idx_next = torch.multinomial(probs_fast, num_samples=1)
            else:
                _, idx_next = torch.topk(probs_fast, k=1, dim=-1)
            return idx_next.squeeze(-1)

        probs, source_mask = self.pick_more_confident(logits_slow, logits_fast)

        if do_sample:
            idx_next = torch.multinomial(probs, num_samples=1)
        else:
            _, idx_next = torch.topk(probs, k=1, dim=-1)

        return idx_next.squeeze(-1)  # [B]

    def entropy(self, p):
        """Return entropy (...,) for probabilities p (..., C)."""
        p = torch.clamp(p, min=1e-8)  # Avoid log(0).
        return -(p * p.log()).sum(dim=-1)

    def pick_more_confident(self, logits_slow, logits_fast):
        """
        Select the lower-entropy head for each sample.

        Inputs:
            logits_slow: [B, C]
            logits_fast: [B, C]
        Outputs:
            probs: [B, C] — probabilities from the selected head
            source_mask: [B] — True selects the slow head; False selects the fast head
        """
        p_slow = F.softmax(logits_slow, dim=-1)  # [B, C]
        p_fast = F.softmax(logits_fast, dim=-1)  # [B, C]

        entropy_slow = self.entropy(p_slow)  # [B]
        entropy_fast = self.entropy(p_fast)  # [B]

        mask = entropy_slow < entropy_fast  # [B] — True selects the slow head
        mask_exp = mask.unsqueeze(-1)  # [B, 1]

        probs = torch.where(mask_exp, p_slow, p_fast)  # Select per sample.
        return probs, mask  # Return the selected head for diagnostics.


def FastAndSlow_S_2(
    *,
    sparse_attn: bool = False,
    sparse_attn_window: int = 0,
    sparse_attn_mode: str = "sliding",
):
    world_config = ModelConfig()
    world_config.n_layer = 4
    world_config.n_head = 4
    world_config.n_embd = 160
    world_config.block_size = 256
    world_config.sparse_attn = sparse_attn
    world_config.sparse_attn_window = sparse_attn_window
    world_config.sparse_attn_mode = sparse_attn_mode

    action_config = GPTConfig(dropout=world_config.dropout)
    action_config.n_layer = 5
    action_config.n_head = 5
    action_config.n_embd = 160
    action_config.block_size = 256
    action_config.sparse_attn = sparse_attn
    action_config.sparse_attn_window = sparse_attn_window
    model = FastAndSlow(world_config, action_config)

    return model


def FastAndSlow_B_2(
    *,
    sparse_attn: bool = False,
    sparse_attn_window: int = 0,
    sparse_attn_mode: str = "sliding",
):
    world_config = ModelConfig()
    world_config.n_layer = 8
    world_config.n_head = 8
    world_config.n_embd = 160
    world_config.block_size = 256
    world_config.sparse_attn = sparse_attn
    world_config.sparse_attn_window = sparse_attn_window
    world_config.sparse_attn_mode = sparse_attn_mode

    action_config = GPTConfig(dropout=world_config.dropout)
    action_config.n_layer = 5
    action_config.n_head = 5
    action_config.n_embd = 160
    action_config.block_size = 256
    action_config.sparse_attn = sparse_attn
    action_config.sparse_attn_window = sparse_attn_window
    model = FastAndSlow(world_config, action_config)

    return model
