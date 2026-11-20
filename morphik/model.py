"""The tokenized Transformer and adaptive LayerNorm flow head from Figures 3–4."""

from dataclasses import dataclass
from math import log

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import IKBatch
from .geometry import MAX_DOF, RobotBatch, canonical_quaternion
from .kinematics import forward_kinematics


@dataclass(frozen=True)
class ModelConfig:
    width: int = 256
    attention_heads: int = 8
    encoder_blocks: int = 4
    flow_width: int = 512
    flow_blocks: int = 6
    time_width: int = 64


class ReZeroBlock(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_out = nn.Linear(width, width)
        self.ffn = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))
        self.attention_scale = nn.Parameter(torch.zeros(()))
        self.ffn_scale = nn.Parameter(torch.zeros(()))

    def forward(self, tokens: Tensor, valid: Tensor) -> Tensor:
        batch_size, length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(batch_size, length, 3, self.heads, width // self.heads)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind()
        attention = F.scaled_dot_product_attention(
            query, key, value, attn_mask=valid[:, None, None, :]
        )
        attention = attention.transpose(1, 2).reshape(batch_size, length, width)
        tokens = tokens + self.attention_scale * self.attention_out(attention)
        return tokens + self.ffn_scale * self.ffn(tokens)


class AdaLNBlock(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Linear(width, 2 * width)
        self.mlp = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, hidden: Tensor, condition: Tensor) -> Tensor:
        scale, shift = self.modulation(condition).chunk(2, dim=-1)
        return hidden + self.mlp(self.norm(hidden) * (1 + scale) + shift)


class MorphIK(nn.Module):
    def __init__(self, config: ModelConfig = ModelConfig()):
        super().__init__()
        self.config = config
        width = config.width
        self.geometry_projection = nn.Linear(10, width)
        self.tool_projection = nn.Linear(7, width)
        self.target_projection = nn.Linear(7, width)
        self.encoder = nn.ModuleList(
            ReZeroBlock(width, config.attention_heads) for _ in range(config.encoder_blocks)
        )
        self.register_buffer("chain_index", torch.arange(MAX_DOF), persistent=False)
        self.register_buffer(
            "time_frequencies",
            1000 * torch.exp(torch.linspace(0, -log(10_000), config.time_width // 2)),
            persistent=False,
        )
        hidden = config.flow_width
        self.context_projection = nn.Linear((MAX_DOF + 2) * width + 7, hidden)
        self.time_projection = nn.Sequential(nn.Linear(config.time_width, hidden), nn.SiLU(),
                                             nn.Linear(hidden, hidden))
        self.state_projection = nn.Linear(MAX_DOF, hidden)
        self.flow = nn.ModuleList(AdaLNBlock(hidden) for _ in range(config.flow_blocks))
        self.readout = nn.Linear(hidden, MAX_DOF)

    def encode(self, robots: RobotBatch, position: Tensor, orientation: Tensor) -> Tensor:
        actual_joints = robots.joint_mask
        limits = robots.half_ranges[..., None]
        chain_position = self.chain_index / (actual_joints.sum(dim=-1, keepdim=True) - 1).clamp_min(1)
        geometry = torch.cat(
            (robots.points, robots.axes, -limits, limits, robots.radii[..., None],
             chain_position[..., None]),
            dim=-1,
        )
        tool = torch.cat((robots.tool_position, robots.tool_quaternion), dim=-1)
        target = torch.cat((position, canonical_quaternion(orientation)), dim=-1)
        tokens = torch.cat(
            (self.geometry_projection(geometry), self.tool_projection(tool)[:, None],
             self.target_projection(target)[:, None]),
            dim=1,
        )
        valid = F.pad(actual_joints, (0, 2), value=True)
        for block in self.encoder:
            tokens = block(tokens, valid)
        tokens = tokens.masked_fill(~valid[..., None], 0)
        return torch.cat((tokens.flatten(1), target), dim=-1)

    def velocity(
        self, state: Tensor, time: Tensor, projected_context: Tensor, joint_mask: Tensor
    ) -> Tensor:
        angles = time * self.time_frequencies
        time_embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        condition = F.silu(projected_context + self.time_projection(time_embedding))
        hidden = self.state_projection(state.masked_fill(~joint_mask, 0))
        for block in self.flow:
            hidden = block(hidden, condition)
        return self.readout(hidden).masked_fill(~joint_mask, 0)

    def forward(
        self, robots: RobotBatch, position: Tensor, orientation: Tensor, state: Tensor, time: Tensor
    ) -> Tensor:
        context = self.context_projection(self.encode(robots, position, orientation))
        return self.velocity(state, time, context, robots.joint_mask)


def flow_matching_loss(
    model: MorphIK, batch: IKBatch, generator: torch.Generator
) -> tuple[Tensor, dict[str, Tensor]]:
    """Equations (1)–(8) with the weight ratios in Section III-E."""
    q = batch.joints
    time = torch.rand((len(q), 1), dtype=q.dtype, device=q.device, generator=generator)
    noise = torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator)
    active = batch.robots.joint_mask
    noise = noise.masked_fill(~active, 0)
    state = torch.lerp(noise, q, time)
    velocity = model(batch.robots, batch.position, batch.orientation, state, time).to(q.dtype)
    flow = (velocity - (q - noise)).square().sum(dim=-1) / active.sum(dim=-1)

    # One differentiable step estimates the integration endpoint for the FK loss.
    endpoint = state + (1 - time) * velocity
    position, orientation = forward_kinematics(batch.robots, endpoint)
    position_loss = F.smooth_l1_loss(position, batch.position, reduction="none").mean(dim=-1)
    positive = F.smooth_l1_loss(orientation, batch.orientation, reduction="none").mean(dim=-1)
    negative = F.smooth_l1_loss(-orientation, batch.orientation, reduction="none").mean(dim=-1)
    orientation_loss = torch.minimum(positive, negative)
    time = time.squeeze(-1)
    gate = torch.where(time >= 0.5, time.square(), 0.0)
    fk_loss = gate * (5 * position_loss + orientation_loss)
    loss = (flow + 6 * fk_loss).mean()
    return loss, {"flow": flow.mean().detach(), "fk": fk_loss.mean().detach()}
