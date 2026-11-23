"""ActionFlow: a parking model that gives the car's controls directly - the alternative to the spline parking model.

The condition is that of StructureFlow (model.py): the Fused BEV of the DDP front end with the target and the ego
state, read by 24 queries. What Flow Matching generates is not a path but the controls of the next 64 frames (6.4 s at
10 Hz): throttle, steer, brake, reverse, normalised per channel. A 1D U-Net over the 64 steps (the blocks of
StructureFlow's phase U-Net) generates them in 8 Euler steps, reading the BEV tokens and the queries at its middle.
No path tracker is needed: the car is given the generated controls frame by frame (action_server.py, benchmark.py
--actions).
"""
from dataclasses import fields, replace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from k4_flow_solver.model import TemporalBlock, time_embedding
from .config import DEFAULT
from .model import BevConditionEncoder

CHANNELS = ('throttle', 'steer', 'brake', 'reverse')


class ActionFlow(nn.Module):
    def __init__(self, cfg=DEFAULT, normalization=None, steps=64):
        super().__init__()
        self.cfg, self.steps = cfg, steps
        d, w = cfg.context_dim, cfg.temporal_channels
        self.encoder = BevConditionEncoder(cfg)
        self.time = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.pool = nn.MultiheadAttention(d, cfg.heads, batch_first=True)
        self.pool_query = nn.Parameter(torch.randn(1, 1, d)*.02)
        self.down1 = TemporalBlock(len(CHANNELS)+1, w, d)
        self.down2 = TemporalBlock(w, w*2, d)
        self.middle = TemporalBlock(w*2, w*3, d)
        self.bev_norm = nn.LayerNorm(w*3)
        self.bev_read = nn.MultiheadAttention(w*3, cfg.heads, kdim=d, vdim=d, batch_first=True)
        self.middle2 = TemporalBlock(w*3, w*3, d)
        self.up2 = TemporalBlock(w*5, w*2, d)
        self.up1 = TemporalBlock(w*3, w, d)
        self.head = nn.Conv1d(w, len(CHANNELS), 1)
        norm = normalization or dict(mean=[0.]*len(CHANNELS), std=[1.]*len(CHANNELS))
        self.register_buffer('action_mean', torch.tensor(norm['mean'], dtype=torch.float32))
        self.register_buffer('action_std', torch.tensor(norm['std'], dtype=torch.float32))
        self.register_buffer('u', torch.linspace(0, 1, steps))

    def encode(self, bev, target, ego):
        return self.encoder(bev, target, ego)

    def forward(self, y, t, context):
        """y [B, steps, 4] normalised controls at Flow time t [B] -> velocity [B, steps, 4]."""
        batch = len(y)
        query = self.pool_query.expand(batch, -1, -1)+self.time(time_embedding(t, self.cfg.context_dim))[:, None]
        condition = (query+self.pool(query, context.queries, context.queries, need_weights=False)[0])[:, 0]
        x = torch.cat((y.transpose(1, 2), self.u[None, None].expand(batch, 1, -1)), 1)
        a = self.down1(x, condition)
        b = self.down2(F.avg_pool1d(a, 2, ceil_mode=True), condition)
        c = self.middle(F.avg_pool1d(b, 2, ceil_mode=True), condition)
        place = time_embedding(torch.linspace(0, 1, c.shape[-1], device=y.device), c.shape[1])
        tokens = torch.cat((context.spatial, context.queries), 1)
        hidden = torch.cat((~context.valid, torch.zeros(context.queries.shape[:2], dtype=torch.bool, device=y.device)), 1)
        q = self.bev_norm(c.transpose(1, 2)+place[None])
        c = c+self.bev_read(q, tokens, tokens, key_padding_mask=hidden, need_weights=False)[0].transpose(1, 2)
        c = self.middle2(c, condition)
        d = self.up2(torch.cat((F.interpolate(c, size=b.shape[-1], mode='linear', align_corners=True), b), 1), condition)
        e = self.up1(torch.cat((F.interpolate(d, size=a.shape[-1], mode='linear', align_corners=True), a), 1), condition)
        return self.head(e).transpose(1, 2)

    @torch.no_grad()
    def sample(self, bev, target, ego, count=8, steps=None, generator=None):
        """One scene (batch 1) -> count control sequences [count, steps, 4] in CARLA units (not yet clipped)."""
        steps = steps or self.cfg.euler_steps
        context = self.encode(bev, target, ego).expand(count)
        y = torch.randn((count, self.steps, len(CHANNELS)), device=bev.device, generator=generator)
        for k in range(steps):
            t = torch.full((count,), k/steps, device=y.device)
            y = y+self(y, t, context)/steps
        return y*self.action_std+self.action_mean


def executed(controls, dead=.05):
    """[..., 4] generated controls -> what the car is given: throttle and brake in [0, 1] with the values below `dead`
    set to 0 (the expert's controls are exactly 0 there, and the DDP evaluator counts a car as parked only with no
    throttle), steer in [-1, 1], reverse when above 0.5."""
    out = np.array(controls, np.float64)
    for channel in (0, 2):
        out[..., channel] = np.where(out[..., channel] < dead, 0., np.minimum(out[..., channel], 1.))
    out[..., 1] = np.clip(out[..., 1], -1., 1.)
    out[..., 3] = (out[..., 3] > .5).astype(np.float64)
    return out


def medoid(samples, frames=10):
    """The sample nearest to the others over the first `frames` frames (1 s): the controls about to be driven."""
    head = np.asarray(samples)[:, :frames, :3].reshape(len(samples), -1)
    return int(np.abs(head[:, None]-head[None]).sum((1, 2)).argmin())


def load(path, device='cuda'):
    state = torch.load(path, map_location=device, weights_only=True)
    known = {field.name for field in fields(DEFAULT)}
    cfg = replace(DEFAULT, **{key: value for key, value in state['config'].items() if key in known})
    model = ActionFlow(cfg, state['normalization'], state['steps']).to(device).eval()
    model.load_state_dict(state['state'])
    return model
