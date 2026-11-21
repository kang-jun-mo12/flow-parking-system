"""Metric depth models the open front end can load.

Every model is called as models.DepthAnythingMetric is: 6 RGB uint8 images [6, H, W, 3] -> z-depth [6, H, W] in metres
(along the optical axis). make_depth(name):
  da2s / da2b / da2l   Depth Anything V2 Metric Outdoor Small / Base / Large (transformers; no camera input - on this
                       100 degree rig the depth comes out about 4x too far and the ground check rescales it)
  unidepth-s           UniDepth V2 ViT-S given the rig's intrinsics; unidepth-s-nocam: the camera estimated by the model
  moge2-s              MoGe-2 ViT-S given the rig's horizontal field of view; moge2-s-nocam: estimated by the model
The third-party code is put on sys.path only here: UniDepth comes with this repository (third_party/UniDepth), MoGe
would be a github clone beside it (third_party/MoGe, not included). The pure-python packages they need (einops, timm)
are installed, or kept in the folder OPEN_FRONTEND_PACKAGES names.
"""
import os
from pathlib import Path
import sys
import types

import numpy as np

from . import geometry as G
from .models import DepthAnythingMetric

THIRD_PARTY = str(Path(__file__).resolve().parents[2]/'third_party')
EXTRA_PACKAGES = os.environ.get('OPEN_FRONTEND_PACKAGES', THIRD_PARTY+'/extra_pkgs')
FAR = 80.                   # metres for pixels a model gives no depth (sky)


def _paths(repo):
    for path in (EXTRA_PACKAGES, '{}/{}'.format(THIRD_PARTY, repo)):
        if path not in sys.path:
            sys.path.append(path)


class UniDepthV2:
    """UniDepth V2 (lpiccinelli-eth/UniDepth, CC BY-NC 4.0). camera=True: the pinhole intrinsics of the rig are given."""

    def __init__(self, name='lpiccinelli/unidepth-v2-vits14', device='cuda', camera=True):
        import torch
        _paths('UniDepth')
        sys.modules.setdefault('wandb', types.ModuleType('wandb'))      # imported by its training utilities only
        from unidepth.models import UniDepthV2 as Model
        self.torch, self.device = torch, device
        self.model = Model.from_pretrained(name).to(device).eval()
        self.model.interpolation_mode = 'bilinear'
        self.intrinsics = torch.tensor(G.K, dtype=torch.float32) if camera else None

    def __call__(self, images):
        torch = self.torch
        rgb = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.float16, enabled=self.device != 'cpu'):
            # half precision: faster for the 6 images, the same depth
            if self.intrinsics is None:
                depth = self.model.infer(rgb)['depth'][:, 0]
            else:       # one image at a time: infer() builds one camera for the batch (a batch of K fails in its reshape)
                depth = torch.cat([self.model.infer(image[None], self.intrinsics)['depth'][:, 0] for image in rgb])
        return depth.float().cpu().numpy()


class MoGe2:
    """MoGe-2 (microsoft/MoGe, MIT): metric point map; fov=True: the rig's horizontal field of view is given."""

    def __init__(self, name='Ruicheng/moge-2-vits-normal', device='cuda', fov=True):
        import torch
        _paths('MoGe')
        from moge.model.v2 import MoGeModel
        self.torch, self.device = torch, device
        self.model = MoGeModel.from_pretrained(name).to(device).eval()
        self.fov = float(G.FOV) if fov else None
        self.fp16 = device != 'cpu'

    def __call__(self, images):
        torch = self.torch
        rgb = torch.from_numpy(np.ascontiguousarray(images)).permute(0, 3, 1, 2).float().div(255.).to(self.device)
        with torch.no_grad():
            out = self.model.infer(rgb, fov_x=self.fov, use_fp16=self.fp16)
        depth = out['depth'].float().cpu().numpy()
        return np.where(np.isfinite(depth), depth, FAR).astype(np.float32)


NAMES = dict(da2s='Small', da2b='Base', da2l='Large')


def make_depth(name='da2s', device='cuda'):
    if name in NAMES:
        return DepthAnythingMetric('depth-anything/Depth-Anything-V2-Metric-Outdoor-{}-hf'.format(NAMES[name]), device)
    if name.startswith('unidepth-'):
        size = name.split('-')[1]
        return UniDepthV2('lpiccinelli/unidepth-v2-vit{}14'.format(size), device, camera=not name.endswith('-nocam'))
    if name.startswith('moge2-'):
        size = name.split('-')[1]
        return MoGe2('Ruicheng/moge-2-vit{}-normal'.format(size), device, fov=not name.endswith('-nocam'))
    raise ValueError(name)
