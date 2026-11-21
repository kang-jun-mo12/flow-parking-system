"""Training-free front end: YOLO-World + metric depth -> BEV, in the grid and classes of the DDP segmentation."""
from .frontend import Config, OpenFrontend, fuse_with_ddp, LAYERS
