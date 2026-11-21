"""Camera and BEV geometry of the training-free front end.

Frames
  optical  X right, Y down, Z forward (pinhole, full 400x300 image, no crop)
  ego      CARLA vehicle frame: x forward, y right, z up, origin on the ground under the car centre
  BEV      200x200 cells of 0.1 m, row = 100 + 10 x, col = 100 + 10 y, the grid of the DDP segmentation
           and of `hits` in tools/eval_structure_flow.py, so the planner reads it unchanged.
"""
import numpy as np

from ddp_frontend import rig

CAMERAS, MOUNTS = rig.CAMERAS, rig.MOUNTS
WIDTH, HEIGHT, FOV = rig.WIDTH, rig.HEIGHT, rig.FOV
BEV_SIZE, BEV_RES = 200, .1
BEV_HALF = BEV_SIZE*BEV_RES/2          # 10 m
GROUND_Z = 0.                          # vehicle origin of vehicle.tesla.model3 sits on the ground
EGO_HALF = (2.45, 1.05)                # half length, half width of the ego body, never an obstacle
OPTICAL_FROM_CARLA = rig.CAM_FROM_VEHICLE_AXES     # carla camera axes (x fwd, y right, z up) -> optical


def intrinsics_full():
    focal = WIDTH/(2*np.tan(np.radians(FOV)/2))
    return np.array([[focal, 0, WIDTH/2], [0, focal, HEIGHT/2], [0, 0, 1.]])


def ego_from_optical(name):
    """4x4 matrix taking optical-frame points of camera `name` to the ego frame."""
    x, y, z, yaw = MOUNTS[name]
    return rig.carla_matrix(x, y, z, yaw)@np.linalg.inv(OPTICAL_FROM_CARLA)


K = intrinsics_full()
K_INV = np.linalg.inv(K)
EGO_FROM_OPTICAL = np.stack([ego_from_optical(name) for name in CAMERAS])      # [6, 4, 4]
OPTICAL_FROM_EGO = np.linalg.inv(EGO_FROM_OPTICAL)


def pixel_rays(stride=1):
    """Optical rays with unit Z for the pixel centres, [H/stride, W/stride, 3]."""
    v, u = np.mgrid[stride/2-.5:HEIGHT:stride, stride/2-.5:WIDTH:stride]
    return np.stack((u, v, np.ones_like(u)), -1)@K_INV.T


def backproject(depth, camera, stride=1):
    """z-depth map [H, W] (metres along the optical axis) of camera index -> ego points [H/s, W/s, 3]."""
    rays = pixel_rays(stride)
    depth = depth[::stride, ::stride] if stride > 1 else depth
    points = rays*depth[..., None]
    matrix = EGO_FROM_OPTICAL[camera]
    return points@matrix[:3, :3].T+matrix[:3, 3]


def ground_depth(camera, stride=1, ground=GROUND_Z):
    """z-depth at which each pixel ray meets the flat ground, inf above the horizon. [H/s, W/s]."""
    rays = pixel_rays(stride)
    matrix = EGO_FROM_OPTICAL[camera]
    direction_z = rays@matrix[2, :3]
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = (ground-matrix[2, 3])/direction_z
    return np.where(scale > 0, scale, np.inf)


def ground_point(u, v, camera, ground=GROUND_Z):
    """Ego point where the ray of pixel (u, v) meets the ground, or None above the horizon."""
    ray = K_INV@np.array([u, v, 1.])
    matrix = EGO_FROM_OPTICAL[camera]
    direction = matrix[:3, :3]@ray
    if direction[2] >= -1e-6:
        return None
    scale = (ground-matrix[2, 3])/direction[2]
    return matrix[:3, 3]+scale*direction


def height_point(u, v, camera, height):
    """Ego point under where the ray of pixel (u, v) reaches `height` above the ground (a head top), or None."""
    ray = K_INV@np.array([u, v, 1.])
    matrix = EGO_FROM_OPTICAL[camera]
    direction = matrix[:3, :3]@ray
    rise = GROUND_Z+height-matrix[2, 3]
    if abs(direction[2]) < 1e-6 or rise/direction[2] <= 0:
        return None
    point = matrix[:3, 3]+rise/direction[2]*direction
    point[2] = GROUND_Z
    return point


def project(points, camera):
    """Ego points [N, 3] -> pixels [N, 2] and z-depth [N] of camera index."""
    matrix = OPTICAL_FROM_EGO[camera]
    optical = points@matrix[:3, :3].T+matrix[:3, 3]
    with np.errstate(divide='ignore', invalid='ignore'):
        pixels = optical[:, :2]/optical[:, 2:3]*K[0, 0]+K[:2, 2]
    return pixels, optical[:, 2]


def to_cells(xy):
    """Ego x, y [N, 2] -> integer (row, col) [N, 2] and a mask of cells inside the grid."""
    cells = np.floor((np.asarray(xy)+BEV_HALF)/BEV_RES).astype(int)
    inside = ((cells >= 0) & (cells < BEV_SIZE)).all(-1)
    return cells, inside


def raster_points(xy, minimum=1):
    """Occupancy of ego points [N, 2]: cells holding at least `minimum` points."""
    cells, inside = to_cells(xy)
    counts = np.zeros((BEV_SIZE, BEV_SIZE), np.int32)
    np.add.at(counts, (cells[inside, 0], cells[inside, 1]), 1)
    return counts >= minimum


def raster_polygon(mask, corners):
    """Fill an ego-frame polygon [N, 2] (x, y) into a BEV mask in place."""
    import cv2
    pts = np.stack(((corners[:, 1]+BEV_HALF)/BEV_RES, (corners[:, 0]+BEV_HALF)/BEV_RES), -1)    # (col, row)
    layer = mask.astype(np.uint8)
    cv2.fillPoly(layer, [np.round(pts-.5).astype(np.int32)], 1)
    mask |= layer.astype(bool)
    return mask


def box_corners(centre, size, yaw):
    """Rectangle (length along yaw, width) around ego centre (x, y) -> 4 corners [4, 2]. yaw in radians."""
    c, s = np.cos(yaw), np.sin(yaw)
    half = np.array([[1, 1], [1, -1], [-1, -1], [-1, 1]])*np.asarray(size)/2
    rotation = np.array([[c, -s], [s, c]])
    return half@rotation.T+np.asarray(centre)


def target_slot(target, size=(5.5, 3.1)):
    """Parking slot of the DDP labels (55 x 31 cells) around target (x m, y m, yaw deg) as a BEV mask."""
    mask = np.zeros((BEV_SIZE, BEV_SIZE), bool)
    return raster_polygon(mask, box_corners(target[:2], size, np.radians(target[2])))


def ego_body():
    xs, ys = EGO_HALF
    mask = np.zeros((BEV_SIZE, BEV_SIZE), bool)
    return raster_polygon(mask, box_corners((0, 0), (2*xs, 2*ys), 0.))
