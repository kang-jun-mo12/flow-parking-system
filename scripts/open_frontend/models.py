"""Pretrained models used as they are: an open-vocabulary detector and a metric depth network.

Neither is trained or fine-tuned here. Both are loaded lazily so the geometry and the BEV
composition run (and are tested) without them.

Detector output per camera: float array [N, 6] = x1, y1, x2, y2 (pixels of the 400x300 image), score, category id.
Depth output: float array [6, 300, 400], z-depth in metres along each optical axis.
"""
from pathlib import Path

import numpy as np

# Category ids of the BEV; text prompts are what YOLO-World is asked for.
VEHICLE, PERSON, OBSTACLE = 0, 1, 2
CATEGORY_NAMES = ('vehicle', 'person', 'obstacle')
PROMPTS = {
    'car': VEHICLE, 'truck': VEHICLE, 'van': VEHICLE, 'bus': VEHICLE, 'motorcycle': VEHICLE, 'bicycle': VEHICLE,
    'person': PERSON, 'pedestrian': PERSON,
    'traffic cone': OBSTACLE, 'barrier': OBSTACLE, 'shopping cart': OBSTACLE, 'box': OBSTACLE,
}
# Footprint priors (length, width) in metres; observed points only show the visible faces.
PRIOR_SIZE = {VEHICLE: (4.5, 1.9), PERSON: (.5, .5), OBSTACLE: (.5, .5)}
# Largest footprint a box may grow to (a delivery truck); beyond it the points hold more than one thing.
MAX_SIZE = {VEHICLE: (7.5, 2.7), PERSON: (1., 1.), OBSTACLE: (1.5, 1.5)}
PERSON_HEIGHT = 1.75     # metres, for people whose feet leave the image
# Categories that may move: tracked for velocity and given a predicted sweep.
MOVABLE = (VEHICLE, PERSON)


# Open vocabulary for what CARLA can place on a parking lot (static.prop.*, vehicles, walkers). Encoded once
# with CLIP ViT-B/32 and stored in yolov8s-worldv2-carla.pt, so running it needs no text encoder.
CARLA_PROMPTS = {
    'car': VEHICLE, 'truck': VEHICLE, 'van': VEHICLE, 'bus': VEHICLE, 'motorcycle': VEHICLE, 'bicycle': VEHICLE,
    'person': PERSON,
    'traffic cone': OBSTACLE, 'barrier': OBSTACLE, 'barrel': OBSTACLE, 'trash can': OBSTACLE,
    'garbage bag': OBSTACLE, 'cardboard box': OBSTACLE, 'bench': OBSTACLE, 'chair': OBSTACLE, 'table': OBSTACLE,
    'potted plant': OBSTACLE, 'shopping cart': OBSTACLE, 'suitcase': OBSTACLE, 'vending machine': OBSTACLE,
    'atm': OBSTACLE, 'kiosk': OBSTACLE, 'mailbox': OBSTACLE, 'street sign': OBSTACLE, 'warning sign': OBSTACLE,
    'bus stop': OBSTACLE, 'fountain': OBSTACLE, 'container': OBSTACLE, 'dog house': OBSTACLE, 'slide': OBSTACLE,
    'swing': OBSTACLE, 'trampoline': OBSTACLE, 'lamp post': OBSTACLE, 'billboard': OBSTACLE, 'garden gnome': OBSTACLE,
    'bag': OBSTACLE, 'guitar case': OBSTACLE, 'helmet': OBSTACLE, 'watering can': OBSTACLE, 'barbecue grill': OBSTACLE,
}
CARLA_WEIGHTS = str(Path(__file__).resolve().parents[2]/'weights'/'yolov8s-worldv2-carla.pt')

# The vocabulary stored in yolov8s-worldv2.pt (COCO 80) needs no text encoder; these names are kept.
COCO_VOCABULARY = {
    'car': VEHICLE, 'truck': VEHICLE, 'bus': VEHICLE, 'motorcycle': VEHICLE, 'bicycle': VEHICLE, 'train': VEHICLE,
    'person': PERSON, 'dog': PERSON, 'cat': PERSON,
    'fire hydrant': OBSTACLE, 'stop sign': OBSTACLE, 'parking meter': OBSTACLE, 'bench': OBSTACLE,
    'suitcase': OBSTACLE, 'chair': OBSTACLE, 'potted plant': OBSTACLE,
}


class YoloWorld:
    """ultralytics YOLO-World (open vocabulary, zero-shot).

    prompts=None keeps the stored COCO vocabulary (no extra download). A dict such as PROMPTS sets new classes,
    which runs the CLIP ViT-B/32 text encoder once (ultralytics fetches it on first use)."""

    def __init__(self, weights='C:/CARLA/open_frontend/yolov8s-worldv2.pt', prompts=None, score=.25, device='cuda',
                 image_size=416):
        from ultralytics import YOLOWorld
        self.model = YOLOWorld(weights)
        if prompts:
            stored = [self.model.names[i] for i in range(len(self.model.names))]
            if stored != list(prompts):         # weights saved with this vocabulary skip the text encoder
                self.model.set_classes(list(prompts))
            vocabulary = prompts
        else:
            vocabulary = COCO_VOCABULARY
        names = self.model.names        # class index -> name
        self.category = np.array([vocabulary.get(names[i], -1) for i in range(len(names))])
        self.score, self.device, self.image_size = score, device, image_size

    def __call__(self, images):
        """images [6, H, W, 3] RGB uint8 -> list of 6 arrays [N, 6]."""
        results = self.model.predict([image[..., ::-1] for image in images], conf=self.score, imgsz=self.image_size,
                                     device=self.device, verbose=False)      # ultralytics takes BGR numpy
        out = []
        for result in results:
            boxes = result.boxes
            if boxes is None or not len(boxes):
                out.append(np.zeros((0, 6), np.float32))
                continue
            xyxy, score = boxes.xyxy.cpu().numpy(), boxes.conf.cpu().numpy()
            category = self.category[boxes.cls.cpu().numpy().astype(int)]
            keep = category >= 0           # classes outside the vocabulary (airplane, cup, ...) are dropped
            out.append(np.concatenate((xyxy[keep], score[keep, None], category[keep, None]), 1).astype(np.float32))
        return out


class DepthAnythingMetric:
    """Depth Anything V2, metric outdoor head (trained on Virtual KITTI), through transformers."""

    def __init__(self, name='depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf', device='cuda'):
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        self.torch, self.device = torch, device
        self.processor = AutoImageProcessor.from_pretrained(name)
        self.model = AutoModelForDepthEstimation.from_pretrained(name).to(device).eval()

    def __call__(self, images):
        torch = self.torch
        with torch.no_grad():
            inputs = self.processor(images=list(images), return_tensors='pt').to(self.device)
            depth = self.model(**inputs).predicted_depth                      # [6, h, w]
            depth = torch.nn.functional.interpolate(depth[:, None], size=images.shape[1:3], mode='bilinear',
                                                    align_corners=False)[:, 0]
        return depth.float().cpu().numpy()


class GroundPlaneDepth:
    """No network: every pixel below the horizon gets the depth of the flat ground. Only for tests and as the
    floor of the scale check, it cannot see anything standing on the ground."""

    def __call__(self, images):
        from .geometry import ground_depth
        return np.stack([np.minimum(ground_depth(camera), 80.) for camera in range(len(images))])
