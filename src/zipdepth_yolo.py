"""zipdepth_yolo.py - YOLO26n with its backbone (layers 0-10) replaced by the ZipDepth encoder.

Kept in a separate module so that checkpoints (which pickle the model) can be loaded from any notebook,
provided this file and the ZipDepth repo are on sys.path.

Variants: adapter depth (1 = Run B, 3 = Run B-A3) and neck/head initialisation (COCO = Run B, random = Run B-RN).
The 1-layer adapter keeps the exact module layout of Run B, so existing Run B checkpoints still load.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from zipdepth.model.architecture import create_model

YOLO_OUT_CH = (128, 128, 256)          # channels YOLO26n's neck expects at P3 / P4 / P5
REPLACED = range(0, 11)                # YOLO26n backbone layers (0-10) replaced by the encoder


class SafeBatchNorm2d(nn.BatchNorm2d):
    """BatchNorm2d that falls back to running statistics when a channel has a single value in training
    (the encoder's GlobalContextBlock applies BN to a pooled 1x1 vector, which fails for a final batch of 1 image)."""

    def forward(self, x):
        if self.training and x.numel() // x.shape[1] == 1:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


def conv_bn_silu(ci, co, k=1):
    """Conv(k x k, no bias) + BN + SiLU, flattened into a list so the 1-layer adapter keeps Run B's state_dict keys."""
    return [nn.Conv2d(ci, co, k, padding=k // 2, bias=False), nn.BatchNorm2d(co), nn.SiLU()]


def make_adapter(ci, co, n_layers=1):
    """1 layer: 1x1 (Run B). 2 layers: 1x1 -> 3x3. 3 layers: 1x1 -> 3x3 -> 1x1. Channels change in the first layer."""
    kernels = {1: [1], 2: [1, 3], 3: [1, 3, 1]}[n_layers]
    layers = []
    for i, k in enumerate(kernels):
        layers += conv_bn_silu(ci if i == 0 else co, co, k)
    return nn.Sequential(*layers)


class ZipDepthBackbone(nn.Module):
    """ImageNet-normalised ZipDepth encoder + Conv-BN-SiLU adapters. Input in [0, 1]; returns [P3, P4, P5]."""

    def __init__(self, ckpt=None, out_channels=YOLO_OUT_CH, adapter_layers=1):
        super().__init__()
        zd = create_model("base", upsample_unfold=True)
        if ckpt is not None:
            zd.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=True))
        self.encoder = zd.encoder                                   # decoder is discarded
        for m in self.encoder.modules():                            # same parameters, safe for batch-1 steps
            if type(m) is nn.BatchNorm2d:
                m.__class__ = SafeBatchNorm2d
        self.register_buffer("mean", zd.mean.clone())
        self.register_buffer("std", zd.std.clone())
        enc_ch = (self.encoder.stage2[0].out_ch, self.encoder.stage3[0].out_ch, self.encoder.stage4[0].out_ch)
        self.adapters = nn.ModuleList(make_adapter(ci, co, adapter_layers) for ci, co in zip(enc_ch, out_channels))

    def encode(self, x):
        """Raw encoder outputs (s2, s3', s4') at strides 8 / 16 / 32."""
        _, (_, s2, s3, s4) = self.encoder((x - self.mean) / self.std)
        return s2, s3, s4

    def forward(self, x):
        return [a(f) for a, f in zip(self.adapters, self.encode(x))]


class Pick(nn.Module):
    """Selects one tensor from the backbone's output list (keeps the neck's layer indices valid)."""

    def __init__(self, idx):
        super().__init__()
        self.idx = idx

    def forward(self, x):
        return x[self.idx]


class PassThrough(nn.Module):
    """Placeholder for a removed backbone layer; its output is never consumed."""

    def forward(self, x):
        return x


def _set_meta(m, i, f):
    m.i, m.f, m.type, m.np = i, f, type(m).__name__, sum(p.numel() for p in m.parameters())
    return m


def build_zipdepth_yolo(nc, names, zipdepth_ckpt=None, coco_weights="yolo26n.pt", adapter_layers=1, verbose=False):
    """YOLO26n (nc classes) with a ZipDepth backbone.

    zipdepth_ckpt=None gives a randomly initialised encoder; coco_weights=None gives a randomly initialised neck and
    head. Layer indices 0-10 are kept: layer 0 = ZipDepthBackbone, layers 4 / 6 / 10 pick P3 / P4 / P5 from it,
    the rest are placeholders.
    """
    model = DetectionModel("yolo26n.yaml", nc=nc, verbose=False)
    model.names = names
    if coco_weights:                                                # same transfer as run A (incl. cls_remap)
        from ultralytics import YOLO
        model.load(YOLO(coco_weights).model, verbose=verbose)
    layers = list(model.model)
    layers[0] = _set_meta(ZipDepthBackbone(zipdepth_ckpt, adapter_layers=adapter_layers), 0, -1)
    for i in REPLACED[1:]:
        layers[i] = _set_meta(PassThrough(), i, -1)
    layers[4] = _set_meta(Pick(0), 4, 0)
    layers[6] = _set_meta(Pick(1), 6, 0)
    layers[10] = _set_meta(Pick(2), 10, 0)
    model.model = nn.Sequential(*layers)
    model.save = sorted(set(model.save) | {0})                     # keep layer 0's output for layers 4 / 6 / 10
    model.yaml["backbone_override"] = (f"ZipDepth-base encoder ({'pretrained' if zipdepth_ckpt else 'random'}), "
                                       f"{adapter_layers}-layer adapters, neck/head {'COCO' if coco_weights else 'random'}")
    return model


class ZipDepthTrainer(DetectionTrainer):
    """DetectionTrainer that builds the ZipDepth model; on resume it restores weights from the checkpoint model.
    Set the class attributes before training."""

    zipdepth_ckpt = None                                             # None = random encoder
    adapter_layers = 1                                               # 1 = Run B, 3 = Run B-A3
    coco_neck_head = True                                            # False = random neck and head (Run B-RN)

    def get_model(self, cfg=None, weights=None, verbose=True):
        coco = "yolo26n.pt" if (weights is None and self.coco_neck_head) else None
        model = build_zipdepth_yolo(self.data["nc"], self.data["names"], self.zipdepth_ckpt, coco_weights=coco,
                                    adapter_layers=self.adapter_layers, verbose=verbose)
        if weights is not None:                                      # resume / fine-tune from a saved run
            model.load_state_dict(weights.float().state_dict())
        return model


def fuse_zipdepth_yolo(model):
    """Inference fusion: RepVGG blocks of the encoder + Ultralytics Conv/BN fusion of neck and head."""
    model.model[0].encoder.fuse()
    return model.fuse(verbose=False)
