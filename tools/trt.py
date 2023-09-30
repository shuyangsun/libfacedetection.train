#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import argparse
import os
import cv2
import shutil
import time

import tensorrt as trt
import numpy as np
import torch
from torch2trt import torch2trt

from mmcv import Config
from mmcv.runner import load_checkpoint

from mmdet.models import build_detector

from typing import List, Tuple


def make_parser():
    parser = argparse.ArgumentParser(description="YuNet TensorRT Deploy")
    parser.add_argument("config", help="test config file path")
    parser.add_argument("checkpoint", help="checkpoint file")
    parser.add_argument(
        "--out", type=str, default="./work_dirs/", help="trt model output directory"
    )
    parser.add_argument(
        "-w", "--workspace", type=int, default=32, help="max workspace size in detect"
    )
    parser.add_argument(
        "-b", "--batch", type=int, default=1, help="max batch size in detect"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cuda:0", help="cuda device"
    )
    parser.add_argument(
        "-s", "--samples", type=str, help="path to sample input directory"
    )
    parser.add_argument(
        "-i",
        "--iters",
        type=int,
        default=1,
        help="number of frames for benchmark, rounded up to nearest multiple of batch size",
    )
    parser.add_argument(
        "--score_thresh", type=float, default=0.5, help="score threshold"
    )
    parser.add_argument("--nms_thresh", type=float, default=0.45, help="nms threshold")
    return parser


def preproc(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3), dtype=np.uint8) * 114
    else:
        padded_img = np.ones(input_size, dtype=np.uint8) * 114

    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(np.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img

    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def prepare_samples(
    img_dir: str, input_size: Tuple[int, int], num_samples: int
) -> torch.Tensor:
    res: List[torch.Tensor] = list()
    for root, _, files in os.walk(img_dir):
        for f in files:
            f_lower = f.lower()
            if (
                not f_lower.endswith(".png")
                or f_lower.endswith(".jpg")
                or f_lower.endswith(".jpeg")
            ):
                continue
            img = cv2.imread(os.path.join(root, f))
            img, _ = preproc(img, input_size)
            img = torch.from_numpy(img).unsqueeze(0).half().to("cuda:0")
            res.append(img)
            if len(res) >= num_samples:
                break
    i = 0
    while len(res) < num_samples:
        res.append(res[i])
        i += 1
    return torch.cat(res)


def resize_img(img, mode):
    if mode == "ORIGIN":
        det_img, det_scale = img, 1.0
    elif mode == "AUTO":
        assign_h = ((img.shape[0] - 1) & (-32)) + 32
        assign_w = ((img.shape[1] - 1) & (-32)) + 32
        det_img = np.zeros((assign_h, assign_w, 3), dtype=np.uint8)
        det_img[: img.shape[0], : img.shape[1], :] = img
        det_scale = 1.0
    else:
        if mode == "VGA":
            input_size = (640, 480)
        else:
            input_size = list(map(int, mode.split(",")))
        assert len(input_size) == 2
        x, y = max(input_size), min(input_size)
        if img.shape[1] > img.shape[0]:
            input_size = (x, y)
        else:
            input_size = (y, x)
        im_ratio = float(img.shape[0]) / img.shape[1]
        model_ratio = float(input_size[1]) / input_size[0]
        if im_ratio > model_ratio:
            new_height = input_size[1]
            new_width = int(new_height / im_ratio)
        else:
            new_width = input_size[0]
            new_height = int(new_width * im_ratio)
        det_scale = float(new_height) / img.shape[0]
        resized_img = cv2.resize(img, (new_width, new_height))
        det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
        det_img[:new_height, :new_width, :] = resized_img

    return det_img, det_scale


@torch.no_grad()
def main():
    args = make_parser().parse_args()
    with torch.cuda.device(args.device):
        cfg = Config.fromfile(args.config)
        if cfg.get("custom_imports", None):
            from mmcv.utils import import_modules_from_strings

            import_modules_from_strings(**cfg["custom_imports"])
        # set cudnn_benchmark
        if cfg.get("cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        if cfg.model.get("neck"):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get("rfp_backbone"):
                        if neck_cfg.rfp_backbone.get("pretrained"):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get("rfp_backbone"):
                if cfg.model.neck.rfp_backbone.get("pretrained"):
                    cfg.model.neck.rfp_backbone.pretrained = None

        # in case the test dataset is concatenated
        if isinstance(cfg.data.test, dict):
            cfg.data.test.test_mode = True
        elif isinstance(cfg.data.test, list):
            for ds_cfg in cfg.data.test:
                ds_cfg.test_mode = True

        cfg.model.test_cfg.score_thr = args.score_thresh
        cfg.model.test_cfg.nms.iou_threshold = args.nms_thresh

        model = build_detector(cfg.model, train_cfg=None, test_cfg=None)
        checkpoint = load_checkpoint(model, args.checkpoint, map_location="cpu")

        if "CLASSES" in checkpoint["meta"]:
            model.CLASSES = checkpoint["meta"]["CLASSES"]
        else:
            model.CLASSES = None
        model = model.eval().half().to(args.device)

        inputs = [
            torch.ones(
                1,
                3,
                640,
                640,
                dtype=torch.float16,
                device=args.device,
            )
        ]
        while len(inputs) < args.batch:
            inputs.append(inputs[0])
        if args.samples is not None:
            inputs = [prepare_samples(args.samples, (640, 640), args.batch)]

        num_frames = int((((args.iters - 1) // args.batch) + 1) * args.batch)
        start = time.time()
        for _ in range(num_frames // args.batch):
            with torch.no_grad():
                # TODO: use feature_test as forward instead of simple_test, which requires img_metas to decode to bbox.
                pred = model(inputs[0])

        print(
            "PyTorch model fps (avg of {num} samples): {fps:.1f}".format(
                num=num_frames, fps=num_frames / (time.time() - start)
            )
        )

        model_trt = torch2trt(
            model,
            inputs,
            fp16_mode=True,
            log_level=trt.Logger.INFO,
            max_workspace_size=(1 << args.workspace),
            max_batch_size=args.batch,
        )

        # model(inputs[0]) # populate model.head
        start = time.time()
        for _ in range(num_frames // args.batch):
            pred = model_trt(inputs[0])
            # model.head.decode_outputs(pred, dtype=torch.float16, device="cuda:0")
        print(
            "TensorRT model fps (avg of {num} samples): {fps:.1f}".format(
                num=num_frames, fps=num_frames / (time.time() - start)
            )
        )

        torch.save(
            model_trt.state_dict(),
            os.path.join(args.out, f"yunet_n_trt_b{args.batch}.pth"),
        )

        print("Converted TensorRT model done.")
        engine_file = os.path.join(args.out, f"yunet_n_trt_b{args.batch}.engine")
        with open(engine_file, "wb") as f:
            f.write(model_trt.engine.serialize())

        print("Converted TensorRT model engine file is saved for C++ inference.")


if __name__ == "__main__":
    main()
