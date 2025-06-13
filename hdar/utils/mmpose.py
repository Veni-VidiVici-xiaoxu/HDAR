# mmpose.py
import os
import cv2
import numpy as np
from tqdm import tqdm
from mmpose.apis import MMPoseInferencer
from pare.core.config import (
    MMPOSE_PATH,
    MMDET_PATH,
    MMPOSE_CKPT,
    MMPOSE_CFG,
    MMDET_CFG,
    MMDET_CKPT
)


def run_mmpose_with_dets(image_file_names, detections, show_results=False, results_folder=None):
    """Run 2D pose estimation with pre-computed detections"""
    device = 'cuda:0'
    bbox_thr = 0.1
    kpt_thr = 0.1

    # Initialize inferencer
    inferencer = MMPoseInferencer(
        pose2d=MMPOSE_CFG,
        pose2d_weights=MMPOSE_CKPT,
        device=device,
        show_progress=False
    )

    if results_folder:
        os.makedirs(results_folder, exist_ok=True)

    joints2d = []
    for img_idx, img_fname in enumerate(image_file_names):
        # Get detections for current image
        person_bboxes = detections[img_idx].copy()

        if len(person_bboxes) == 0:
            joints2d.append(np.zeros((133, 3)))
            continue

        # Convert to xywh format with confidence
        person_bboxes[:, 0] -= person_bboxes[:, 2] // 2
        person_bboxes[:, 1] -= person_bboxes[:, 3] // 2
        person_bboxes = np.concatenate(
            [person_bboxes, np.ones((person_bboxes.shape[0], 1))],
            axis=-1
        )

        # Run inference and get results
        results = list(inferencer(
            img_fname,
            det_bboxes=person_bboxes,
            bbox_format='xywh',
            return_vis=show_results or (results_folder is not None),
            vis_out_dir=results_folder,
            draw_bbox=True,
            draw_heatmap=False,
            skeleton_style='mmpose',
            radius=3,
            thickness=1,
            kpt_thr=kpt_thr,
            bbox_thr=bbox_thr
        ))

        # Process results
        current_joints = np.zeros((133, 3))
        if len(results) > 0 and 'predictions' in results[0]:
            predictions = results[0]['predictions']
            if len(predictions) > 0 and 'keypoints' in predictions[0]:
                kpts = predictions[0]['keypoints']
                if kpts.shape[0] >= 133:
                    current_joints = kpts[:133]
                else:
                    current_joints[:kpts.shape[0]] = kpts

        joints2d.append(current_joints)

    return np.array(joints2d)


def run_mmpose(image_folder, show_results=False):
    """Run full pipeline (detection + pose estimation)"""
    device = 'cuda:0'
    bbox_thr = 0.3
    kpt_thr = 0.3

    inferencer = MMPoseInferencer(
        pose2d=MMPOSE_CFG,
        pose2d_weights=MMPOSE_CKPT,
        det_model=MMDET_CFG,
        det_weights=MMDET_CKPT,
        device=device,
        show_progress=True
    )

    image_names = sorted([
        os.path.join(image_folder, x)
        for x in os.listdir(image_folder)
        if x.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    joints2d = []
    for image_name in tqdm(image_names, desc="Processing images"):
        # Get results
        results = list(inferencer(
            image_name,
            return_vis=show_results,
            draw_bbox=True,
            draw_heatmap=False,
            skeleton_style='mmpose',
            radius=3,
            thickness=1,
            kpt_thr=kpt_thr,
            bbox_thr=bbox_thr
        ))

        # Process results
        current_joints = np.zeros((133, 3))
        if len(results) > 0 and 'predictions' in results[0]:
            predictions = results[0]['predictions']
            if len(predictions) > 0 and 'keypoints' in predictions[0]:
                kpts = predictions[0]['keypoints']
                if kpts.shape[0] >= 133:
                    current_joints = kpts[:133]
                else:
                    current_joints[:kpts.shape[0]] = kpts

        joints2d.append(current_joints)

    return np.array(joints2d)