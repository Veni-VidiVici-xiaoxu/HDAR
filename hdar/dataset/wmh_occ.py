# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2019 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: ps-license@tuebingen.mpg.de

# Adapted from https://github.com/isarandi/synthetic-occlusion/blob/master/augmentation.py

#!/usr/bin/env python

import functools
import os.path
import random
import sys
import xml.etree.ElementTree
import numpy as np
import matplotlib.pyplot as plt
import skimage.data
import cv2
import PIL.Image

import skimage.io
def main():
    """Demo of how to use the code"""

    path = '/home/aita/uaitadisknew/wmhcode/spin/datasets/VOCdevkit/VOC2012'

    print('Loading occluders from Pascal VOC dataset...')
    occluders = load_occluders(pascal_voc_root_path=path)
    print('Found {} suitable objects'.format(len(occluders)))
    input_dir = "/home/aita/uaitadisknew/wmhcode/PARE-master/test_madison/madison.png"  # <<< 修改为你的图片文件夹路径
    original_im =skimage.io.imread(input_dir)

    # 创建保存图片的目录
    output_dir = "outputocc"
    os.makedirs(output_dir, exist_ok=True)

    for i in range(9):  # 生成 9 张不同的遮挡图
        occluded_im = occlude_with_objects(original_im, occluders)
        output_path = os.path.join(output_dir, f'occluded_{i}.jpg')
        plt.imsave(output_path, occluded_im)  # 单独保存每张图
        print(f"已保存: {output_path}")

    print("所有图片已保存！")



def load_occluders(pascal_voc_root_path):
    occluders = []
    structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (8, 8))

    annotation_paths = list_filepaths(os.path.join(pascal_voc_root_path, 'Annotations'))
    for annotation_path in annotation_paths:
        xml_root = xml.etree.ElementTree.parse(annotation_path).getroot()
        is_segmented = (xml_root.find('segmented').text != '0')

        if not is_segmented:
            continue

        boxes = []
        for i_obj, obj in enumerate(xml_root.findall('object')):
            is_table = (obj.find('name').text == 'diningtable')  # 只选桌子
            is_difficult = (obj.find('difficult').text != '0')
            is_truncated = (obj.find('truncated').text != '0')
            if is_table and not is_difficult and not is_truncated:
                bndbox = obj.find('bndbox')
                box = [int(bndbox.find(s).text) for s in ['xmin', 'ymin', 'xmax', 'ymax']]
                boxes.append((i_obj, box))

        if not boxes:
            continue

        im_filename = xml_root.find('filename').text
        seg_filename = im_filename.replace('jpg', 'png')

        im_path = os.path.join(pascal_voc_root_path, 'JPEGImages', im_filename)
        seg_path = os.path.join(pascal_voc_root_path, 'SegmentationObject', seg_filename)

        im = np.asarray(PIL.Image.open(im_path))
        labels = np.asarray(PIL.Image.open(seg_path))

        for i_obj, (xmin, ymin, xmax, ymax) in boxes:
            object_mask = (labels[ymin:ymax, xmin:xmax] == i_obj + 1).astype(np.uint8) * 255
            object_image = im[ymin:ymax, xmin:xmax]
            if cv2.countNonZero(object_mask) < 500:
                continue

            eroded = cv2.erode(object_mask, structuring_element)
            object_mask[eroded < object_mask] = 192
            object_with_mask = np.concatenate([object_image, object_mask[..., np.newaxis]], axis=-1)

            object_with_mask = resize_by_factor(object_with_mask, 0.5)
            occluders.append(object_with_mask)

    return occluders


def occlude_with_objects(im, occluders):
    result = im.copy()
    width_height = np.asarray([im.shape[1], im.shape[0]])  # [width, height]

    if not occluders:
        return result  # 没有遮挡物则返回原图

    occluder = random.choice(occluders)  # 选择一个遮挡物

    scale_factor = 5.0  # 固定缩放比例
    occluder = resize_by_factor(occluder, scale_factor)

    # 计算遮挡物的中心位置
    center_x = width_height[0] // 2  # 图片水平中心
    center_y = int(width_height[1] * 0.75)  # 靠近下方（75% 位置）

    center = np.array([center_x, center_y])  # 组合中心坐标
    paste_over(im_src=occluder, im_dst=result, center=center)  # 叠加遮挡物

    return result  # 返回处理后的图片


def paste_over(im_src, im_dst, center):
    """Pastes `im_src` onto `im_dst` at a specified position, with alpha blending, in place.

    Locations outside the bounds of `im_dst` are handled as expected (only a part or none of
    `im_src` becomes visible).

    Args:
        im_src: The RGBA image to be pasted onto `im_dst`. Its size can be arbitrary.
        im_dst: The target image.
        alpha: A float (0.0-1.0) array of the same size as `im_src` controlling the alpha blending
            at each pixel. Large values mean more visibility for `im_src`.
        center: coordinates in `im_dst` where the center of `im_src` should be placed.
    """

    width_height_src = np.asarray([im_src.shape[1], im_src.shape[0]])
    width_height_dst = np.asarray([im_dst.shape[1], im_dst.shape[0]])

    center = np.round(center).astype(np.int32)
    raw_start_dst = center - width_height_src // 2
    raw_end_dst = raw_start_dst + width_height_src

    start_dst = np.clip(raw_start_dst, 0, width_height_dst)
    end_dst = np.clip(raw_end_dst, 0, width_height_dst)
    region_dst = im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]]

    start_src = start_dst - raw_start_dst
    end_src = width_height_src + (end_dst - raw_end_dst)
    region_src = im_src[start_src[1]:end_src[1], start_src[0]:end_src[0]]
    color_src = region_src[..., 0:3]
    alpha = region_src[..., 3:].astype(np.float32) / 255

    im_dst[start_dst[1]:end_dst[1], start_dst[0]:end_dst[0]] = (
            alpha * color_src + (1 - alpha) * region_dst)


def resize_by_factor(im, factor):
    """Returns a copy of `im` resized by `factor`, using bilinear interp for up and area interp
    for downscaling.
    """
    new_size = tuple(np.round(np.array([im.shape[1], im.shape[0]]) * factor).astype(int))
    interp = cv2.INTER_LINEAR if factor > 1.0 else cv2.INTER_AREA
    return cv2.resize(im, new_size, fx=factor, fy=factor, interpolation=interp)


def list_filepaths(dirpath):
    names = os.listdir(dirpath)
    paths = [os.path.join(dirpath, name) for name in names]
    return sorted(filter(os.path.isfile, paths))


def test_usage():
    from ..core.config import PASCAL_ROOT
    import matplotlib.pyplot as plt
    occluders = load_occluders(pascal_voc_path=PASCAL_ROOT)
    example_image = cv2.resize(skimage.data.astronaut(), (256, 256))
    occluded_image = occlude_with_objects(example_image, occluders)
    plt.show(occluded_image)

if __name__ == '__main__':
    main()
