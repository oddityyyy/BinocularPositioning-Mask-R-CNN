# use activate tf36
# 命名原则：变量名和函数名用小写+下划线
import balloon
from mrcnn.model import log
import mrcnn.model as modellib
from mrcnn.visualize import display_images
from mrcnn import visualize
from mrcnn import utils
import os
import sys
import random
import math
import re
import time
import numpy as np
import tensorflow as tf
import cv2
import colorsys
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.measure import find_contours
from matplotlib.patches import Polygon
import imgmatch

global FOCAL
FOCAL = 1614.22
global BASELINE
BASELINE = 0.239867  # 0.119928
global BF
BF = BASELINE*FOCAL/2
# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library


def match_fit(bbox):  # maskrcnn预测的bbox其顺序是ymin,xmin,ymax,xmax与我处理的顺序xmin,ymin,xmax,ymax相反因此用match_fit调转顺序
    l = int(len(bbox) / 2)
    for i in range(l):
        i = i*2
        bbox[i], bbox[i + 1] = bbox[i + 1], bbox[i]
    return bbox


def del_minus1_in_bbox(LBBOX, MBBOX):
    flag = 0
    while(-1 in MBBOX[flag:]):
        index = MBBOX.index(-1)
        flag = index+4
        LBBOX[index:index+4] == -1
    return LBBOX, MBBOX


def apple_mask(image, mask, color, alpha=0.5):      #apple改为calyx
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def random_colors(N, bright=True):
    """s
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None,
                      show_mask=True, show_bbox=True,
                      colors=None, captions=None):

    
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    if not ax:
        _, ax = plt.subplots(1, figsize=figsize)
        auto_show = True

    # Generate random colors
    colors = colors or random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]

        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1, y1 + 8, caption,
                color='w', size=11, backgroundcolor="none")

        # Mask
        mask = masks[:, :, i]
        if show_mask:
            masked_image = apple_mask(masked_image, mask, color)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))
    # print('jan')
    plt.savefig('D:/maskrcnn/Mask_RCNN-master/apple/val/1.jpg')     #apple改为calyx
    # print('feb')


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
# TODO: update this path
BALLON_WEIGHTS_PATH = "D:\maskrcnn\Mask_RCNN-master\logs\apple20200430T1315\mask_rcnn_apple_0300.h5"

config = balloon.appleConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "apple")

class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

config.IMAGE_SHAPE = np.array([1920, 1088, 3])
config.IMAGE_MAX_DIM = 1920
config.IMAGE_RESIZE_MODE = 'none'

config.display()

DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size*cols, size*rows))
    return ax


def main(BALLOON_DIR, DEVICE, MODEL_DIR, config, ID, name):
    # Load validation dataset
    dataset = balloon.appleDataset()                 #apple改为calyx
    dataset.load_apple(BALLOON_DIR, "val")           #apple改为calyx

    # Must call before using the dataset
    dataset.prepare()

    print("Images: {}\nClasses: {}".format(
        len(dataset.image_ids), dataset.class_names))
    # Create model in inference mode
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                                  config=config)
    weights_path = model.find_last()

    # Load weights
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True,exclude=[ "mrcnn_class_logits", "mrcnn_bbox_fc", "mrcnn_bbox", "mrcnn_mask"])
    image_id = ID
    # image_id = random.choice(dataset.image_ids)

    image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset, config, image_id, use_mini_mask=False)
    info = dataset.image_info[image_id]
    # print("image ID: {}.{} ({}) {}".format(info["source"], info["id"], image_id,
    # dataset.image_reference(image_id)))
    a = time.time()
    # Run object detection
    results = model.detect([image], verbose=1)
    b = time.time()
    print('detect cost'+str(b-a))

    # Display results
    ax = get_ax(1)
    r = results[0]
    # r是一个字典.其中r['rois']代表faster RCNN返回的bbox.r['masks']代表三维的mask map.其最后一个维度是果实的个数.r['masks']只有True和False两个值
    display_instances(image, r['rois'], r['masks'], r['class_ids'],
                      ['BG', 'apple','kiwi','banana','berry'], r['scores'])          #apple改为calyx
    z = np.zeros((r['masks'].shape[0], r['masks'].shape[1], 3))
    c = 0
    '''
    for i in range(r['masks'].shape[2]):
        a = np.array([random.randint(0, 255), random.randint(
            0, 255), random.randint(0, 255)])
        b = a.tolist()
        cv2.rectangle(z,  (r['rois'][c, 1], r['rois'][c, 0]), (r['rois'][c, 3], r['rois'][c, 2]),
                      (int(b[0]), int(b[1]), int(b[2])), 1)
        c += 1
        for j in range(r['masks'].shape[0]):
            for k in range(r['masks'].shape[1]):
                if r['masks'][j, k, i] == True:
                    z[j, k] = a
    cv2.imwrite(name + '.png', z)'''
    return [r['rois'], r['masks']]


lbbox, lmasks = main(BALLOON_DIR, DEVICE, MODEL_DIR, config, 0, 'lmask')   
len(lbbox)
