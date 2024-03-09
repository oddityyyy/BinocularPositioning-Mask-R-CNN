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


def apply_mask(image, mask, color, alpha=0.5):
    """Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image


def random_colors(N, bright=True):
    """
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
            masked_image = apply_mask(masked_image, mask, color)

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
    '''ax.imshow(masked_image.astype(np.uint8))
    print('jan')
    plt.savefig('D:/maskrcnn/apple/val/1.jpg')
    print('feb')'''


# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Path to Ballon trained weights
# You can download this file from the Releases page
# https://github.com/matterport/Mask_RCNN/releases
# TODO: update this path
BALLON_WEIGHTS_PATH = r"..\..\logs\apple20200430T1315\mask_rcnn_apple_0300.h5"
config = balloon.appleConfig()
BALLOON_DIR = os.path.join(ROOT_DIR, "apple")


class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()

config.IMAGE_SHAPE = np.array([960, 1280, 3])  # 此处解决了输出mask与输入图片分辨率不同的问题
config.IMAGE_MAX_DIM = 1280
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
    dataset = balloon.appleDataset()
    dataset.load_apple(BALLOON_DIR, "val")

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
    model.load_weights(weights_path, by_name=True)
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

    z = np.zeros((r['masks'].shape[0], r['masks'].shape[1], 3))
    c = 0

    '''for i in range(r['masks'].shape[2]):
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

lbbox = lbbox.tolist()
Lbbox = []

for i in lbbox:
    for j in i:
        Lbbox.append(j)

lbbox = Lbbox
lbbox = match_fit(lbbox)
left = cv2.imread('D:/maskrcnn/Mask_RCNN-master/apple/val/val/Left.png')
left2 = left
right = cv2.imread('D:/maskrcnn/Mask_RCNN-master/apple/val/val/Right.png')
right2 = right
mbbox = imgmatch.match(left, right, lbbox)
c = time.time()

rbbox, rmasks = main(BALLOON_DIR, DEVICE, MODEL_DIR, config, 1, 'rmask')

rbbox = rbbox.tolist()
Rbbox = []
for i in rbbox:
    for j in i:
        Rbbox.append(j)
rbbox = Rbbox
rbbox = match_fit(rbbox)


def bbox_fix(bbox):  # 令lbbox不在出现1280,960
    for i in range(int(len(bbox)/2)):
        index = i*2
        if bbox[index] == 1280:
            bbox[index] = 1279
        if bbox[index+1] == 960:
            bbox[index+1] = 959
    return bbox


# 到此已经得到了maskrcnn对两张图的检测结果rbbox、rmasks和lbbox、lmasks
# rbbox lbbox mbbox到这里已经都是list类型,然后接下来开始匹配rbbox和mbbox，在匹配的过程中需要注意rbbox和mask的对应关系不能改变
# 比如说rbbox[0:4]对应的mask是rmasks[:,:,0]，rbbox[4:8]对应rmasks[:,:,1]，其中rmasks是一个三维np数组
print(len(lbbox))
print(lbbox)
print('mbbox:'+str(mbbox))
print('rbbox:'+str(rbbox))
#input('jan')
mbbox_rmasks_related_list, mbbox = imgmatch.mrbbox_match(mbbox, rbbox)
print('lbbox:'+str(lbbox))
print('mbbox:'+str(mbbox))
#input('feb')
# mbbox_rmasks_related_list,rbbox2=mbbox_rmasks_related_list[0],mbbox_rmasks_related_list[1]
# 最终的结果是通过lbbox、lmasks、mbbox、rmasks、mbbox_rmasks_related_list得出的.rbbox仅用于做mbbox和rmasks匹配的载体
# 从本行开始的rbbox就已经作废了.因为rbbox中的被匹配元素已经被换成了10000

# 当你设置class的成员的属性的时候，也可以使用setattr(cls_name,member_name,value)，相当于cls_name.member_name=value
# 获取属性的时候可以使用getattr(cls_name,member_name)，相当于输出cls_name.member_name


class process(object):
    def __init__(self, left, right, lbbox, lmasks, mbbox, rmasks, mbbox_rmasks_related_list):
        self.left = left
        self.right = right
        self.lbbox = lbbox
        self.lmasks = lmasks
        self.mbbox = mbbox
        self.rmasks = rmasks
        self.mbbox_rmasks_related_list = mbbox_rmasks_related_list
        self.Luboxlist = []
        self.Ldboxlist = []
        self.Llboxlist = []
        self.Lrboxlist = []
        self.Ruboxlist = []
        self.Rdboxlist = []
        self.Rlboxlist = []
        self.Rrboxlist = []

    def test_masks_and_bbox(self):
        print(self.mbbox_rmasks_related_list)
        for i in range(int(len(self.lbbox)/4)):
            flag = 4*i
            if self.mbbox_rmasks_related_list[i] == -1:
                continue
            self.color = (random.randint(0, 255), random.randint(
                0, 255), random.randint(0, 255))
            cv2.rectangle(self.left, (self.lbbox[flag], self.lbbox[flag+1]),
                          (self.lbbox[flag+2], self.lbbox[flag+3]), self.color, 2)
            # cv2.imwrite('left.jpg',self.left)
            w, h = lmasks.shape[:2]
            for j in range(w):
                for k in range(h):
                    if lmasks[j, k, i] == True:
                        self.left[j, k] = self.color
            cv2.rectangle(self.right, (self.mbbox[flag], self.mbbox[flag+1]),
                          (self.mbbox[flag+2], self.mbbox[flag+3]), self.color, 2)
            for j in range(w):
                for k in range(h):
                    if rmasks[j, k, self.mbbox_rmasks_related_list[i]] == True:
                        self.right[j, k] = self.color
        cv2.imwrite('right.png', self.right)
        cv2.imwrite('left.png', self.left)

    def find4pos(self):
        w, h = self.lmasks.shape[:2]
        l = int(len(lbbox)/4)
        for k in range(l):
            index = k*4
            if self.mbbox_rmasks_related_list[k] == -1:
                continue
            uflag = 0
            dflag = 0
            lflag = 0
            rflag = 0
            '''for i in range(w):
                for j in range(h):
                    if self.lmasks[i, j, k] == True:
                        self.left[i, j, 0] = 255'''

            for i in range(self.lbbox[index+3]-self.lbbox[index+1]):
                if uflag == 1 and dflag == 1:
                    break
                for j in range(self.lbbox[index+2]-self.lbbox[index]):
                    if uflag == 1 and dflag == 1:
                        break
                    '''if self.lmasks[self.lbbox[index+1]+i,self.lbbox[index]+j,k]==True:
                        self.left[self.lbbox[index+1]+i,self.lbbox[index]+j,0]=255'''
                    if self.lmasks[self.lbbox[index+1]+i, self.lbbox[index]+j, k] == True and uflag == 0:
                        self.Luboxlist.append(
                            [self.lbbox[index]+j, self.lbbox[index+1]+i])
                        uflag = 1
                    if self.lmasks[self.lbbox[index+3]-i, self.lbbox[index+2]-j, k] == True and dflag == 0:
                        self.Ldboxlist.append(
                            [self.lbbox[index+2]-j, self.lbbox[index+3]-i])
                        dflag = 1
            cv2.line(self.left, tuple(
                self.Luboxlist[-1]), tuple(self.Ldboxlist[-1]), (0, 0, 255), 3)
            for i in range(self.lbbox[index+2]-self.lbbox[index]):
                if lflag == 1 and rflag == 1:
                    break
                for j in range(self.lbbox[index+3]-self.lbbox[index+1]):
                    if rflag == 1 and lflag == 1:
                        break
                    if self.lmasks[self.lbbox[index+1]+j, self.lbbox[index]+i, k] == True and lflag == 0:
                        self.Llboxlist.append(
                            [self.lbbox[index]+i, self.lbbox[index+1]+j])
                        lflag = 1
                    if self.lmasks[self.lbbox[index+3]-j, self.lbbox[index+2]-i, k] == True and rflag == 0:
                        self.Lrboxlist.append(
                            [self.lbbox[index+2]-i, self.lbbox[index+3]-j])
                        rflag = 1

            cv2.line(self.left, tuple(
                self.Llboxlist[-1]), tuple(self.Lrboxlist[-1]), (0, 255, 0), 3)

            '''for i in range(w):
                for j in range(h):
                    if self.rmasks[i, j, self.mbbox_rmasks_related_list[k]] == True:
                        self.right[i, j, 0] = 255'''
            uflag, dflag, rflag, lflag = 0, 0, 0, 0
            for i in range(self.mbbox[index+3]-self.mbbox[index+1]):
                if uflag == 1 and dflag == 1:
                    break
                for j in range(self.mbbox[index+2]-self.mbbox[index]):
                    if uflag == 1 and dflag == 1:
                        break
                    '''if self.rmasks[self.mbbox[index+1]+i,self.mbbox[index]+j,self.mbbox_rmasks_related_list[k]]==True:
                        self.right[self.mbbox[index+1]+i,self.mbbox[index]+j,0]=255'''
                    if self.rmasks[self.mbbox[index+1]+i, self.mbbox[index]+j, self.mbbox_rmasks_related_list[k]] == True and uflag == 0:
                        self.Ruboxlist.append(
                            [self.mbbox[index]+j, self.mbbox[index+1]+i])
                        uflag = 1
                    if self.rmasks[self.mbbox[index+3]-i, self.mbbox[index+2]-j, self.mbbox_rmasks_related_list[k]] == True and dflag == 0:
                        self.Rdboxlist.append(
                            [self.mbbox[index+2]-j, self.mbbox[index+3]-i])
                        dflag = 1
            cv2.line(self.right, tuple(
                self.Ruboxlist[-1]), tuple(self.Rdboxlist[-1]), (0, 0, 255), 3)
            for i in range(self.mbbox[index+2]-self.mbbox[index]):
                if lflag == 1 and rflag == 1:
                    break
                for j in range(self.mbbox[index+3]-self.mbbox[index+1]):
                    if rflag == 1 and lflag == 1:
                        break
                    if self.rmasks[self.Llboxlist[-1][-1], self.mbbox[index]+i, self.mbbox_rmasks_related_list[k]] == True and lflag == 0:
                        self.Rlboxlist.append(
                            [self.mbbox[index]+i, self.Llboxlist[-1][-1]])
                        lflag = 1
                    if self.rmasks[self.Lrboxlist[-1][-1], self.mbbox[index+2]-i, self.mbbox_rmasks_related_list[k]] == True and rflag == 0:
                        self.Rrboxlist.append(
                            [self.mbbox[index+2]-i, self.Lrboxlist[-1][-1]])
                        rflag = 1
                    if self.rmasks[self.mbbox[index+1]+j, self.mbbox[index]+i, self.mbbox_rmasks_related_list[k]] == True and lflag == 0:
                        self.Rlboxlist.append(
                            [self.mbbox[index]+i, self.mbbox[index+1]+j])
                        lflag = 1
                    if self.rmasks[self.mbbox[index+3]-j, self.mbbox[index+2]-i, self.mbbox_rmasks_related_list[k]] == True and rflag == 0:
                        self.Rrboxlist.append(
                            [self.mbbox[index+2]-i, self.mbbox[index+3]-j])
                        rflag = 1
                    
                        
                    
            cv2.line(self.right, tuple(
                self.Rlboxlist[-1]), tuple(self.Rrboxlist[-1]), (0, 255, 0), 3)

        # cv2.imshow('right',self.right)
        # cv2.imshow('left',self.left)
        # cv2.waitKey(0)

        '''print(str(self.Ldboxlist),str(self.Luboxlist),str(self.Llboxlist),str(self.Lrboxlist))
        print(str(len(self.Ldboxlist)),str(len(self.Luboxlist)),str(len(self.Llboxlist)),str(len(self.Lrboxlist)))
        print()
        print(str(self.Rdboxlist),str(self.Ruboxlist),str(self.Rlboxlist),str(self.Rrboxlist))
        print(str(len(self.Rdboxlist)),str(len(self.Ruboxlist)),str(len(self.Rlboxlist)),str(len(self.Rrboxlist)))'''
        # 至此为止process类中的8个boxlist都有了相同长度的上下左右点的坐标，其格式为[(502, 824), (521, 328), (491, 753), (282, 280),........]

    def locate(self):
        # Z=(BASELINE*FOCAL)/disparity
        self.uD = []
        self.dD = []
        self.lD = []
        self.rD = []
        self.uX = []
        self.dX = []
        self.lX = []
        self.rX = []
        self.uY = []
        self.dY = []
        self.lY = []
        self.rY = []
        for i in range(len(self.Lrboxlist)):
            uD = self.Luboxlist[i][0]-self.Ruboxlist[i][0]
            self.uD.append(
                round(BF/abs(uD), 2))
            self.uX.append(round(self.Luboxlist[i][0]*BASELINE/uD, 2))
            self.uY.append(round(self.Luboxlist[i][1]*BASELINE/uD, 2))
            dD = self.Ldboxlist[i][0]-self.Rdboxlist[i][0]
            self.dD.append(
                round(BF/abs(dD), 2))
            self.dX.append(round(self.Ldboxlist[i][0]*BASELINE/dD, 2))
            self.dY.append(round(self.Ldboxlist[i][1]*BASELINE/dD, 2))
            lD = self.Llboxlist[i][0]-self.Rlboxlist[i][0]
            self.lD.append(
                round(BF/abs(lD), 2))
            self.lX.append(round(self.Llboxlist[i][0]*BASELINE/lD, 2))
            self.lY.append(round(self.Llboxlist[i][1]*BASELINE/lD, 2))
            rD = self.Lrboxlist[i][0]-self.Rrboxlist[i][0]
            self.rD.append(
                round(BF/abs(rD), 2))
            self.rX.append(round(self.Lrboxlist[i][0]*BASELINE/rD, 2))
            self.rY.append(round(self.Lrboxlist[i][1]*BASELINE/rD, 2))

    def vis(self):
        for i in range(len(self.uD)):
            cv2.putText(self.left, str(self.uD[i]), tuple(
                self.Luboxlist[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.left, str(self.dD[i]), tuple(
                self.Ldboxlist[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(self.left, str(self.lD[i]), tuple(
                self.Llboxlist[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(self.left, str(self.rD[i]), tuple(
                self.Lrboxlist[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    def save(self):
        cv2.imwrite('left4pos.png', self.left)
        cv2.imwrite('right4pos.png', self.right)

    def point(self):
        print('Lu:'+str(self.Luboxlist))
        print(len(self.Luboxlist))
        print('Ld:'+str(self.Ldboxlist))
        print(len(self.Ldboxlist))
        print('Ll:'+str(self.Llboxlist))
        print(len(self.Llboxlist))
        print('Lr:'+str(self.Lrboxlist))
        print(len(self.Lrboxlist))
        print('Ru:'+str(self.Ruboxlist))
        print(len(self.Ruboxlist))
        print('Rd:'+str(self.Rdboxlist))
        print(len(self.Rdboxlist))
        print('Rl:'+str(self.Rlboxlist))
        print(len(self.Rlboxlist))
        print('Rr:'+str(self.Rrboxlist))
        print(len(self.Rrboxlist))

    def show_xyz(self):
        print('lX'+str(self.lX))
        print('lY'+str(self.lY))
        print('lD'+str(self.lD))
        print('rX'+str(self.rX))
        print('rY'+str(self.rY))
        print('rD'+str(self.rD))
        print('dX'+str(self.dX))
        print('dY'+str(self.dY))
        print('dD'+str(self.dD))
        print('uX'+str(self.uX))
        print('uY'+str(self.uY))
        print('uD'+str(self.uD))

    def avg(self):  # 计算平均深度
        self.cZ = []
        for i in range(len(self.uD)):
            self.cZ.append(round((self.uD[i] + self.dD[i] + self.lD[i] + self.rD[i]) / 4, 2))
        print(self.cZ)

    def z_vis(self):  # 仅显示平均深度
        for i in range(len(self.cZ)):
            cv2.putText(self.left, str(self.cZ[i]), tuple(
                self.Ldboxlist[i]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    def std4pos(self):
        std = []
        for i in range(len(self.uD)):
            v = np.zeros([4], dtype=int)
            v[0], v[1], v[2], v[3] = self.uD[i], self.dD[i], self.lD[i], self.rD[i]
            print(v)
            std.append(round(np.std(v), 2))
        return std


'''print(len(lbbox))
print(lbbox)
print(len(mbbox))
print(mbbox)
print(mbbox_rmasks_related_list)
input('f')'''
lbbox = bbox_fix(lbbox)
mbbox = bbox_fix(mbbox)
sample = process(left2, right2, lbbox, lmasks, mbbox,
                 rmasks, mbbox_rmasks_related_list)
#a = time.time()
sample.find4pos()
#sample.point()
#b = time.time()
sample.locate()
#print('locate cost:'+str(b-a))
sample.vis()
cv2.imshow('left4pos.png', sample.left)
cv2.imshow('right4pos.png', sample.right)
cv2.waitKey(0)
sample.save()
sample.show_xyz()
# sample.point()
