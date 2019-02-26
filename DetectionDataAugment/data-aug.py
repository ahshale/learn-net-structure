##########
# Data augmentation for Detection in YOLOv3
##########

# Contents
"""
changes for images & labels
===========
image -- jitter, scale, crop,        hue/saturation/exposure transform
label -- scale,  scale, translation, none
===========
"""

# pipeline
"""
+++++++++++                                            |    +++++++++++
1. randomly jitter image                               |    scale bbox
2. randomly scale image (based on input size to net)   |    scale bbox
+++++++++++                                            |    +++++++++++
3. crop image                                          |    crop/translate bbox
+++++++++++                                            |    +++++++++++
4. randomly flip image                                 |    flip bbox
+++++++++++                                            |    +++++++++++
5. randomly distort image                              |    -
+++++++++++                                            |    +++++++++++
"""

import copy
import numpy as np
import cv2

def _data_aug(instance, net_h, net_w, jitter=0.3, scale=(0.25,2), hue=18, saturation=1.5, exposure=1.5):
    imagename, bboxes = instance['filename'], instance['bboxes'] # bboxes = [{'xmin': ?, 'xmax': ?, 'ymin': ?, 'yamx': ?}, {...}, ...]
    image             = cv2.imread(imagename)[:, :, ::-1]

    image_h, image_w, _ = image.shape

    dh = image_h * jitter
    dw = image_w * jitter

    ######
    #  get new_size based on the scale of net_size
    ######
    # get new aspect ratio based on image size
    new_ar = (image_h + np.random.randint(-dh, dh)) / (image_w + np.random.randint(-dw, dw))
    # get new size based on net size & new aspect ratio
    scale  = np.random.uniform(scale[0], scale[1])
    ## scale the max side and then scale according to ar, limiting the value of new size
    if new_ar > 1:
        new_h = int(net_h * scale)
        new_w = int(new_h / new_ar)
    else:
        new_w = int(net_w * scale)
        new_h = int(new_w * new_ar)

    ######
    # get size difference & randomly set offset
    ######
    diff_w = net_w - new_w
    diff_h = net_h - new_h
    dx     = int(np.random.uniform(0, diff_w))
    dy     = int(np.random.uniform(0, diff_h))

    """
    deal with image
    """
    ######
    # resize image
    ######
    ## new image with proper scale to net_size and slightly random size jitter
    image = cv2.resize(image, (new_w, new_h))

    ######
    # crop or pad image to get fixed size
    ######
    if dx > 0:
        image = np.pad(image, ((0,0), (dx, diff_w-dx), (0,0)), mode='constant', constant_values=127)
    else:
        image = image[:, -dx:-diff_w+dx, :]
    if dy > 0:
        image = np.pad(image, ((dy, diff_h-dy), (0, 0), (0, 0)), mode='constant', constant_values=127)
    else:
        image = image[-dy:-diff_h+dy, :, :]
    
    ######
    # random flip image (left/right & top/bottom)
    ######
    lr_flip  = np.random.randint(2)
    image = cv2.flip(image, 1) if lr_flip == 1 else image
    tb_flip = np.random.randint(2)
    image = cv2.flip(image, 0) if tb_flip == 1 else image

    ######
    # random distort image in hsv space
    ######
    dhue = np.random.uniform(-hue, hue)
    dsat = np.random.uniform(1, saturation)
    dsat = dsat if np.random.randint(2) == 0 else 1./dsat
    dexp = np.random.uniform(1, exposure)
    dexp = dexp if np.random.randint(2) == 0 esle 1./dexp

    # convert image from RGB space to HSV space
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype("float")

    # change saturation and exposure
    image[:, :, 1] *= dsat
    image[:, :, 2] *= dexp
    # change hue
    image[:,:,0] += dhue
    image[:,:,0] -= (image[:,:,0] > 180)*180
    image[:,:,0] += (image[:,:,0] < 0)  *180

    # convert image back from HSV space to RGB space
    image = cv.cvtColor(image.astype('int8'), cv2.COLOR_HSV2RGB)

    """
    deal with box
    """
    # bboxes会依据图像变化而发生变化，使用deepcopy不改变bboxes的原值
    bboxes = copy.deepcopy(bboxes)

    # get the resize scale
    sx, sy = float(new_w)/image_w, float(new_h)/image_h
    # record the invalid bbox after transformation
    invalid_bboxes = []

    # get new bboxes
    for i in range(len(bboxes)):
        # crop
        bboxes[i]['xmin'] = np.maximum(bboxes[i]['xmin'] * sx + dx, 0)
        bboxes[i]['xmax'] = np.minimum(bboxes[i]['xmax'] * sx + dx, net_w)
        bboxes[i]['ymin'] = np.maximum(bboxes[i]['ymin'] * sy + dy, 0)
        bboxes[i]['ymax'] = np.minimum(bboxes[i]['ymax'] * sy + dy, net_h)

        if bboxes[i]['xmin'] > bboxes[i]['xmax'] or bboxes[i]['ymin'] > bboxes[i]['ymax']:
            invalid_bboxes.append(i)
            continue
        
        # flip (left and right)
        if lr_flip == 1:
            x_cache       = bboxes[i]['xmax']
            bboxes[i]['xmax'] = net_w - bboxes[i]['xmin']
            bboxes[i]['xmin'] = net_w - coord_cache
        # flip (top and bottom)
        if tb_flip == 1:
            y_cache       = bboxes[i]['ymax']
            bboxes[i]['ymax'] = new_h - bboxes[i]['ymin']
            bboxes[i]['ymin'] = new_h - y_cache

    bboxes = [bboxes[i] for i in range(len(bboxes)) if i not in invalid_bboxes]

    return image, bboxes