import numpy as np


from babydetector import get_YOLO, get_output, get_bboxes, get_blobImg, directly_get_output
from visTool import get_labels, show_image
from fishtool import FishBBoxedCounter
import cv2 as cv
import os
from fromCamera import snapshot, launch_camera, close_camera
import time
import sys
sys.path.append(r'/home/jiang/PycharmProjects/oanet/OANet/demo')
sys.path.append(r'/home/jiang/PycharmProjects/oanet/OANet/core')
from demo import match_interface



# yolo config
yolo_dir = 'yolov3'  # YOLO文件路径
weightsPath = os.path.join(yolo_dir, 'yolov3-obj_30000.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3-obj.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'fishbaby.names')  # label名称

imgPath = 'snapshot/snap14.jpg'     # 测试图像
fishPath = 'testpictures/fish.png'  # 用于video得到fishSize
laserStation = .618     # 图中扫描线百分比位置
fishSize = tuple()
videoPath = 'testpictures/fs1.mp4'
criteria_root = 'criteria_fish'
fishScales = os.listdir(criteria_root)  # 不同level的鱼的图片文件名
crit_fish = [os.path.join(criteria_root, fishScale) for fishScale in fishScales]


def get_time_interval():
    pre = snapshot()
    pos = snapshot()
    """cv.imshow('previous', pre)
    cv.imshow('posterior', pos)"""

    corr1, corr2 = match_interface(pre, pos)

    print("The time interval between two frames has calculated successfully")
    return


def main(auto_interval=False):
    fishCounter = FishBBoxedCounter(crit_fish)
    net = get_YOLO(configPath, weightsPath)
    if launch_camera(toggle_mode=0) is True:
        if auto_interval:
            sleepTime = get_time_interval()
        while cv.waitKey(2) != ord('q'):
            img = snapshot()
            if img is None:
                continue
            elif isinstance(img, np.ndarray):
                idxs, boxes, _, _ = directly_get_output(img, net)
                img = fishCounter.get_bboxed_fish_size(idxs, boxes, image=img)
                cv.imshow('cap', img)
                #time.sleep(0)
            elif img is False:
                break
        close_camera()
        print('Work finished.')
        cv.destroyAllWindows()
        return fishCounter.get_count()
    else:
        print('\033[0;31mSomething error occurred in launch process !\033[0m')
        return None


if __name__ == '__main__':
    """img = cv.imread(imgPath)
    fishCounter = FishBBoxedCounter(crit_fish)
    hw = img.shape[:2]
    net = get_YOLO(configPath, weightsPath)

    blobImg = get_blobImg(img)
    layerOutputs = get_output(net, blobImg)
    idxs, boxes, confidences, classIDs = get_bboxes(layerOutputs, hw)
    names = get_labels(labelsPath)
    fishCounter.get_bboxed_fish_size(idxs, boxes, image=img)
    print('\033[0;35mThere you got {} Fish babies\033[0m'.format(fishCounter.get_count()))"""

    main(auto_interval=True)
