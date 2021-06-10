import time

import numpy as np

from babydetector import *
from visTool import *
from fishtool import *
import cv2 as cv
import os


yolo_dir = 'yolov3'  # YOLO文件路径
weightsPath = os.path.join(yolo_dir, 'yolov3-obj_30000.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3-obj.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'fishbaby.names')  # label名称
imgPath = 'snapshot/snap17.jpg'     # 测试图像
fishPath = 'testpictures/fish.png'
laserStation = .618     # 扫描线图中百分比位置
fishSize = tuple()
videoPath = 'testpictures/fs1.mp4'
len_criteria = [0, 1, 100, 200, 400, 600, 800, 1000]


if __name__ == '__main__':
    img = cv.imread(imgPath)
    fishCounter = FishBBoxedCounter(len_criteria)
    fishSize = get_fish_hw(fishPath)
    hw = img.shape[:2]
    net = get_YOLO(configPath, weightsPath)
    # video_process(videoPath, net, fishsize=fishSize, laserstation=laserStation, labelspath=labelsPath, show=True)
    # fast_video_process(videoPath, net, fishsize=fishSize, laserstation=laserStation, shape=(256, 256))

    blobImg = get_blobImg(img)
    layerOutputs = get_output(net, blobImg)
    idxs, boxes, confidences, classIDs = get_bboxes(layerOutputs, hw)
    detectedTotal = len(idxs)
    """names = get_labels(labelsPath)
    paintBBoxesForOneImage(img, idxs, boxes, confidences, classIDs, names)"""

    fishCounter.get_bboxed_fish_size(idxs, boxes, image=img)
    fishCounter.get_count()


    show_image(img)
    print('[INFO] There you got {} Fish babies '.format(detectedTotal))


