import numpy as np

from babydetector import get_YOLO, get_output, get_bboxes, get_blobImg
from visTool import get_labels, show_image
from fishtool import FishBBoxedCounter
import cv2 as cv
import os
from fromCamera import snapshot, launch_camera
import time

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
    pass
    print("The time interval between two frames has calculated successfully")
    return


def main():
    fishCounter = FishBBoxedCounter(crit_fish)
    net = get_YOLO(configPath, weightsPath)
    if launch_camera() is True:
        while True:
            img = snapshot()
            if img is np.ndarray:
                blobImg = get_blobImg(img)
                layerOutputs = get_output(net, blobImg)
                idxs, boxes, confidences, classIDs = get_bboxes(layerOutputs, img.shape[:2])
                fishCounter.get_bboxed_fish_size(idxs, boxes, image=img)
                time.sleep(5)
            elif img is None:
                break
        print('Work finished.')
        return fishCounter.get_count()
    else:
        print('\033[0;31mSomething error!\033[0m')
        return


if __name__ == '__main__':
    img = cv.imread(imgPath)
    fishCounter = FishBBoxedCounter(crit_fish)
    hw = img.shape[:2]
    net = get_YOLO(configPath, weightsPath)
    #video_process(videoPath, net, fishsize=fishSize, laserstation=laserStation, labelspath=labelsPath, show=True)
    # fast_video_process(videoPath, net, fishsize=fishSize, laserstation=laserStation, shape=(256, 256))

    blobImg = get_blobImg(img)
    layerOutputs = get_output(net, blobImg)
    idxs, boxes, confidences, classIDs = get_bboxes(layerOutputs, hw)
    names = get_labels(labelsPath)
    #paintBBoxesForOneImage(img, idxs, boxes, confidences, classIDs, names)

    fishCounter.get_bboxed_fish_size(idxs, boxes, image=img)
    print('\033[0;35mThere you got {} Fish babies\033[0m'.format(fishCounter.get_count()))


