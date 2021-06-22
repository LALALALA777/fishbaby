import numpy as np
from babydetector import get_YOLO, get_output, get_bboxes, get_blobImg, directly_get_output, refine_bboxes, get_useful_boxes
from visTool import get_labels, show_image
from fishtool import FishBBoxedCounter
import cv2 as cv
import os
from fromCamera import snapshot, launch_camera, close_camera
import time

# yolo config
yolo_dir = 'yolov3'  # YOLO文件路径
weightsPath = os.path.join(yolo_dir, 'yolov3-obj_30000.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3-obj.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'fishbaby.names')  # label名称

imgPath = 'snapshot/snap13.jpg'     # 测试图像
fishPath = 'testpictures/fish.png'  # 用于video得到fishSize
laserStation = .618     # 图中扫描线百分比位置
fishSize = tuple()
videoPath = 'testpictures/fs1.mp4'
criteria_root = 'criteria_fish'
fishScales = os.listdir(criteria_root)  # 在root下不同level的鱼的图片文件名
crit_fish = [os.path.join(criteria_root, fishScale) for fishScale in fishScales]


def get_time_interval():
    source = snapshot()
    time.sleep(1)
    target = snapshot()
    #source = cv.imread('testpictures/f20.png')
    #target = cv.imread('testpictures/f21.png')
    cv.cvtColor(source, cv.COLOR_RGB2GRAY)
    cv.cvtColor(target, cv.COLOR_RGB2GRAY)
    ph, pw = target.shape
    template = source[int(ph*0.35):int(ph*0.65), int(pw*0.45):int(pw*0.55)]
    res = cv.matchTemplate(target, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
    top_left = min_loc  # (x, y), see official document why take min_loc
    """bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])
    cv.rectangle(target, top_left, bottom_right, (255, 0, 0))
    cv.imshow('s', target)
    cv.waitKeyEx()
    cv.destroyAllWindows()"""
    d = np.abs(top_left[0] - int(pw*0.45))
    waitTime = 0 if d == 0 else pw // d
    print("The time interval between two frames has calculated\n\twait time:", waitTime)
    return waitTime


def main(waitTime: int, auto_interval=False):
    fishCounter = FishBBoxedCounter(crit_fish)
    net = get_YOLO(configPath, weightsPath)
    if launch_camera(toggle_mode=0) is True:
        if auto_interval:
            waitTime = get_time_interval()
        assert isinstance(waitTime, int), \
            '\033[0;31mTime interval between snapshot must be integral type. \033[0m'
        while cv.waitKey(2) != ord('q'):
            img = snapshot()
            if img is None:
                continue
            elif isinstance(img, np.ndarray):
                idxs, boxes, _, _ = directly_get_output(img, net)
                img = fishCounter.get_bboxed_fish_size(idxs, boxes, image=img)
                cv.imshow('cap', img)
                time.sleep(waitTime)
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
    img = cv.imread(imgPath)
    fishCounter = FishBBoxedCounter(crit_fish)
    hw = img.shape[:2]
    net = get_YOLO(configPath, weightsPath)

    blobImg = get_blobImg(img)
    layerOutputs = get_output(net, blobImg)
    idxs, boxes, confidences, classIDs = get_bboxes(layerOutputs, hw)
    #boxes = refine_bboxes(img, get_useful_boxes(idxs, boxes), display=True), show_image(img)
    names = get_labels(labelsPath)
    fishCounter.get_bboxed_fish_size(idxs, boxes, img, display=True)
    print('\033[0;35mThere you got {} Fish babies\033[0m'.format(fishCounter.get_count()))

    main(waitTime=0, auto_interval=True)