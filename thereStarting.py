import numpy as np
from babydetector import get_YOLO, get_output, get_bboxes, get_blobImg, directly_get_output, refine_bboxes, get_useful_boxes
from visTool import get_labels, show_image
from fishtool import FishBBoxedCounter
import cv2 as cv
import os
from fromCamera import snapshot, launch_camera, close_camera
import time
import sys
sys.path.append('..')


# yolo config
root = os.path.split(os.path.abspath(__file__))[0]
yolo_dir = os.path.join(root, 'yolov3')  # YOLO文件路径
weightsPath = os.path.join(yolo_dir, 'yolov3-obj_30000.weights')  # 权重文件
configPath = os.path.join(yolo_dir, 'yolov3-obj.cfg')  # 配置文件
labelsPath = os.path.join(yolo_dir, 'fishbaby.names')  # label名称

#imgPath = './snapshot/snap13.jpg'     # 测试图像
testImageRoot = os.path.join(root, 'testpictures')
imgPath = os.path.join(testImageRoot, 't4.png')
fishPath = os.path.join(testImageRoot, 'fish.png')  # 用于video得到fishSize
laserStation = .618     # 图中扫描线百分比位置
videoPath = os.path.join(testImageRoot, 'fs1.mp4')
criteria_root = os.path.join(root, 'criteria_fish')
fishScales = os.listdir(criteria_root)  # 在root下不同level的鱼的图片文件名
crit_fish = [os.path.join(criteria_root, fishScale) for fishScale in fishScales]

camera_mode = 0
main_mode = 'w'
reset_referred_length = False
background = 'black'
foregroundRate = .45
show = False

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


def main(waitTime, auto_interval=False):
    fishCounter = FishBBoxedCounter(crit_fish, reset=reset_referred_length, background=background, fgRate=foregroundRate,
                                    show=False, display=True)
    net = get_YOLO(configPath, weightsPath)
    if launch_camera(toggle_mode=camera_mode) is True:
        if auto_interval:
            waitTime = get_time_interval()
        if main_mode in ('work', 'w'):
            key = cv.waitKey(1)
            while key != ord('q'):
                start = time.time()
                img = snapshot()
                if img is None:
                    continue
                elif isinstance(img, np.ndarray):
                    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
                    #cv.imshow('ori', img)
                    idxs, boxes, _, _ = directly_get_output(img, net)
                    simg = fishCounter.get_bboxed_fish_size(idxs, boxes, img.copy())
                    cv.imshow('Press key q to quit; key s to save current image', simg)
                    print('process one image take {} secs'.format(time.time()-start))
                    time.sleep(waitTime)
                elif img is False:
                    break
                key = cv.waitKey(2)
                if key == ord('s'):
                    cv.imwrite(os.path.join(testImageRoot, 't50.png'), img)
                    print('saved image')
        elif main_mode in ('init', 'i'):
            # get fish in different levels
            levels = int(input('How much grades of fish are there:'))
            print('{} photos will be take'.format(levels))
            for i in range(levels):
                print("Press key c to determine this photo")
                while cv.waitKey(1) != ord('c'):
                    img = snapshot()
                    cv.imshow('gripped image', img)
                name = os.path.join(criteria_root, input('This fish (file) name:'))
                cv.imwrite(name, img)
                input('Type any word to next snapshot')
        close_camera()
        print('Work finished.')
        cv.destroyAllWindows()
        return fishCounter.get_count()
    else:
        print('\033[0;31mSomething error occurred in launch process !\033[0m')
        return None


def get_local(net=None):
    img = cv.imread(imgPath)
    fishCounter = FishBBoxedCounter(crit_fish, reset=reset_referred_length, background=background, display=True,
                                    show=show, fgRate=foregroundRate)
    hw = img.shape[:2]
    if net is None:
        net = get_YOLO(configPath, weightsPath)

    blobImg = get_blobImg(img)
    layerOutputs = get_output(net, blobImg)
    idxs, boxes, confidences, classIDs = get_bboxes(layerOutputs, hw)
    names = get_labels(labelsPath)
    fishCounter.get_bboxed_fish_size(idxs, boxes, img)
    show_image(img)
    print('\033[0;35mThere you got {} Fish babies\033[0m'.format(fishCounter.get_count()))

    if not crit_fish and main_mode != 'init':
        i = input('\033[0;31mNone of criteria fish for classification, do you wanna take some?: yes or no\033[0m')
        if i == 'yes':
            main_mode = 'init'
    cv.destroyAllWindows()
    return img


if __name__ == '__main__':
    """for i in range(5):
        imgPath = os.path.join(testImageRoot, 't'+str(i+1)+'.png')
        pass
        get_local()"""
    print(main(waitTime=0.01, auto_interval=False))