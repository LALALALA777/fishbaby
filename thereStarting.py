from babydetector import get_YOLO, get_output, get_bboxes, get_blobImg
from visTool import get_labels, show_image
from fishtool import FishBBoxedCounter
import cv2 as cv
import os
from fromCamera import capture_snapshot, launch_camera
import time


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
fishScales = ['fish1.png']  # 不同level的鱼的图片文件名
crit_fish = [os.path.join(criteria_root, fishScale) for fishScale in fishScales]


def main():
    fishCounter = FishBBoxedCounter(crit_fish)
    net = get_YOLO(configPath, weightsPath)
    if launch_camera() is True:
        while True:
            img = capture_snapshot()
            if img is not None:
                blobImg = get_blobImg(img)
                layerOutputs = get_output(net, blobImg)
                idxs, boxes, confidences, classIDs = get_bboxes(layerOutputs, img.shape[:2])
                fishCounter.get_bboxed_fish_size(idxs, boxes, image=img)
                time.sleep(5)


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
    show_image(img)
    print('[INFO] There you got {} Fish babies '.format(fishCounter.get_count()))


