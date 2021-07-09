import os
from fishtool import FishBBoxedCounter
from babydetector import directly_get_output, get_YOLO
from visTool import horizontally_joint_photos
import cv2 as cv

root = os.path.split(os.path.abspath(__file__))[0]
criteria_root = os.path.join(root, 'criteria_fish')
test_root = os.path.join(root, 'snapshot')
test_result_root = os.path.join(test_root, 'testResult')
test_path = os.listdir(test_root)
fishScales = os.listdir(criteria_root)
crit_fish = [os.path.join(criteria_root, fishScale) for fishScale in fishScales]

yolo_dir = os.path.join(root, 'yolov3')
weightsPath = os.path.join(yolo_dir, 'yolov3-obj_30000.weights')
configPath = os.path.join(yolo_dir, 'yolov3-obj.cfg')



class Test():
    def __init__(self, net, crit_fish: list):
        self.net = net
        self.fishCounter = FishBBoxedCounter(crit_fish, background='black', display=True)

    def reckon(self, img):
        if img is None:
            return None
        idxs, boxes, _, _ = directly_get_output(img, self.net)
        ret = self.fishCounter.get_bboxed_fish_size(idxs, boxes, img.copy())
        count = self.fishCounter.get_count()
        info = self.fishCounter.counter.items()
        img = horizontally_joint_photos([img, ret])
        return img, count, list(info)


def makedir(path):
    if not os.path.exists(path):
        os.mkdir(path)


if __name__ == '__main__':
    makedir(test_result_root)
    count = 0
    with open(os.path.join(test_root, 'gt.txt'), 'r') as f:
        gt = f.read().split('\n')
        gf2 = list(map(int, gt[0]))
        gf3 = list(map(int, gt[1]))
        gf4 = list(map(int, gt[2]))
    gf2n = sum(gf2)
    gf3n = sum(gf3)
    gf4n = sum(gf4)
    f2, f3, f4 = 0, 0, 0
    net = get_YOLO(configPath, weightsPath)
    testNet = Test(net, crit_fish)
    images = [image_path for image_path in test_path if image_path.endswith('.png')]
    for image in images:
        img = os.path.join(test_root, image)
        img = cv.imread(img)
        img, fn, fs = testNet.reckon(img)
        if fn > 0:
            img = cv.resize(img, (1080, 480))
            cv.imwrite(os.path.join(test_result_root, image), img)
            name = os.path.splitext(image)[0]
            count += fn
            for l, c in fs:
                if l == 4:
                    f4 += c
                elif l == 3:
                    f3 += c
                elif l == 2:
                    f2 +=c
        #testNet.fishCounter.renew()
    print('ground true:\n\ttotal: {}\nlevel 2:{}\nlevel 3:{}\nlevel4:{}'.format(gf2n + gf3n + gf4n, gf2n, gf3n, gf4n))
    print('estimated:\n\ttotal: {}\nlevel 2:{}\nlevel 3:{}\nlevel4:{}'.format(count, f2, f3, f4))
    print(testNet.fishCounter.get_count())
