from fishtool import FishBBoxedCounter
from babydetector import directly_get_output
from fromCamera import launch_camera, snapshot, close_camera
import cv2 as cv


class FishFish():
    def __init__(self, net, crit_fish: list):
        self.net = net
        self.fishCounter = FishBBoxedCounter(crit_fish, background='black', display=True)

    def set_camera_mode(self, mode):
        launch_camera(mode)

    def reckon(self):
        img = snapshot()    # if no camera points out an image or otherwise
        # c = cv.VideoCapture(0)
        # img = c.read()[1]
        if img is None:
            return None
        if img is False:
            raise RuntimeWarning('Check camera.')
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        idxs, boxes, _, _ = directly_get_output(img, self.net)
        beforeN = self.fishCounter.get_count()
        img = self.fishCounter.get_bboxed_fish_size(idxs, boxes, img)
        afterN = self.fishCounter.get_count()
        return img, afterN - beforeN, afterN, list(self.fishCounter.counter.items())

    def exit(self):
        close_camera()

    def give(self):
        img = snapshot()
        return cv.cvtColor(img, cv.COLOR_BGR2RGB)

def get_YOLO():
    """@return YOLO network"""
    pass

def get_crit():
    """ @return List containing three fish images """
    return list()

if __name__ == '__main__':
    net = get_YOLO()
    crit = get_crit()
    fishf = FishFish(net, crit)
    fishf.reckon()

