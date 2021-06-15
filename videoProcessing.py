import cv2 as cv
import math


class FishScanProcessing():
    def __init__(self, length: int, width: int, fish_size: tuple, laser: float, incident=math.pi/360):
        assert laser < 1, 'laser located must be under video length'
        self.length = length
        self.width = width
        self.lFish = fish_size[1] // 2
        self.wFish = fish_size[0] // 2
        self.count = 0
        self.laser = laser * length
        self.down_margin = math.ceil(math.tan(incident)*(length - self.laser))
        self.seat = [0] * (width + self.down_margin)

    def scan(self, boxes):
        n = len(boxes)
        if n > 0:
            for box in boxes:
                x, y, w, h = box
                wCentra = x + w//2
                lCentra = y + h//2
                if y > self.laser:  # 鱼通过laser
                    if self.discharge(wCentra):
                        self.count += 1
                elif y <= self.laser <= y + h:  # 鱼被laser击中
                    if self.occupy(wCentra):
                        pass

        else:
            self.reset()

    def discharge(self, centra):
        for i in range(1, self.wFish):
            if self.seat[centra + i] == 1:
                self.seat[centra + i] = 0
            else:
                return False
            if self.seat[centra - i] == 1:
                self.seat[centra - i] = 0
            else:
                return False
        self.seat[centra] = 0
        return True

    def occupy(self, centra):
        for i in range(self.wFish + self.down_margin):
            self.seat[centra + i] = 1
            self.seat[centra - i] = 1
        return True

    def get_count(self):
        return self.count

    def reset(self):
        self.seat = [0] * (self.width + self.down_margin)


def get_all_frames(v: cv.VideoCapture, reshape=False, shape=None):
    photos = []
    frame = v.read()
    if reshape and shape is not None:
        while frame[0]:
            photos.append(cv.resize(frame[1], shape))
            frame = v.read()
    else:
        while frame[0]:
            photos.append(frame[1])
            frame = v.read()
    return photos


def get_interval():
    pass