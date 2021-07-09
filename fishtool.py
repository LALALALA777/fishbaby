import numpy as np
from collections import Counter, Iterable
from visTool import drawBBoxesWithKindAndLength, show_image
from babydetector import get_fish_hw, get_useful_boxes, refine_bboxes, BBoxRefiner
from pprint import pprint
import os

root = os.path.split(os.path.realpath(__file__))[0]
lenpath = os.path.join(root, 'length.txt')


def get_real_length_from_txt(reset):
    fishlen = None
    try:
        with open(lenpath, 'r') as f:
            fishlen = f.readline().split()
            fishlen = tuple(map(float, fishlen))
    except:
        pass
    if not fishlen or reset is True:
        fishlen = input("According to previous printed, "
                        "press a fish referred length with format ('pixel length' 'real length(cm))':\n")
        with open(lenpath, 'w') as f:
            f.write(fishlen)
        fishlen = tuple(map(float, fishlen.split()))
    return fishlen


def fish_angle(box):
    _, _, h, w = box
    return np.round(np.arctan(h/w), 4)


def estimate_fish_length(box, refined=False):
    if refined:
        leftTop, rightTop, rightBottom, lefBottom = box
        opposite = np.abs(rightBottom - rightTop)
        adjacent = np.abs(rightBottom - lefBottom)
        length = max(np.linalg.norm(adjacent), np.linalg.norm(opposite))
    else:
        _, _, h, w = box
        length = np.sqrt((h**2 + w**2))
    return length


def get_fish_benchmarks(fishbaby_path):
    # note that the orientation of fish poses should be horizontal or vertical
    dic = {}
    fish_hw = get_fish_hw(fishbaby_path, show=False)   # returned is (x, y), already is orthogonal of (w, h)
    dic['boxLength'] = max(fish_hw) * 3/4
    dic['boxWidth'] = min(fish_hw)
    box = [0, 0, dic['boxWidth'], dic['boxLength']]
    dic['theta'] = fish_angle(box)
    dic['radius'] = int(estimate_fish_length(box))
    return dic


class FishBBoxedCounter():
    def __init__(self, len_criteria: Iterable, max_fish_size=np.iinfo('uint16').max,
                 reset=False, background='black', fgRate=.3,  **kwargs):

        assert isinstance(len_criteria, Iterable), \
            'Criteria fish should be Iterable object'
        kwargs = kwargs
        self.counter = Counter()
        self.fish = list(map(get_fish_benchmarks, list(len_criteria)))
        print('Each level fish info:')
        self.fish.sort(key=lambda elem: elem['boxLength'])
        pprint(self.fish, indent=4)
        self.lengthBase = [(i+1, j['boxLength']) for i, j in enumerate(self.fish)]
        self.lengthBase.append((len(self.lengthBase)+1, max_fish_size))
        self.lengthBase.insert(0, (0, 0))
        print('Fish levels (level, pixels): ', self.lengthBase)
        self.referredRealLen = get_real_length_from_txt(reset=reset)  # first element is in pixels, the other is in centimeter
        self.reset = reset
        self.display = kwargs['display'] if 'display' in kwargs.keys() and kwargs['display'] is True else False
        show = kwargs['show'] if 'show' in kwargs.keys() else False
        self.bboxesRefiner = BBoxRefiner(background=background, fgRate=fgRate, show=show, display=self.display)

    def classify(self, length):
        for i, hc in self.lengthBase:
            if length < hc:
                return i

    def get_bboxed_fish_size(self, idxs, bboxes, img):
        bboxes = get_useful_boxes(idxs, bboxes)
        if bboxes:
            realBboxes = self.bboxesRefiner.refine(img, bboxes)
            kinds = [0] * len(bboxes)   # for convenient drawing
            length = [0] * len(bboxes)  # the same thing up here
            for i, box in enumerate(realBboxes):
                fish_len = estimate_fish_length(box, refined=True)
                kinds[i] = self.classify(fish_len)
                length[i] = fish_len
            length = self.real_length(length) if self.reset is False else length
            if self.display and bboxes:
                drawBBoxesWithKindAndLength(img, bboxes, lens=length, kinds=kinds)
            self.counter.update(kinds)
        return img

    def get_count(self):
        """
        @return: the number of what fish had counted so far
        """
        print('Fish statistics:', self.counter)
        return sum(self.counter.values())

    def real_length(self, length: list):
        l = np.array(length, dtype=np.float)
        realLength = self.referredRealLen[1] * l / self.referredRealLen[0]  # referred RealLen=(pixels, cm)
        return realLength

    def renew(self):
        self.counter.clear()
