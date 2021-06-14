import numpy as np
from collections import Counter, Iterable
from visTool import drawBBoxesWithKindAndLength
from babydetector import get_fish_hw, get_real_boxes
import cv2 as cv


def fish_angle(box):
    _, _, h, w = box
    return np.arctan(h/w)


def estimate_fish_length(box):
    _, _, h, w = box
    length = np.sqrt((h**2 + w**2))
    return length


def get_fish_benchmarks(fishbaby_path):
    dic = {}
    fish_hw = get_fish_hw(fishbaby_path, show=False)[::-1]   # returned is (x, y), already is orthogonal of (w, h)
    dic['length'] = max(fish_hw)
    dic['width'] = min(fish_hw)
    box = [0, 0, dic['width'], dic['length']]
    dic['theta'] = fish_angle(box)
    dic['radius'] = int(estimate_fish_length(box))

    return dic


class FishBBoxedCounter():
    def __init__(self, len_criteria: Iterable, max_fish_size=1000):
        assert isinstance(len_criteria, Iterable), \
            'Criteria fish should be Iterable object'
        def secondofelement(element):
            return element[1]

        self.h_criteria = list(len_criteria)
        self.counter = Counter()

        self.fish = list(map(get_fish_benchmarks, len_criteria))
        self.lengthBase = [(i, j['radius']) for i, j in enumerate(self.fish)]
        self.lengthBase.sort(key=secondofelement)
        self.lengthBase.append((len(self.lengthBase), max_fish_size))

    def classify(self, length):
        for i, hc in self.lengthBase:
            if length < hc:
                return i

    def get_bboxed_fish_size(self, idxs, bboxes, **kwargs):
        kwargs = kwargs
        bboxes = get_real_boxes(idxs, bboxes)
        kinds = [0] * len(bboxes)
        length = [0] * len(bboxes)
        for i, box in enumerate(bboxes):
            fish_len = estimate_fish_length(box)
            length[i] = fish_len
            kinds[i] = self.classify(fish_len)

        if 'image' in kwargs.keys():
            img = kwargs['image']
            drawBBoxesWithKindAndLength(img, bboxes, lens=length, kinds=kinds)

        self.counter.update(kinds)

    def get_count(self):
        print('Fish Total: ', self.counter)
        return sum(self.counter.values())


if __name__ == '__main__':
    fishPath = 'testpictures/fish.png'
    get_fish_benchmarks(fishPath)
