import numpy as np
from collections import Counter, Iterable
from visTool import paintBBoxesForOneImage
from babydetector import get_fish_hw
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
    dic['radius'] = estimate_fish_length(box)

    return dic


class FishBBoxedCounter():
    def __init__(self, len_criteria: Iterable):

        def take(elements):
            return elements[2]

        assert isinstance(len_criteria, Iterable),\
            'Criteria fish should be Iterable object'
        self.h_criteria = list(len_criteria)
        self.counter = Counter()

        self.fish = list(map(get_fish_benchmarks, len_criteria))
        self.lengths = [(i, j['radius']) for i, j in enumerate(self.fish)].sort(key=take)


    def classify(self, length):
        for i, hc in enumerate(self.lengths):
            if length < hc:
                return hc

    def get_bboxed_fish_size(self, idxs, bboxes, **kwargs):
        dic = kwargs
        kinds = [0] * len(bboxes)
        length = [0] * len(bboxes)
        if len(idxs) > 0:
            for i in idxs.flatten():
                box_len = estimate_fish_length(bboxes[i])
                length[i] = box_len
                kinds[i] = self.classify(box_len)

        if 'image' in dic.keys():
            img = dic['image']
            paintBBoxesForOneImage(img, idxs, bboxes, confidences=length, classIDs=np.arange(0, len(bboxes)),
                                   labels=kinds)

        self.counter.update(kinds)

    def get_count(self):
        print('Fish Total: ', self.counter)
        return len(self.counter.values())


if __name__ == '__main__':
    fishPath = 'testpictures/fish.png'
    get_fish_benchmarks(fishPath)
