import numpy as np
from collections import Counter, Iterable
from visTool import drawBBoxesWithKindAndLength, show_image
from babydetector import get_fish_hw, get_real_boxes
from pprint import pprint


referredRealLen = (218.9, 12)   # first element is in pixels, the other is in centimeter


def fish_angle(box):
    _, _, h, w = box
    return np.round(np.arctan(h/w), 4)


def estimate_fish_length(box):
    _, _, h, w = box
    length = np.sqrt((h**2 + w**2))
    return length


def get_fish_benchmarks(fishbaby_path):
    dic = {}
    fish_hw = get_fish_hw(fishbaby_path, show=False)   # returned is (x, y), already is orthogonal of (w, h)
    dic['boxLength'] = max(fish_hw)
    dic['boxWidth'] = min(fish_hw)
    box = [0, 0, dic['boxWidth'], dic['boxLength']]
    dic['theta'] = fish_angle(box)
    dic['radius'] = int(estimate_fish_length(box))
    return dic


class FishBBoxedCounter():
    def __init__(self, len_criteria: Iterable, max_fish_size=1000):
        """
        :param len_criteria: levels of different fish file name
        :param max_fish_size: ceiling of fish size
        """

        assert isinstance(len_criteria, Iterable), \
            'Criteria fish should be Iterable object'

        def SecondofElement(element):
            return element[1]

        self.referredRealLen = referredRealLen
        self.counter = Counter()
        self.fish = list(map(get_fish_benchmarks, list(len_criteria)))
        print('Each level fish info:')
        pprint(self.fish, indent=4)
        self.lengthBase = [(i+1, j['radius']) for i, j in enumerate(self.fish)]
        self.lengthBase.sort(key=SecondofElement)
        print('Fish levels: ', self.lengthBase)
        self.lengthBase.append((len(self.lengthBase), max_fish_size))

    def classify(self, length):
        for i, hc in self.lengthBase:
            if length < hc:
                return i

    def get_bboxed_fish_size(self, idxs, bboxes, **kwargs):
        kwargs = kwargs
        bboxes = get_real_boxes(idxs, bboxes)
        kinds = [0] * len(bboxes)   # for convenient drawing
        length = [0] * len(bboxes)  # the same thing up here
        for i, box in enumerate(bboxes):
            fish_len = estimate_fish_length(box)
            length[i] = fish_len
            kinds[i] = self.classify(fish_len)
        length = self.real_length(length)
        if 'image' in kwargs.keys():
            img = kwargs['image']
            drawBBoxesWithKindAndLength(img, bboxes, lens=length, kinds=kinds)
            show_image(img)
            return img

        self.counter.update(kinds)

    def get_count(self):
        """

        @return: the number of what fish had counted so far
        """
        print('Fish statistics:', self.counter)
        return sum(self.counter.values())

    def real_length(self, length: list):
        l = np.array(length, dtype='float')
        realLength = self.referredRealLen[1] * l / self.referredRealLen[0]  # referredRealLen=(pixels, cm)
        return realLength


if __name__ == '__main__':
    fishPath = 'testpictures/fish.png'
    get_fish_benchmarks(fishPath)
