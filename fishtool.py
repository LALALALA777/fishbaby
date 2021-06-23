import numpy as np
from collections import Counter, Iterable
from visTool import drawBBoxesWithKindAndLength, show_image
from babydetector import get_fish_hw, get_useful_boxes, refine_bboxes
from pprint import pprint


def get_real_length(reset):
    fishlen = None
    try:
        with open('length.txt', 'r') as f:
            fishlen = f.readline().split()
            fishlen = tuple(map(float, fishlen))
    except:
        pass
    if fishlen is None or reset is True:
        fishlen = input("According to previous printed, "
                        "press a fish referred length with format ('pixel length' 'real length(cm))':\n")
        with open('length.txt', 'w') as f:
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
    dic['boxLength'] = max(fish_hw)
    dic['boxWidth'] = min(fish_hw)
    box = [0, 0, dic['boxWidth'], dic['boxLength']]
    dic['theta'] = fish_angle(box)
    dic['radius'] = int(estimate_fish_length(box))
    return dic


class FishBBoxedCounter():
    def __init__(self, len_criteria: Iterable, max_fish_size=np.iinfo('uint16').max, reset=False):
        """
        @param len_criteria: different fish level
        @param max_fish_size: upper limit of fish
        @param reset: whether or not to reset real fish
        """

        assert isinstance(len_criteria, Iterable), \
            'Criteria fish should be Iterable object'

        def sortByElement(element):
            return element['boxLength']

        self.counter = Counter()
        self.fish = list(map(get_fish_benchmarks, list(len_criteria)))
        print('Each level fish info:')
        self.fish.sort(key=sortByElement)
        pprint(self.fish, indent=4)
        self.lengthBase = [(i+1, j['boxLength']) for i, j in enumerate(self.fish)]
        self.lengthBase.append((len(self.lengthBase)+1, max_fish_size))
        self.lengthBase.insert(0, (0, 0))
        print('Fish levels (level, pixels): ', self.lengthBase)
        self.referredRealLen = get_real_length(reset=reset)  # first element is in pixels, the other is in centimeter

    def classify(self, length):
        for i, hc in self.lengthBase:
            if length < hc:
                return i

    def get_bboxed_fish_size(self, idxs, bboxes, img, **kwargs):
        kwargs = kwargs
        bboxes = get_useful_boxes(idxs, bboxes)
        realBboxes = refine_bboxes(img, bboxes, display=False)
        kinds = [0] * len(bboxes)   # for convenient drawing
        length = [0] * len(bboxes)  # the same thing up here
        for i, box in enumerate(realBboxes):
            fish_len = estimate_fish_length(box, refined=True)
            kinds[i] = self.classify(fish_len)
            length[i] = fish_len
        length = self.real_length(length)
        if 'display' in kwargs.keys():
            drawBBoxesWithKindAndLength(img, bboxes, lens=length, kinds=kinds)
            show_image(img)
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

