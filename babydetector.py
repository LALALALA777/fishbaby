import cv2
import numpy as np
import cv2 as cv
import time
from videoProcessing import FishScanProcessing, get_all_frames
from sklearn.linear_model import LogisticRegression


CONFIDENCE = 0.3  # 过滤弱检测的最小概率
THRESHOLD = 0.5  # 非最大值抑制阈值,  重叠面积比小于这个的框保留
color = (0, 255, 0)


def directly_get_output(img, net):
    blobImg = get_blobImg(img)
    layerOutputs = get_output(net, blobImg)
    return get_bboxes(layerOutputs, img.shape[:2])


# 加载网络、配置权重
def get_YOLO(configPath, weightsPath):
    print("Loading YOLO from disk...")
    net = cv.dnn.readNetFromDarknet(configPath, weightsPath)  # #  利用下载的文件
    return net


# 加载图片、转为blob格式、送入网络输入层
def get_blobImg(img, shape=(256, 256)):
    #blobImg = cv.dnn.blobFromImage(img, 1.0/255.0, (416, 416), None, True, False)   # # net需要的输入是blob格式的，用blobFromImage这个函数来转格式
    blobImg = cv.dnn.blobFromImage(img, 1.0 / 255.0, shape, None, True, False)
    return blobImg


def get_output(net, blobImg):
    net.setInput(blobImg)  # # 调用setInput函数将图片送入输入层

    # 获取网络输出层信息（所有输出层的名字），设定并前向传播
    outInfo = net.getUnconnectedOutLayersNames()  # # 前面的yolov3架构也讲了，yolo在每个scale都有输出，outInfo是每个scale的名字信息，供net.forward使用
    start = time.time()
    layerOutputs = net.forward(outInfo)  # 得到各个输出层的、各个检测框等信息，是二维结构。
    end = time.time()
    print("YOLO took {:.6f} seconds".format(end - start))
    return layerOutputs


def get_bboxes(layerOutputs, highWeight: tuple):
    # 过滤layerOutputs
    # layerOutputs的第1维的元素内容: [center_x, center_y, width, height, objectness, N-class score data]
    # 过滤后的结果放入：
    (H, W) = highWeight
    boxes = [] # 所有边界框（各层结果放一起）
    confidences = [] # 所有置信度
    classIDs = [] # 所有分类ID

    # # 1）过滤掉置信度低的框框
    for out in layerOutputs:  # 各个输出层
        for detection in out:  # 各个框框
            # 拿到置信度
            scores = detection[4:]  # 各个类别的置信度, 前四个是box位置
            classID = np.argmax(scores)  # 最高置信度的id即为分类id
            confidence = scores[classID]  # 拿到置信度

            # 根据置信度筛查
            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])  # 将边界框放会图片尺寸
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # 2）apply non-maxima suppression，nms for further filter
    idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD) # boxes中，保留的box的索引index存入idxs(按自信度从大到小排序)
    return idxs, boxes, confidences, classIDs


def get_labels(labelsPath):
    # 得到labels列表
    with open(labelsPath, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    return labels


def get_useful_boxes(idxs, boxes):
    # get genuine bboxes in idxs
    real = []
    if len(idxs)>0:
        for i in idxs.flatten():
            real.append(boxes[i])
    return real


def get_fish_hw(fish_path, show=False):
    if isinstance(fish_path, str):
        fish = cv.imread(fish_path, cv.IMREAD_GRAYSCALE)
    else:
        fish = cv.cvtColor(fish_path, cv.COLOR_RGB2GRAY)
    y_min, x_min = fish.shape   # fish in here is gray scale
    y_max, x_max = 0, 0
    #ret, t = cv.threshold(fish, 128, 255, cv.THRESH_OTSU)
    ret, t = cv.threshold(fish, 20, 255, cv.THRESH_BINARY)
    t = t[int(y_min * 0.2):int(y_min * 0.8), int(x_min * 0.2):int(x_min * 0.8)]
    if show:
        cv.imshow('crop', t)
    contours = cv.findContours(t, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    for contour in contours:
        if len(contour) < 50:   # regard it as outlier
            continue
        for point in contour:
            x, y = point.flatten()
            if x < x_min:
                x_min = x
            if x > x_max:
                x_max = x
            if y < y_min:
                y_min = y
            if y > y_max:
                y_max = y
    if show:
        fish = cv.cvtColor(fish, cv.COLOR_GRAY2RGB)
        cv.rectangle(fish, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=1)
        cv.drawContours(fish, contours, -1, (255, 2, 0), 2)
        cv.imshow('LW result', fish)
        cv.waitKey(0)
        cv.destroyAllWindows()
    return [x_max-x_min, y_max-y_min]     # return x, y


def fast_video_process(v_path, net, fishsize, laserstation, shape=None):
    v = cv.VideoCapture(v_path)
    frames = get_all_frames(v)
    hw = frames[0].shape[:2]
    scaner = FishScanProcessing(hw[0], hw[1], fish_size=fishsize, laser=laserstation)
    blobImgs = cv.dnn.blobFromImages(frames, 1./255., shape)
    start = time.time()
    print('Starting process video....')
    outs = get_output(net, blobImgs)
    for n in range(len(frames)):
        out = [outs[0][n], outs[1][n], outs[2][n]]
        idxs, boxes, confidences, classIDs = get_bboxes(out, hw)
        real_boxes = get_useful_boxes(idxs, boxes)
        scaner.scan(real_boxes)
    print('Total num: {}'.format(scaner.get_count()))
    print('Runtime: {:.4f}'.format(time.time()-start))
    return


def video_process(v_path, net, fishsize, laserstation, labelspath, show=False):
    from visTool import show_effect_picture  # avoid circular dependent imports

    v = cv.VideoCapture(v_path)
    have, photo = v.read()
    hw = photo.shape[:2]
    scanner = FishScanProcessing(hw[0], hw[1], fish_size=fishsize, laser=laserstation)
    print("Count FishFish....")
    start = time.time()
    while have:
        img = photo.copy()
        blobImg = get_blobImg(img)
        layerOutputs = get_output(net, blobImg)
        idxs, boxes, confidences, classIDs = get_bboxes(layerOutputs, hw)
        real_boxes = get_useful_boxes(idxs, boxes)
        scanner.scan(real_boxes)
        if show:
            show_effect_picture(photo, img, idxs, boxes, confidences,
                                classIDs, labelspath, laserstation, scanner.get_count())
        have, photo = v.read()
    print('Runtime: {:.4f}s'.format((time.time() - start)))
    print('Fish Amount:{}'.format(scanner.get_count()))
    return


contractRate = 0.3
kernel3 = np.ones((3, 3), dtype=np.uint8)
kernel5 = np.ones((5, 5), dtype=np.uint8)
kernelCross = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))


def separate(img, show):
    # img = background
    bg = img.copy()
    #cv.imshow('bg', bg)
    bg = cv.erode(bg, kernel3, iterations=1)
    bg = cv.morphologyEx(bg, cv.MORPH_CLOSE, kernel3)

    dis_transform = cv.distanceTransform(bg, cv.DIST_L2, cv.DIST_MASK_5)
    _, sure_fg = cv.threshold(dis_transform, contractRate*dis_transform.max(), 255, 0)
    sure_fg = cv.morphologyEx(sure_fg, cv.MORPH_OPEN, kernel=kernelCross, iterations=1)
    sure_fg = np.uint8(sure_fg)
    #cv.imshow('fg', sure_fg)
    #cv.waitKey()
    contours = cv.findContours(sure_fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
    contours = [contour for contour in contours if len(contour) > 10]
    n = len(contours)
    if n <= 1:
        if show:
            cv.imshow('background', bg)
            cv.imshow('foreground', sure_fg)
            cv.waitKey()
        return img
    else:
        samples = []
        labels = []
        for i, contour in enumerate(contours):
            cimg = np.zeros_like(sure_fg)
            cv.drawContours(cimg, [contour], 0, 255, cv.FILLED)
            x, y = np.where(cimg == 255)
            c = np.vstack((x, y))
            samples.append(c.transpose((1, 0)))
            labels.append(np.zeros_like(x) + i)
    samples = np.concatenate(samples)
    labels = np.concatenate(labels)
    new_bi = img.copy()
    h, s = sure_fg.shape
    clf = LogisticRegression().fit(samples, labels)
    for i in range(len(clf.coef_)):
        w, b = clf.coef_[i], clf.intercept_[i]
        f = lambda x: int(-(b + x*w[0])/w[1])
        for x in range(h):
            y = f(x)
            new_bi[x, y-2:y+2] = 0
    if show:
        if n > 2:
            print('This local patch contains components more than 2, causing decision boundary there showed has wrong')
        cv.imshow('background', bg)
        cv.imshow('foreground', sure_fg)
        pixels = np.array([[x, y] for x in range(h) for y in range(s)], dtype=np.int32)
        r = clf.predict(pixels)
        db = np.zeros_like(sure_fg, dtype=np.uint8)
        c1 = pixels[np.where(r == 1)]
        for p in c1:
            db[p[0], p[1]] = 255
        db = cv.bitwise_not(db) if len(np.where(db == 0)[0]) > .5*h*s else db
        cv.imshow('decision boundary', db)
        cv.waitKey()
    return new_bi


def refine_bboxes(img, useful_bboxes, ground='white', display=False, show=False):
    bboxes = np.array(useful_bboxes, dtype=np.int)
    bboxes = np.maximum(bboxes, 0)
    new = []
    for bbox in bboxes:
        x, y, w, h = bbox
        loc = img[y:y+h, x:x+w]
        loc = cv.cvtColor(loc, cv.COLOR_BGR2HSV) if ground != 'white' else loc
        gray = cv.cvtColor(loc, cv.COLOR_RGB2GRAY)
        _, th = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)    # white background
        #bi = cv.morphologyEx(th2, cv.MORPH_CLOSE, kernel5)
        th = cv.bitwise_not(th) if ground != 'black' else th
        bi = separate(img=th, show=show) if ground != 'white' else th
        contours = cv.findContours(bi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        contoursLen = map(len, contours)
        points = max(contoursLen)
        for contour in contours[::-1]:
            if len(contour) == points:
                points = contour
                break
        #s = contour_separate(bi, points)
        rect = cv.minAreaRect(points)   # find white area
        box = cv.boxPoints(rect)
        box = np.int32(box)     # each box is 4 * 2 (left top, right top, right bottom, left bottom) * (x, y)
        nx = box[:, 0, np.newaxis] + x
        ny = box[:, 1, np.newaxis] + y
        nxy = np.concatenate((nx, ny), axis=1)
        # show differences above connected components
        if show:
            c = 255 if ground == "black" else 0
            cv.imshow('original', loc)
            cv.imshow('threshold', th)
            cv.imshow('new binary', bi)
            #cv.imshow('watershed', sep)
            areas = [cv2.contourArea(contours[i]) for i in range(len(contours))]
            idx = np.argmax(areas)
            [cv2.fillPoly(bi, [contours[i]], 0) for i in range(len(contours)) if i != idx]
            cv.imshow('max connected component', bi)
            [cv.circle(gray, box[i], 1, [0, 0, 0], 2) for i in range(len(box))]
            cv.drawContours(gray, [points], 0, c, 2)
            cv.drawContours(gray, [box], 0, c, 2)
            [cv.putText(gray, str(i), j, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=c) for i, j in enumerate(box)]
            cv.imshow('result', gray)
            cv.waitKey()
        new.append(nxy)
    [cv.drawContours(img, [new[i]], 0, (255, 0, 0), 1) for i in range(len(new))] if display else None
    return new


class BBoxRefiner():
    def __init__(self, background, fgRate, display=False, show=False):
        self.fgRate = fgRate
        self.kernel3 = np.ones((3, 3), dtype=np.uint8)
        self.kernel5 = np.ones((5, 5), dtype=np.uint8)
        self.kernelCross = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
        self.display = display
        self.show = show
        self.bg = background
        self.color = 255 if background == 'black' else 0
        self.temp = np.array([0], dtype=np.uint8)

    def refine(self, img, bboxes):
        bboxes = np.array(bboxes, dtype=np.int)
        bboxes = np.maximum(bboxes, 0)
        new = []
        for bbox in bboxes:
            x, y, w, h = bbox
            loc = img[y:y + h, x:x + w]
            loc = cv.cvtColor(loc, cv.COLOR_BGR2HSV) if self.bg not in ('white', 'black') else loc
            gray = cv.cvtColor(loc, cv.COLOR_RGB2GRAY)
            _, th = cv.threshold(gray, 20, 255, cv.THRESH_BINARY)
            notTh = cv.bitwise_not(th) if self.bg != 'black' else th.copy()     # all convert to black background
            #notTh = cv.morphologyEx(notTh, cv.MORPH_CLOSE, self.kernel3, iterations=1)  # 如果有白边
            self.bound(th, notTh)
            holes = self.fillHoleByContours(notTh)
            notTh += holes
            #self.deep_seg(notTh, self.show)
            bi = self.separate(notTh) if self.bg != 'white' else notTh.copy()
            contours = cv.findContours(bi, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
            contoursLen = map(cv.contourArea, contours)
            points = max(contoursLen)
            for contour in contours[::-1]:
                if cv.contourArea(contour) == points:
                    points = contour
                    break
            rect = cv.minAreaRect(points)  # find white area
            box = cv.boxPoints(rect)
            box = np.int32(box)  # each box is 4 * 2 (left top, right top, right bottom, left bottom) * (x, y)
            nx = box[:, 0, np.newaxis] + x
            ny = box[:, 1, np.newaxis] + y
            nxy = np.concatenate((nx, ny), axis=1)
            # show differences above connected components
            if self.show:
                cv.imshow('original', loc)
                cv.imshow('threshold', th)
                cv.imshow('not threshold', notTh)
                cv.imshow('filled hole', holes)
                cv.imshow('new binary', bi)
                areas = [cv2.contourArea(contours[i]) for i in range(len(contours))]
                idx = np.argmax(areas)
                [cv2.fillPoly(bi, [contours[i]], 0) for i in range(len(contours)) if i != idx]
                cv.imshow('max connected component', bi)
                [cv.circle(gray, box[i], 1, [0, 0, 0], 2) for i in range(len(box))]
                cv.drawContours(gray, [points], 0, self.color, 2)
                cv.drawContours(gray, [box], 0, self.color, 2)
                [cv.putText(gray, str(i), j, fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=1, color=self.color) for i, j in
                 enumerate(box)]
                cv.imshow('result', gray)
                cv.waitKey()
            new.append(nxy)
        [cv.drawContours(img, [new[i]], 0, (255, 255, 255), 1) for i in range(len(new))] if self.display else None
        return new

    def separate(self, img):
        bg = img.copy()
        bg = cv.erode(bg, self.kernel5, iterations=1)
        dis_transform = cv.distanceTransform(bg, cv.DIST_L2, cv.DIST_MASK_5)
        _, sure_fg = cv.threshold(dis_transform, self.fgRate * dis_transform.max(), 255, 0)
        sure_fg = cv.morphologyEx(sure_fg, cv.MORPH_OPEN, kernel=self.kernelCross, iterations=1)
        sure_fg = np.uint8(sure_fg)
        seg = self.deep_seg(sure_fg, self.show)
        fg = sure_fg - cv.bitwise_and(seg, sure_fg)
        h, s = sure_fg.shape
        contours = cv.findContours(fg, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        contours = [contour for contour in contours if len(contour) > 25]

        n = len(contours)
        if n <= 1:
            if self.show:
                cv.imshow('background', bg)
                cv.imshow('foreground', sure_fg)
            return img
        else:
            samples = []
            labels = []
            for i, contour in enumerate(contours):
                cimg = np.zeros_like(sure_fg)
                cv.drawContours(cimg, [contour], 0, 255, cv.FILLED)
                x, y = np.where(cimg == 255)
                c = np.vstack((x, y))
                samples.append(c.transpose((1, 0)))
                labels.append(np.zeros_like(x) + i)
        samples = np.concatenate(samples)
        labels = np.concatenate(labels)
        new_bi = img.copy()
        clf = LogisticRegression().fit(samples, labels)
        for i in range(len(clf.coef_)):
            w, b = clf.coef_[i], clf.intercept_[i]
            f = lambda x: int(-(b + x * w[0]) / w[1])
            xx = np.linspace(0, h, num=h * 10)
            for x in xx:
                y = f(x)
                x = int(x)
                if y > 0:
                    new_bi[x - 2:x + 2, y - 2:y + 2] = 0
                #cv.imshow('new bi', new_bi)
                #cv.waitKey()
        if self.show:
            if len(clf.coef_) >= 2:
                print('This local patch contains components more than 2,'
                      ' causing decision boundary there showed has wrong')
            cv.imshow('background', bg)
            cv.imshow('foreground', fg)
            pixels = np.array([[x, y] for x in range(h) for y in range(s)], dtype=np.int32)
            r = clf.predict(pixels)
            db = np.zeros_like(sure_fg, dtype=np.uint8)
            c1 = pixels[np.where(r == 1)]
            for p in c1:
                db[p[0], p[1]] = 255
            db = cv.bitwise_not(db) if len(np.where(db == 0)[0]) > .5 * h * s else db
            cv.imshow('decision boundary', db)
        return new_bi

    def melt(self, img, n):
        melted = cv.erode(img, self.kernel3)
        melted[0, :], melted[-1, :], melted[:, 0], melted[:, -1] = img[0, :], img[-1, :], img[:, 0], img[:, -1]
        cv.imshow('melted', melted)
        if np.all(melted == img):
            return 0
        contours = cv.findContours(melted, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
        contours = [contour for contour in contours if len(contour) > 6]
        nn = len(contours)
        if nn == n:
            return self.melt(melted, n)
        elif nn > n:
            for contour in contours:
                t = np.zeros_like(melted, dtype=np.uint8)
                #cv.drawContours(t, [contour], 0, 255)
                ellipse = cv.fitEllipse(contour)
                #cv.drawContours(t, [ellipse], 0, 255, cv.FILLED)
                cv.ellipse(t, ellipse, color=255, thickness=1)
                contour = cv.findContours(t, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)[0]
                cv.drawContours(t, contour, 0, 255, cv.FILLED)
                cv.imshow('ellipse', t)
                cv.waitKey()
                if self.melt(t, 1) == 1:
                    self.temp += t
        elif nn + 1 == n:
            return 1
        return

    def fillHoleByContours(self, bi):
        notBi = cv.bitwise_not(bi)
        #cv.imshow('not bi', notBi)
        new = np.zeros_like(notBi)
        h, w = new.shape
        contours = cv.findContours(notBi, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)[0]
        for contour in contours:
            contour = np.squeeze(contour)
            if contour.ndim == 1:
                new[contour[1], contour[0]] = 255
                continue
            # judging length in case that two fishtails compose a big hole
            elif cv.arcLength(contour, True) > 77 \
                    or np.any(contour == 0) or np.any(contour[:, 0] == w-1) or np.any(contour[:, 1] == h-1):
                continue
            cv.drawContours(new, [contour], 0, 255, cv.FILLED)
        return new

    def deep_seg(self, img, show):
        new = np.zeros_like(img, dtype=np.uint8)
        contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)[0]
        contours = [contour for contour in contours if cv.arcLength(contour, True) > 20]
        points = set()
        for contour in contours:
            epsilon = 0.04 * cv.arcLength(contour, True)
            approx = cv.approxPolyDP(contour, epsilon, False)
            approx = np.squeeze(approx).tolist()
            approx = list(approx)
            n = len(approx)
            '''[cv.circle(img, p, 5, 255, 1) for p in approx]
            cv.imshow('circle', img)
            cv.waitKey()'''
            if n == 2:
                continue
            elif n > 2:
                for i in range(n):
                    p1 = approx.pop()
                    for p2 in approx:
                        for p3 in approx:
                            if p2 == p3:
                                continue
                            mask = np.zeros_like(img, dtype=np.uint8)
                            triangle = np.array([p1, p2, p3], dtype=np.int32)
                            s1 = triangle[1] - triangle[0]
                            s2 = triangle[2] - triangle[0]
                            s3 = triangle[2] - triangle[1]
                            ls = np.linalg.norm(np.array([s1, s2, s3]), axis=1)     # length of sides
                            s = np.sum(ls) / 2
                            a = np.sqrt(s*(s-ls[0])*(s-ls[1])*(s-ls[2])) + 10e-6
                            cv.drawContours(mask, [triangle], 0, 255, cv.FILLED)
                            ais = len(np.where(cv.bitwise_and(mask, img) == 255)[0])   # area of intersection
                            p = ais / a
                            if p < 0.1:
                                idx = np.argmin(ls)
                                if idx == 0:
                                    for i in (0, 1):
                                        k, b = self.two_points_seek_line(triangle[2], triangle[i])
                                        self.draw_line(k, b, new, 255)
                                elif idx == 1:
                                    for i in (0, 2):
                                        k, b = self.two_points_seek_line(triangle[1], triangle[i])
                                        self.draw_line(k, b, new, 255)
                                else:
                                    for i in (2, 1):
                                        k, b = self.two_points_seek_line(triangle[0], triangle[i])
                                        self.draw_line(k, b, new, 255)
                                points.update(map(tuple, triangle))
        for x, y in points:
            #cv.circle(new, (x, y), 5, 255, cv.FILLED)
            pass
        if show and points:
            sh = new.copy()
            [cv.circle(sh, (x, y), 5, 255, cv.FILLED) for x, y in points]
            cv.imshow('deep seg', sh)
        return new

    def bound(self, be, af):
        assert be.shape == af.shape, 'shapes is not matched'
        h, w = be.shape
        af[0, :], af[h-1, :], af[:, 0], af[:, w-1] = be[0, :], be[h-1, :], be[:, 0], be[:, w-1]
        return af

    def two_points_seek_line(self, p1, p2):
        a = p1 - p2
        k = a[1] / a[0]
        b = p1[1] - k * p1[0]
        return k, b

    def draw_line(self, k, b, img, c):
        ax = max(img.shape)
        xx = np.linspace(1, ax, ax*2, endpoint=False)
        yy = xx * k + b
        for x, y in zip(xx, yy):
            if ax > y > 0:
                x = int(x)
                y = int(y)
                try:
                    img[y-1:y+1, x-1:x+1] = c
                except:
                    return