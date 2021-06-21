import numpy as np
import cv2 as cv
import time
from videoProcessing import FishScanProcessing, get_all_frames


CONFIDENCE = 0.8  # 过滤弱检测的最小概率
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
    print("YOLO took {:.6f} seconds".format(end - start))  # # 可以打印下信息
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

    # 2）应用非最大值抑制(non-maxima suppression，nms)进一步筛掉
    idxs = cv.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD) # boxes中，保留的box的索引index存入idxs(按自信度从大到小排序) non maximum suppression
    return idxs, boxes, confidences, classIDs


def get_labels(labelsPath):
    # 得到labels列表
    with open(labelsPath, 'rt') as f:
        labels = f.read().rstrip('\n').split('\n')
    return labels


def get_useful_boxes(idxs, boxes):
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
    ret, t = cv.threshold(fish, 128, 255, cv.THRESH_OTSU)
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
        cv.imshow('s', fish)
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


def fill_hold(binaryImg, background='white'):
    h, w = binaryImg.shape
    fill = binaryImg.copy()
    mask = np.zeros((h+2, w+2), np.uint8) + 255 if background == 'white' else np.zeros((h+2, w+2), np.uint8)
    r, l = np.where(binaryImg == 255 if background == 'white' else binaryImg == 0)
    seed = (l[0], r[0])
    cv.floodFill(fill, mask, seedPoint=seed, newVal=0 if background == 'white' else 255)
    fill = cv.bitwise_not(fill)
    fill = binaryImg | fill
    cv.imshow('filled', fill)

def refine_bboxes(img, useful_bboxes):
    bboxes = np.array(useful_bboxes, dtype=int)
    bboxes = np.maximum(bboxes, 0)
    for bbox in bboxes:
        x, y, w, h = bbox
        loc = img[y:y+h, x:x+w]
        gray = cv.cvtColor(loc, cv.COLOR_RGB2GRAY)
        #cv.imshow('gray', gray)
        ret2, th2 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY+cv.THRESH_OTSU)
        #cv.imshow('OTSU', th2)
        kernel = np.ones((5,5), dtype=np.uint8)
        bi = cv.morphologyEx(th2, cv.MORPH_CLOSE, kernel)
        cv.imshow('refine binary', bi)
        fill_hold(bi, background='black')
        cv.waitKey()

    cv.destroyAllWindows()