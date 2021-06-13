import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from babydetector import get_labels


def video_batch_show(imgs, column=16):
    n = len(imgs)
    show_row = n // column + 1
    flg = plt.figure(figsize=(330, 330))
    for i, img in enumerate(imgs):
        plt.subplot(show_row, column, i+1)
        plt.imshow(img)
        plt.yticks([])
        plt.xticks([])
    plt.show()


def show_images_continuously(img):
    plt.close()
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.ion()
    plt.pause(0.3)


def show_image(img):
    plt.imshow(img)
    plt.yticks([])
    plt.xticks([])
    plt.show()


def show_effect_picture(photo, img, idxs, boxes, confidences, classIDs, labelsPath,
                        laserStation, count: int, color=(255, 255, 0)):
    hw = photo.shape[:2]
    if len(idxs) > 0:
        names = get_labels(labelsPath)
        paintBBoxesForOneImage(img, idxs, boxes, confidences, classIDs, names)
        put_line(img, laserStation)
        show_pic = horizontally_joint_photos([photo, img])
        cv.putText(show_pic, "Down count: " + str(count),
                   (hw[1] - 100, hw[0] - 50), cv.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        show_images_continuously(show_pic)

def put_line(img, row, color=(0, 0, 255)):
    h, w, _ = img.shape
    cv.line(img, (0, int(h*row)), (w, int(h*row)), color, thickness=2)
    return


def paintBBoxesForOneImage(img, idxs, boxes, confidences, classIDs, labels, color=(255, 0, 0)):
    if len(idxs) > 0:
        for i in idxs.flatten():  # indxs是二维的，第0维是输出层，所以这里把它展平成1维
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            cv.rectangle(img, (x, y), (x+w, y+h), color, thickness=1)
            text = "{}: {:.1f}".format(labels[classIDs[i]], confidences[i])
            cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return


def horizontally_joint_photos(photos: list):
    return np.hstack(photos)


def drawBBoxesWithKindAndLength(img, bboxes, lens, kinds, color=(255, 0, 0)):
    for i, bbox in enumerate(bboxes):
        x, y, w, h = bbox
        cv.rectangle(img, (x, y), (x + w, y + h), color, thickness=1)
        text = "{}: {:.2f}".format(kinds[i], lens[i])
        cv.putText(img, text, (x, y-5), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return