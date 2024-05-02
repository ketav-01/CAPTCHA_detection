import cv2
from imutils import contours
import numpy as np


def Segmentation(path):
    image = cv2.imread(path)
    image = cv2.resize(image, ((image.shape[0]*2), (image.shape[1]*2)))
    image = cv2.resize(image, ((1920), (1080)))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(
        gray, 250, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
    thresh = 255-thresh
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    cnts, _ = contours.sort_contours(cnts, method="left-to-right")

    max_area = 0

    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        max_area = max(max_area, area)

    ROI_number = 0
    prevx = 0

    path_list = []

    for c in cnts:
        image_path = "SEGMENTED_IMAGES"
        x, y, w, h = cv2.boundingRect(c)
        area = w*h
        area_prev = 0
        if area > max_area/8.5:
            if x > 1*prevx:
                area_prev = w*h
                side = (w*h)**0.5
                prevx = x+w
                ROI = gray[y-10:y+h+10, max(0, x-(int)(0.25*side))
                                            :min(image.shape[1], x+w+(int)(0.25*side))]
                ROI = cv2.threshold(
                    ROI, 250, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
                ROI = cv2.resize(ROI, ((28), (28)))
                ROI = ROI[:, 4:-4]
                ROI = cv2.resize(ROI, ((28), (28)))
                image_path += f'\ROI_{ROI_number}.jpg'
                cv2.imwrite(image_path, ROI)
                cv2.rectangle(image, (x-(int)(0.25*side), y-10),
                              (x + w+(int)(0.25*side), y + h+10), (36, 255, 12), 1)
                ROI_number += 1
                path_list.append(image_path)
            elif area_prev < w*h:
                #             prevy = y+h
                area_prev = w*h
                side = (w*h)**0.5
                prevx = x+w
                ROI = gray[y-10:y+h+10, max(0, x-(int)(0.25*side))
                                            :min(image.shape[1], x+w+(int)(0.25*side))]
                ROI = cv2.threshold(
                    ROI, 250, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]
                ROI = cv2.resize(ROI, ((28), (28)))
                ROI = ROI[:, 4:-4]
                ROI = cv2.resize(ROI, ((28), (28)))
                image_path += f'\ROI_{ROI_number-1}.jpg'
                cv2.imwrite(image_path, ROI)
                cv2.rectangle(image, (x-(int)(0.25*side), y-10),
                              (x + w+(int)(0.25*side), y + h+10), (36, 255, 12), 1)

    return path_list
