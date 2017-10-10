# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from cv2 import cv2
from skimage.filters import threshold_local, threshold_adaptive
import os
import warnings
import time

warnings.filterwarnings("ignore")

def order_points(pts):
	rect = np.zeros((4, 2), dtype = "float32")
	s = pts.sum(axis = 1)
	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]
	return rect

def four_point_transform(image, pts):
	rect = order_points(pts)
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
	maxWidth = max(int(widthA), int(widthB))
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
	maxHeight = max(int(heightA), int(heightB))
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")
	M = cv2.getPerspectiveTransform(rect, dst)
	warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
	return warped
    
def proccess_card(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    dim = (245, 343)
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

def detect_cards(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    # cv2.imshow('edged', edged)
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)
    return cnts
    
    
def create_labeled_cards():
    image = cv2.imread('data/sample/all_cards.jpg')
    cnts = detect_cards(image)[:120]
    for i, c in enumerate(cnts):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02*peri, True)
        if len(approx) == 4:
            warped = four_point_transform(image, approx.reshape(4, 2))
            card = proccess_card(warped)
            cv2.imwrite('data/labeled_2/card_{}.png'.format(i), card)

def create_labeled_corners():
    image = cv2.imread('data/sample/all_cards.jpg')
    cnts = detect_cards(image)[:130]
    for i_card, c in enumerate(cnts):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            input_card = four_point_transform(image, approx.reshape(4, 2))
            input_card = proccess_card(input_card)
            cv2.imwrite('data/labeled_top_left/card_{}.png'.format(i_card), input_card[5:90,:35])
    
def get_labled_cards():
    labled = {}
    loc = 'data/labeled_2/'
    for file in os.listdir(loc):
        if file.endswith('.png'):
            warped = cv2.imread(loc + file)
            card = proccess_card(warped)
            labled[file.split('.')[0]] = card
    return labled

def get_labled_top_left():
    labled = {}
    loc = 'data/labeled_top_left/'
    for file in os.listdir(loc):
        if file.endswith('.png'):
            warped = cv2.imread(loc + file)
            card = proccess_card(warped)
            labled[file.split('.')[0]] = card
    return labled

def calc_difference(image_1, image_2):
    image_1 = cv2.GaussianBlur(image_1,(5,5),5)
    image_2 = cv2.GaussianBlur(image_2,(5,5),5)
    diff = cv2.absdiff(image_1, image_2)
    diff = cv2.GaussianBlur(diff,(5,5),5)    
    flag, diff_1 = cv2.threshold(diff, 200, 255, cv2.THRESH_BINARY)
    return np.sum(diff)

def top_left_detection(image):
    

def whole_card_detection(image):
    labled = get_labled_cards()
    cnts = detect_cards(image)[:5]
    results = {}
    for i_card, c in enumerate(cnts):
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            input_card = four_point_transform(image, approx.reshape(4, 2))
            input_card = proccess_card(input_card)
            (h, w) = input_card.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, 180, 1.0)
            input_card_rotated = cv2.warpAffine(input_card, M, (w, h))
            diffs = {}
            for label_name, label_card in labled.items():
                diff_1 = calc_difference(input_card, label_card)
                diff_2 = calc_difference(input_card_rotated, label_card)  
                diffs[label_name] = min(diff_1, diff_2)
            results[i_card] = min(diffs, key=diffs.get)
            cv2.imshow('input_card_{}'.format(i_card), input_card)
    return results

if __name__ == '__main__':
    
# =============================================================================
#     results = card_detection(cv2.imread('data/sample/all_cards.jpg'))
#     print(results)
# =============================================================================
    

    
    
    
    
    
    
    
    
    
    
    
    
    