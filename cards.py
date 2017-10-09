# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import cv2.cv2 as cv2
from skimage.filters import threshold_adaptive

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

def exploring():
    all_cards = cv2.imread('data/sample/all.jpg')
    print(type(all_cards))
    print(all_cards.size)
    print(all_cards.shape)
    print(all_cards[0])
    plt.imshow(all_cards)
    plt.show()
    b, g, r = cv2.split(all_cards)
    plt.imshow(b, cmap='gray')
    plt.show()
    all_cards_merged = cv2.cvtColor(all_cards, cv2.COLOR_BGR2RGB)
    plt.imshow(all_cards_merged)
    plt.show()
    
    
if __name__ == '__main__':
    image = cv2.imread('data/sample/7.jpg')
    r = 500 / image.shape[1]
    dim = (500, int(image.shape[0] * r))
    ratio = image.shape[0] / 500.0
    orig = image.copy()
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)   
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 75, 200)
    #cv2.imshow("Image", image)
    #cv2.imshow("Edged", edged)
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:5]
    screenCnts = []
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            screenCnts.append(approx)
    cv2.drawContours(image, cnts, -1, (0, 255, 0), 2)
    cv2.imshow("green", image)
    for i in range(len(screenCnts)):
        warped = four_point_transform(image, screenCnts[i].reshape(4, 2))
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        warped = threshold_adaptive(warped, 251, offset = 10)
        warped = warped.astype("uint8") * 255
        cv2.imshow('warp {}'.format(i), warped)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    