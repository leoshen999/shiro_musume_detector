from common import Point, Size, Rect
import cv2
import numpy as np
import random

class ScrollbarDetector:
    class Type:
        kUnknown = -1
        kBegin = 0
        kEnd = 1
        kNone = 2

    _scrollbar_begin_img = None
    _scrollbar_end_img = None
    _scrollbar_none_img = None
    _method = cv2.TM_SQDIFF_NORMED
    _threshold = 0.01

    def __init__(self):
        self._scrollbar_begin_img = cv2.imread('scrollbar_begin.png', cv2.IMREAD_GRAYSCALE)
        self._scrollbar_end_img = cv2.imread('scrollbar_end.png', cv2.IMREAD_GRAYSCALE)
        self._scrollbar_none_img = cv2.imread('scrollbar_none.png', cv2.IMREAD_GRAYSCALE)

    def detect(self, img):
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[705:720, 1240:1280]

        begin_score = cv2.matchTemplate(
                grayscale,
                self._scrollbar_begin_img,
                self._method)[0][0]
        end_score = cv2.matchTemplate(
                grayscale,
                self._scrollbar_end_img,
                self._method)[0][0]
        none_score = cv2.matchTemplate(
                grayscale,
                self._scrollbar_none_img,
                self._method)[0][0]

        best_score = self._threshold
        best_type = self.Type.kUnknown
        if begin_score < best_score:
            best_score = begin_score
            best_type = self.Type.kBegin
        if end_score < best_score:
            best_score = end_score
            best_type = self.Type.kEnd
        if none_score < best_score:
            best_score = none_score
            best_type = self.Type.kNone

        return best_type
