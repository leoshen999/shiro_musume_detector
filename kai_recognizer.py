import cv2
import numpy as np
from common import Point, Size, Rect

class KaiRecognizer:
    _kai_img = None
    _kai_mask_img = None
    _roi = Rect(5, 90, 40, 24)
    _method = cv2.TM_SQDIFF_NORMED
    _threshold = 0.05

    def __init__(self):
        self._kai_img = cv2.imread('kai.png', cv2.IMREAD_COLOR)
        self._kai_mask_img = cv2.imread('kai_mask.png', cv2.IMREAD_COLOR)

    def recognize(self, input_img):
        # Consider ROI only
        input_roi_img = input_img[
            self._roi.y : self._roi.y + self._roi.h,
            self._roi.x : self._roi.x + self._roi.w,
            :]

        # For each of R, G, B
        kai_h, kai_w, _ = self._kai_img.shape
        scores = np.zeros((self._roi.h - kai_h + 1, self._roi.w - kai_w + 1))
        for channel in range(3):
            result = cv2.matchTemplate(
                    input_roi_img[:, :, channel],
                    self._kai_img[:,:,channel],
                    self._method,
                    mask = self._kai_mask_img[:,:,channel])
            scores = np.add(scores, np.square(result))

        # Filter by the threshold
        result = np.any(scores.flatten() <= self._threshold)
        return result

