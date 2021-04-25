import cv2
import numpy as np
from common import Point, Size, Rect

class RarityRecognizer:
    _star_img = None
    _star_mask_img = None
    _roi = Rect(30, 83, 90, 16)
    _method = cv2.TM_SQDIFF_NORMED
    _threshold = 0.05
    _min_interval = 6

    _precise_star_xs = [
        [41],
        [36, 46],
        [30, 41, 51],
        [25, 36, 46, 57],
        [20, 30, 41, 51, 61],
        [15, 25, 36, 46, 57, 67],
        [10, 20, 30, 41, 51, 61, 71],
        [7, 17, 26, 36, 45, 55, 64, 74]
    ]
    _precise_star_y = 6

    def __init__(self):
        self._star_img = cv2.imread('star.png', cv2.IMREAD_COLOR)
        self._star_mask_img = cv2.imread('star_mask.png', cv2.IMREAD_COLOR)

    def recognize(self, input_img):
        # Consider ROI only
        input_roi_img = input_img[
            self._roi.y : self._roi.y + self._roi.h,
            self._roi.x : self._roi.x + self._roi.w,
            :]

        # For each of R, G, B
        star_h, star_w, _ = self._star_img.shape
        scores = np.zeros((self._roi.h - star_h + 1, self._roi.w - star_w + 1))
        for channel in range(3):
            result = cv2.matchTemplate(
                    input_roi_img[:, :, channel],
                    self._star_img[:,:,channel],
                    self._method,
                    mask = self._star_mask_img[:,:,channel])
            scores = np.add(scores, np.square(result))

        # Filter by the threshold, take x-axis only
        xys = np.where(scores <= self._threshold)
        ys = xys[0]
        xs = xys[1]

        # Prevent count a star twice
        xs.sort()
        last_x = -999999
        star_groups = []
        for x in xs:
            if x - last_x >= self._min_interval:
                last_x = x
                star_groups.append([])
            star_groups[-1].append(x)

        if len(star_groups) == 0 or len(star_groups) > 8:
            return (0, Point())

        best_offset = Point()
        best_score = 0
        for candidate in star_groups[0]:
            dx = candidate - self._precise_star_xs[len(star_groups) - 1][0]
            score = 0
            for g in range(len(star_groups)):
                for x in star_groups[g]:
                    if x - self._precise_star_xs[len(star_groups) - 1][g] == dx:
                        score += 1
            if score > best_score:
                best_score = score
                best_offset.x = dx;

        ys = ys.tolist()
        best_offset.y = max(set(ys), key=ys.count) - self._precise_star_y

        return (len(star_groups), best_offset)
