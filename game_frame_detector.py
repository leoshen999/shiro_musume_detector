from common import Point, Size, Rect
import cv2
import numpy as np
import random

class GameFrameDetector:
    _game_frame_img = None
    _game_frame_mask = None
    _method = cv2.TM_SQDIFF_NORMED

    def __init__(self):
        self._game_frame_img = cv2.imread('game_frame.png', cv2.IMREAD_GRAYSCALE)
        self._game_frame_mask = cv2.imread('game_frame_mask.png', cv2.IMREAD_GRAYSCALE)

    def detect(self, img):
        # TODO: also detect scale, rotation, etc.
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        result = cv2.matchTemplate(
                grayscale,
                self._game_frame_img,
                self._method,
                mask = self._game_frame_mask)

        best_error = np.amin(result)
        best_offset = np.where(result == best_error)

        frame_to_screen = np.array([ [1, 0, best_offset[1][0]], [0, 1, best_offset[0][0]] ], dtype=np.float32)
        screen_to_frame = np.array([ [1, 0, -best_offset[1][0]], [0, 1, -best_offset[0][0]] ], dtype=np.float32)

        return [frame_to_screen, screen_to_frame]
