import numpy as np
import pyautogui
import time
import cv2
from game_frame_detector import GameFrameDetector
from scrollbar_detector import ScrollbarDetector
from kai_recognizer import KaiRecognizer
from rarity_recognizer import RarityRecognizer
from common import Point, Size, Rect

def takeScreenshot():
    raw_captured_data = pyautogui.screenshot()
    return cv2.cvtColor(np.array(raw_captured_data), cv2.COLOR_RGB2BGR)

def click(point):
    pyautogui.moveTo(point.x, point.y, 0.1)
    time.sleep(0.1)
    pyautogui.mouseDown(button='left')
    time.sleep(0.1)
    pyautogui.mouseUp(button='left')

def scroll(point, distance):
    pyautogui.moveTo(point.x, point.y, 0.1)
    time.sleep(0.1)
    pyautogui.scroll(distance)

def affineTransform(point, M):
    src = np.array([[[point.x, point.y]]], dtype=np.float32)
    dst = cv2.transform(src, M)
    result = Point(dst[0][0][0], dst[0][0][1])
    return result

def are_same_images(img1, img2, mask):
    score = 0.0
    for channel in range(3):
        result = cv2.matchTemplate(
                img1[:, :, channel],
                img2[:, :, channel],
                cv2.TM_SQDIFF_NORMED)
        score = score + result[0][0] * result[0][0]
    return score < 0.0001

def main():
    game_frame_detector = GameFrameDetector()
    scrollbar_detector = ScrollbarDetector()
    kai_recognizer = KaiRecognizer()
    rarity_recognizer = RarityRecognizer()

    # Find the game frame first.
    screenshot = takeScreenshot()
    # screen_height, screen_width, _ = screenshot.shape
    frame_to_screen_transform, screen_to_frame_transform = game_frame_detector.detect(screenshot)
    if frame_to_screen_transform == None:
        print("Game frame is not found.")
        return

    for group in range(2):
        # Click the button to view all musumes of current group
        button_pt = Point(624, 76)
        if group == 1: button_pt = Point(1113, 76)
        click_pt = Point(button_pt.x + 79, button_pt.y + 28)
        click_pt_in_screen = affineTransform(click_pt, frame_to_screen_transform)
        click(click_pt_in_screen)
        group_img = cv2.imread("group" + str(group) + ".png", cv2.IMREAD_COLOR)
        group_mask_img = cv2.imread("group_mask.png", cv2.IMREAD_GRAYSCALE)

        # Wait the screen updating by checking the button img.
        while(True):
            game_frame = cv2.warpAffine(takeScreenshot(), screen_to_frame_transform, (1280, 720))
            if are_same_images(
                    game_frame[button_pt.y:button_pt.y+57, button_pt.x:button_pt.x+158],
                    group_img,
                    group_mask_img):
                break
            time.sleep(0.5)

        # Detect whether a scrollbar exists.
        scrollbar_type = scrollbar_detector.detect(game_frame)
        assert scrollbar_type == ScrollbarDetector.Type.kBegin or scrollbar_type == ScrollbarDetector.Type.kNone
        can_scroll = (scrollbar_type == ScrollbarDetector.Type.kBegin)

        # If there is a scrollbar, there will be an extra offset for the last 6x4 musumes' images
        last_frame_extra_offset = 0
        if can_scroll:
            last_frame_extra_offset = 47

        # Keep scrolling if possible
        while can_scroll:
            # Read 6 musumes of a row
            for x in range(6):
                point = Point(497 + x * 124, 192)
                musume_img = game_frame[point.y:point.y+114, point.x:point.x+120]
                # XXX: Detect...
                kai = kai_recognizer.recognize(musume_img)
                rarity, _ = rarity_recognizer.recognize(musume_img)
                print(kai, rarity)

            # Scroll to next row
            scrollbar_img = game_frame[185:714, 1253:1264]
            point = Point(625, 239)
            point_in_screen = affineTransform(point, frame_to_screen_transform)
            scroll(point_in_screen, -1)

            # Wait the screen updating by checking the scrollbar img changing.
            while(True):
                game_frame = cv2.warpAffine(takeScreenshot(), screen_to_frame_transform, (1280, 720))
                if not are_same_images(game_frame[185:714, 1253:1264], scrollbar_img, None):
                    break
                time.sleep(0.3)

            # If the scrollbar is at the end. Use the rule below to read the last 6x4 musumes.
            scrollbar_type = scrollbar_detector.detect(game_frame)
            assert scrollbar_type == ScrollbarDetector.Type.kBegin or scrollbar_type == ScrollbarDetector.Type.kEnd
            can_scroll = (scrollbar_type == ScrollbarDetector.Type.kBegin)

        # Read the last 6x4 musumes.
        ended = False
        for y in range(4):
            for x in range(6):
                point = Point(497 + x * 124, 192 + last_frame_extra_offset + y * 121)
                musume_img = game_frame[point.y:point.y+114, point.x:point.x+120]
                # XXX: Detect...
                kai = kai_recognizer.recognize(musume_img)
                rarity, _ = rarity_recognizer.recognize(musume_img)
                print(kai, rarity)

                # If rarity is 0, there should not be a musume img.
                if rarity == 0:
                    ended = True
                    break
            if ended: break

if __name__ == "__main__":
    main()
