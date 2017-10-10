
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter=cv2.INTER_AREA):
        self._width = width
        self._height = height
        self._inter = inter

    def preprocess(self, image):
        return cv2.resize(image, (self._width, self._height), interpolation=self._inter)
