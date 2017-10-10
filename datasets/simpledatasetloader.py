
import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self._preprocessors = preprocessors

        if self._preprocessors is None:
           self._preprocessors = []


    def load(self, image_paths, verbose_level=-1):
        data = []
        labels = []

        for (i, path) in enumerate(image_paths):
            image = cv2.imread(path)
            label = path.split(os.path.sep)[-2]
