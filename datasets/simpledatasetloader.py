
import numpy as np
import cv2
import os


class SimpleDatasetLoader:
    def __init__(self, preprocessors=None):
        self._preprocessors = preprocessors

    def load(self, image_paths, verbose_level=-1):
        data = []
        labels = []

        for (i, path) in enumerate(image_paths):
            image = cv2.imread(path)
            label = path.split(os.path.sep)[-2]

            if self._preprocessors is not None:
                for p in self._preprocessors:
                    image = p.preprocess(image)

            data.append(image)
            labels.append(label)

            if verbose_level > 0 and i > 0 and (i+1) % verbose_level == 0:
                print('[INFO] processed {}/{}'.format(i+1, len(image_paths)))

        return (np.array(data), np.array(labels))
