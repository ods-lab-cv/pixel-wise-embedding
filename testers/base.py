import torch
import typing as tp
import cv2
import os
from torchvision.transforms import ToTensor
import numpy as np


from vectorizations.base import BaseVectorization


class BaseTester:
    def __init__(
            self,
            images_paths: tp.List[str],
            vectorization: BaseVectorization,
            save_folder: str,
            run_every: int = 100,
    ):
        self.vectorization = vectorization
        self.images_paths = images_paths

        self.save_folder = save_folder

        self.run_every = run_every
        self.iter = 0
        self._real_iter = 0

    def save_results(self, pims: np.ndarray):
        for b in range(len(pims)):
            b_name = os.path.split(self.images_paths[b].split('.')[0])[-1]
            folder_path = os.path.join(self.save_folder, b_name)
            os.makedirs(folder_path, exist_ok=True)
            cv2.imwrite(os.path.join(folder_path, str(self.iter) + '.jpg'), (pims[b]/pims[b].max() * 255))

    def test(self):
        if self._real_iter % self.run_every == 0:
            vectors = self.vectorization.vectorization(self.images_paths)
            self.save_results(vectors)
            self.iter += 1
        self._real_iter += 1
