import torch
import numpy as np
import typing as tp
import cv2
from torchvision.transforms import ToTensor


class BaseVectorization:
    def __init__(
            self,
            model: torch.nn.Module,
            image_size: tp.Tuple[int, int],
            device: str = 'cuda',
    ):
        self.model = model
        self.image_size = image_size

        self.device = device

        self.imgs = None
        self.outs = None

    def _predict(self, imgs: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            outs = self.model(imgs.to(self.device)).detach().cpu().numpy()
        outs = np.moveaxis(outs, 1, -1)
        return outs

    def _vectorization(self, imgs: np.ndarray, outs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def vectorization(self, imgs: tp.Union[tp.List[str], torch.Tensor], always_update: bool = True) -> np.ndarray:
        if always_update or (self.imgs is None) or (self.outs is None):
            if isinstance(imgs, list):
                self.imgs = self._read_images(imgs)
            self.outs = self._predict(self.imgs)
            self.imgs = self.imgs.detach().cpu().numpy()
            self.imgs = np.moveaxis(self.imgs, 1, -1)
        vect = self._vectorization(self.imgs, self.outs)
        return vect

    def _read_images(self, images_paths: tp.List[str]) -> torch.Tensor:
        images = []
        for image_path in images_paths:
            im = cv2.imread(image_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = (cv2.resize(im, self.image_size) / 255).astype('float32')
            im = ToTensor()(im)
            images.append(im.unsqueeze(0))
        images = torch.cat(images, dim=0)
        return images

