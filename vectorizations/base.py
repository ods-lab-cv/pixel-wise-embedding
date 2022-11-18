import torch
import numpy as np
import typing as tp
import cv2
from torchvision.transforms import ToTensor


class BaseVectorization:
    def __init__(
            self,
            model,
            image_size: tp.Tuple[int, int],
            device='cuda',
    ):
        self.model = model
        self.image_size = image_size

        self.device = device

        self.imgs = None
        self.outs = None

    def _predict(self, imgs):
        with torch.no_grad():
            outs = self.model(imgs.to(self.device)).detach().cpu().numpy()
        outs = np.moveaxis(outs, 1, -1)
        return outs

    def _vectorization(self, imgs, outs):
        raise NotImplementedError

    def vectorization(self, imgs: tp.Union[tp.List[str], torch.Tensor], always_update=True):
        if always_update or (self.imgs is None) or (self.outs is None):
            if isinstance(imgs, list):
                self.imgs = self._read_images(imgs)
            self.outs = self._predict(self.imgs)
        vect = self._vectorization(self.imgs, self.outs)
        return vect

    def _read_images(self, images_paths: tp.List[str]):
        images = []
        for image_path in images_paths:
            im = cv2.imread(image_path)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            im = (cv2.resize(im, self.image_size) / 255).astype('float32')
            im = ToTensor()(im)
            images.append(im.unsqueeze(0))
        images = torch.cat(images, dim=0)
        return images

