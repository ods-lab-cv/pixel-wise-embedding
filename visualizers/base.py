import matplotlib.pyplot as plt
import numpy as np
import typing as tp
import cv2


from vectorizations.base import BaseVectorization


class BaseVisualizer:
    def __init__(self, vectorization: BaseVectorization):
        self.vectorization = vectorization
        self.fig, self.axs = plt.subplots(figsize=(16, 16))
        self.images_paths = None
        self.plot_original = None

    def generate_im(self, always_update=True):
        outs = self.vectorization.vectorization(self.images_paths, always_update=always_update)

        plot_outs = np.concatenate(outs, axis=1)
        if len(plot_outs.shape) == 2:
            plot_outs = cv2.cvtColor(plot_outs, cv2.COLOR_GRAY2RGB)
        if self.plot_original:
            plot_imgs = np.concatenate(self.vectorization.imgs, axis=1)
            print(plot_imgs.shape, plot_outs.shape)
            pim = np.concatenate([plot_imgs, plot_outs], axis=0)
        else:
            pim = plot_outs
        return pim

    def show(self, images_paths: tp.List[str], plot_original=True):
        self.images_paths = images_paths
        self.plot_original = plot_original
        pim = self.generate_im(always_update=True)

        self.axs.imshow(pim)

        plt.show()

