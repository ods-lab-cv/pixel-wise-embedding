import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import glob
import albumentations as A
import numpy as np
import argparse


from visualizers.base import BaseVisualizer
from vectorizations.interactive import InteractiveVectorization


class InteractiveVisualizer(BaseVisualizer):
    def __init__(self, vectorization: InteractiveVectorization):
        super(InteractiveVisualizer, self).__init__(
            vectorization=vectorization,
        )
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event):
        self.vectorization: InteractiveVectorization
        if event.ydata is not None and event.xdata is not None:
            if event.ydata < self.vectorization.image_size[1]:
                target_b = int(event.xdata // self.vectorization.image_size[0])
                x = event.xdata - target_b * self.vectorization.image_size[0]
                y = event.ydata
                self.vectorization.set_pos(x, y, target_b)
                pims = self.vectorization.vectorization(self.images_paths, always_update=False)
                pim = np.concatenate(pims, axis=1)
                self.axs.imshow(pim)
                self.fig.canvas.draw()
