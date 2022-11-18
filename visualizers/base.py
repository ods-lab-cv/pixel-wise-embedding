import matplotlib.pyplot as plt
import numpy as np


from vectorizations.base import BaseVectorization


class BaseVisualizer:
    def __init__(self, vectorization: BaseVectorization):
        self.vectorization = vectorization
        self.fig, self.axs = plt.subplots(figsize=(16, 16))
        self.images_paths = None

    def show(self, images_paths):
        self.images_paths = images_paths
        outs = self.vectorization.vectorization(images_paths, always_update=True)
        pim = np.concatenate(outs, axis=1)
        self.axs.imshow(pim)

        plt.show()

