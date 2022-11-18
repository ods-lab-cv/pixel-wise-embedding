import numpy as np
import os
from matplotlib.backend_bases import MouseEvent

from visualizers.base import BaseVisualizer
from vectorizations.interactive import InteractiveVectorization

os.environ['KMP_DUPLICATE_LIB_OK'] = str(True)


class InteractiveVisualizer(BaseVisualizer):
    def __init__(self, vectorization: InteractiveVectorization):
        super(InteractiveVisualizer, self).__init__(
            vectorization=vectorization,
        )
        cid = self.fig.canvas.mpl_connect('button_press_event', self.onclick)

    def onclick(self, event: MouseEvent):
        self.vectorization: InteractiveVectorization
        if event.ydata is not None and event.xdata is not None:
            if event.ydata < self.vectorization.image_size[1]:
                target_b = int(event.xdata // self.vectorization.image_size[0])
                x = event.xdata - target_b * self.vectorization.image_size[0]
                y = event.ydata
                self.vectorization.set_pos(x, y, target_b)
                pim = self.generate_im(always_update=False)
                self.axs.imshow(pim)
                self.fig.canvas.draw()
