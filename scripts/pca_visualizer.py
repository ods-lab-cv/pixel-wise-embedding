import glob
import os
import torch
import segmentation_models_pytorch as smp
from torchvision import models as torch_models

from visualizers.base import BaseVisualizer
from vectorizations.pca import PCAVectorization


MODEL_PATH = r'C:\Repositories\pixel-wise-embedding\weights\pixel_wise_encoder_v2.pt'
DATA_PATH = r'C:\Repositories\pixel-wise-embedding\data\test_images\datchiki'
IMAGE_SIZE = (256, 256)
PCA_EVERY = 10
DEVICE = 'cpu'


class Wrapper(torch.nn.Module):
    def __init__(self):
        super(Wrapper, self).__init__()
        self.model = torch_models.segmentation.deeplabv3_resnet101(pretrained=True).to(DEVICE)
        self.model.classifier[3] = torch.nn.Identity()
        self.model.classifier[4] = torch.nn.Identity()
        self.model.aux_classifier = torch.nn.Identity()

    def forward(self, x):
        out = self.model(x)['out']
        return out


if __name__ == '__main__':
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    # model = Wrapper()
    visualizer = BaseVisualizer(
        vectorization=PCAVectorization(
            model=model,
            image_size=IMAGE_SIZE,
            pca_every=PCA_EVERY,
            device=DEVICE,
        )
    )
    visualizer.show(glob.glob(os.path.join(DATA_PATH, '*')))
