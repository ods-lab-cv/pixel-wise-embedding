import glob
import os
import torch
import segmentation_models_pytorch as smp

from visualizers.base import BaseVisualizer
from vectorizations.pca import PCAVectorization


MODEL_PATH = r'C:\Repositories\pixel-wise-embedding\weights\pixel_wise_encoder_v2.pt'
DATA_PATH = r'C:\Repositories\pixel-wise-embedding\data\test_images\cars'
IMAGE_SIZE = (512, 512)
PCA_EVERY = 10
DEVICE = 'cpu'

if __name__ == '__main__':
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    visualizer = BaseVisualizer(
        vectorization=PCAVectorization(
            model=model,
            image_size=IMAGE_SIZE,
            pca_every=PCA_EVERY,
            device=DEVICE,
        )
    )
    visualizer.show(glob.glob(os.path.join(DATA_PATH, '*')))
