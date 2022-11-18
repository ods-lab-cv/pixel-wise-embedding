import glob
import os
import torch
import segmentation_models_pytorch as smp

from visualizers.interactive import InteractiveVisualizer
from vectorizations.interactive import InteractiveVectorizationCosine


MODEL_PATH = r'C:\Repositories\pixel-wise-embedding\weights\pixel_wise_encoder_v2.pt'
DATA_PATH = r'C:\Repositories\pixel-wise-embedding\data\test_images\cars'
IMAGE_SIZE = (256, 256)
DEVICE = 'cpu'

if __name__ == '__main__':
    model = torch.load(MODEL_PATH, map_location=DEVICE)
    visualizer = InteractiveVisualizer(
        vectorization=InteractiveVectorizationCosine(
            model=model,
            image_size=IMAGE_SIZE,
            device=DEVICE,
        )
    )
    visualizer.vectorization.set_pos(IMAGE_SIZE[0]*0.5, IMAGE_SIZE[1]*0.5, 0)
    visualizer.show(glob.glob(os.path.join(DATA_PATH, '*')))
