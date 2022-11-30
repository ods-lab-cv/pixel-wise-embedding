import numpy as np
from tqdm import tqdm
from collections import defaultdict
import os
import torch
from torch.utils.data import DataLoader
import typing as tp
import segmentation_models_pytorch as smp
import albumentations as A
import glob

from train import TrainStep, ValStep
from datasets.ade20k_classes import ADE20K_CLASSES
from losses import PixelWiseLossWithMeanVector, PixelWiseLossWithVectorsConvFit
from out_collector import (
    OutCollectorWithMeanVector,
    OutCollectorWithLearningVectors,
    OutCollectorWithLearningVectorsWithConv,
    OutCollectorWithSingleConv,
)

from train import MODEL_KEY, COLLECTOR_KEY, LOSS_KEY, METRIC_KEY
from metrics import MulticlassAccuracy
from datasets.ade20k import get_ade20k
from datasets.cocostaff import get_cocostaff
from tester import Tester

from pycocotools.coco import COCO
import cv2
from torchvision.transforms import ToTensor

from config import Config

RESULT_PATH_MODEL = os.path.join(Config.result_path, 'models')
BEST_MODEL = os.path.join(RESULT_PATH_MODEL, 'best.pth')
LAST_MODEL = os.path.join(RESULT_PATH_MODEL, 'last.pth')

os.makedirs(Config.result_path, exist_ok=True)
os.makedirs(Config.result_tester, exist_ok=True)
os.makedirs(RESULT_PATH_MODEL, exist_ok=True)

if Config.previous_config_name is not None:
    config_filename = os.path.join(Config.result_path, Config.previous_config_name)
    with gzip.open(config_filename, 'rb') as f:
        Config = pickle.load(f)
else:
	config_filename = os.path.join(Config.result_path, 'config.pickle')
	with gzip.open(config_filename, 'wb') as f:
	    pickle.dump(Config, f)

train_dataset_ade20k, val_dataset_ade20k = Config.get_dataset()

# https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
# DataLoader wraps an iterable around the Dataset to enable easy access to the samples

train_loader_ade20k = DataLoader(train_dataset_ade20k, batch_size=Config.batch_size, shuffle=True, num_workers=6)
val_loader_ade20k = DataLoader(val_dataset_ade20k, batch_size=Config.batch_size, shuffle=False, num_workers=6)

model = Config.model
loss = Config.loss
out_collector = Config.out_collector

train_step = Config.train_step
val_step = Config.val_step

testers = Config.testers

model.to(Config.device)
loss.to(Config.device)
out_collector.to(Config.device)

model = model.train()

last_metric = None
for epoch in range(Config.epochs):

    train_logs_ade20k, train_logs_ade20k_per_cls = train_step.run(train_loader_ade20k, [])

    val_logs_ade20k, val_logs_ade20k_per_cls = val_step.run(val_loader_ade20k)

    cur_metric = val_logs_ade20k['loss']

    torch.save(model.state_dict(), LAST_MODEL)
    torch.save(loss.state_dict(), LAST_MODEL.replace('.pth', '_loss.pth'))
    torch.save(out_collector.state_dict(), LAST_MODEL.replace('.pth', '_collect.pth'))
    if last_metric is not None:
        if cur_metric < last_metric:
            torch.save(model.state_dict(), BEST_MODEL)
            torch.save(loss.state_dict(), BEST_MODEL.replace('.pth', '_loss.pth'))
            torch.save(out_collector.state_dict(), BEST_MODEL.replace('.pth', '_collect.pth'))
    else:
        last_metric = cur_metric
        
    for tester in testers:
        tester.test()
    
    
    print('train_logs_ade20k =', train_logs_ade20k)
    print('val_logs_ade20k =', val_logs_ade20k)
