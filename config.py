from dataclasses import dataclass
import typing as tp
from train import BaseStep, TrainStep, ValStep
from train import MODEL_KEY, COLLECTOR_KEY, LOSS_KEY, METRIC_KEY
from tester import Tester
from datasets.ade20k import get_ade20k
import albumentations as A
import torch
import os
import glob
import segmentation_models_pytorch as smp
from out_collector import OutCollectorWithSingleConv

# Temporary here, because I don't know where it should be imported from
class DiceLossWithPrepare(torch.nn.Module):
    def __init__(self):
        super(DiceLossWithPrepare, self).__init__()
        self.dice = smp.losses.DiceLoss('multiclass', from_logits=False)
    def forward(self, x, y):
        x, y = multiclass_out(x, y)
        return self.dice(x, y)

def multiclass_out(out, mask):
    out = torch.softmax(out, dim=1)
    mask = mask.argmax(dim=1)
    return out, mask
# Temporary here, because I don't know where it should be imported from

@dataclass
class BaseConfig:
    model: tp.Callable
    model_kwargs: tp.Dict

    loss = tp.Callable
    loss_kwargs = tp.Dict

    ade20k_dataset_path: str
    device: str


class Config:
    previous_config_name: str = None

    main_: str = '/content'
    exp_name: str = 'class_exp8'
    result_path: str = os.path.join('results', exp_name)
    result_tester: str = os.path.join(result_path, 'tester')
    result_config: str = os.path.join(result_path, 'config')

    features_size: int = 2
    n_classes: int = 3688
    device: str = 'cuda'
    amp: bool = True
    shape: tp.Tuple[int, int] = (32, 32)
    batch_size: int = 8
    epochs: int = 1
    backbone: str = 'efficientnet-b0'
    ignore_classes: tp.List[int] = [0]

    train_transforms: str = A.Compose(
        [
        A.Compose([
                A.OneOf([A.Resize(shape[0], shape[1]),], p=1)
            ], p=1), 
        A.Compose([
            A.Flip(),
            A.ShiftScaleRotate(),
            A.OneOf([
                A.HueSaturationValue(),
                A.ColorJitter(),
            ]),
            A.RandomBrightnessContrast(),
            A.RandomGamma(),
            A.MotionBlur(),
            ])
        ]
    )
    val_transforms: str = A.Resize(shape[0], shape[1])
    dataset_path: str = os.path.join(main_, 'ade20k/ADE20K_2021_17_01')

    # Extracting and preprocessing
    get_dataset: tp.Callable = (lambda : get_ade20k(
        train_transforms=Config.train_transforms,
        val_transforms=Config.val_transforms,
        dataset_path=Config.dataset_path,
    ))

    path_to_dataset_archive: str = os.path.join(main_, 'drive/MyDrive/ade20k.zip')
    path_to_extract_dataset_archive: str = os.path.join(main_, 'ade20k')

    loss: torch.nn.Module = DiceLossWithPrepare()

    model: torch.nn.Module = smp.Unet(
        backbone,
        classes=features_size,
        activation=None,
        decoder_channels=[features_size*6, features_size*5, features_size*4, features_size*3, features_size*2],
    )

    out_collector: torch.nn.Module = OutCollectorWithSingleConv(
        n_classes=n_classes,
        features_size=features_size,
        ignore_classes=ignore_classes,
    )

    train_step: BaseStep = TrainStep(
        model=model,
        out_collector=out_collector,
        loss=loss,
        trainable_objects=[MODEL_KEY, COLLECTOR_KEY],
        is_log_per_cls=False,
        device=device,
        amp=amp,
    )

    val_step: BaseStep = ValStep(
        model=model,
        out_collector=out_collector,
        loss=loss,
        is_log_per_cls=False,
        device=device,
        amp=amp,
    )

    testers: tp.List[Tester] = [
        Tester(
            model,
            images_paths=sorted(glob.glob('/content/pixel-wise-embedding/data/test_images/cars/*')),
            x=0.2, y=0.4,
            target_b=0,
            save_folder=os.path.join(result_tester, 'cars'),
            transforms=A.Resize(shape[0], shape[1]),
            threshold=0.9,
            gif_duration=500,
            run_every=1,
            device=device,
        ),

        Tester(
            model,
            images_paths=sorted(glob.glob('/content/pixel-wise-embedding/data/test_images/cats/*')),
            x=0.4, y=0.6,
            target_b=1,
            save_folder=os.path.join(result_tester, 'cats'),
            transforms=A.Resize(shape[0], shape[1]),
            threshold=0.9,
            gif_duration=500,
            run_every=1,
            device=device,
        ),

        Tester(
            model,
            images_paths=sorted(glob.glob('/content/pixel-wise-embedding/data/test_images/fish/*')),
            x=0.7, y=0.7,
            target_b=0,
            save_folder=os.path.join(result_tester, 'fish'),
            transforms=A.Resize(shape[0], shape[1]),
            threshold=0.9,
            gif_duration=500,
            run_every=1,
            device=device,
        ),
    ]
