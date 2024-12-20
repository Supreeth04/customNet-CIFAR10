import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def get_transforms(mean, std):
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=20, p=0.8),
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.CoarseDropout(
            max_holes=2, max_height=8, max_width=8,
            min_holes=2, min_height=8, min_width=8,
            fill_value=mean, p=0.3
        ),
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])

    test_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
    ])
    
    return train_transform, test_transform 