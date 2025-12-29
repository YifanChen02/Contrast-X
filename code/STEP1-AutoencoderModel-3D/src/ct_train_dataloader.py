import os
from glob import glob
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, NormalizeIntensityd,
    RandCropByPosNegLabeld, Orientationd, ToTensord, ScaleIntensityD, CropForegroundd, ResizeWithPadOrCropd
)
from monai.data import PersistentDataset
import nibabel as nib

import numpy as np
from monai.transforms import MapTransform
import pandas as pd

from monai.transforms import RandSpatialCropd
from monai import transforms as trans

def get_dataframe(args, mode):
    dataset_df = pd.read_csv(args.dataset_csv)
    
    if isinstance(mode, list):
        df = dataset_df[dataset_df['Splits'].isin(mode)]
    else:
        df = dataset_df[dataset_df['Splits'] == mode]

    if args.DEBUG:
        df = df[:10]

    return df


def get_ct_dataloader(args,
                         mode='train', batch_size=2, spatial_size = (128, 128, 128), deterministic=False,
                         key_to_load = ["mask", "density"], cache_dir="./cache"):

    # Get datalist
    datalist = get_dataframe(args, mode)
    

    image_keys = args.input_modality  # "ct", "ctc"   # [k for k in key_to_load if k != "mask" and k != "patient_id"]
  
    print("cache_dir:", cache_dir)
    print("Read in List with Keys=", datalist.keys())
    print("Input Modality: ", image_keys, key_to_load)

    datalist = datalist.to_dict(orient='records')

    print(f"length of {mode} dataset_df: ", len(datalist))

    crop_transform = RandCropByPosNegLabeld(
        keys=key_to_load,
        label_key="mask",
        spatial_size=spatial_size,
        pos=1,
        neg=0,
        num_samples=1
    )

    

    crop_transform = RandSpatialCropd(
        keys=key_to_load,
        roi_size=spatial_size,   # crop size
        random_center=True,      # pick random center
        random_size=False        # fixed size
    )

    # Define transforms
    transforms = [
        
        LoadImaged(keys=key_to_load),
        EnsureChannelFirstd(keys=key_to_load),
        Orientationd(keys=key_to_load, axcodes="RAS"),

        # 
        trans.ScaleIntensityRanged(   # -1000 -> 1000
            keys=image_keys,
            a_min=-1000,
            a_max=1000,
            b_min=0.0,
            b_max=1.0,
            clip=True,
        ),

        

        crop_transform,
        ResizeWithPadOrCropd(
            keys=key_to_load,
            spatial_size=spatial_size,   # <- correct arg name
            allow_missing_keys=False,
            mode="constant",
            value=0  
        ),

        ScaleIntensityD(minv=0, maxv=1, keys=image_keys),  # Normalize all but mask

        
        ToTensord(keys=key_to_load),
    ]


    padding_mode = "reflection"
    # {'maximum', 'median', 'edge', 'symmetric', 'wrap', 'constant', 'linear_ramp', 'empty', 'minimum', 'reflect', 'mean'}.

    if mode=='train':
        transforms.extend( [
            trans.RandFlipD(prob=0.5, spatial_axis=0, keys=image_keys),
            trans.RandFlipD(prob=0.5, spatial_axis=1, keys=image_keys),
            # trans.RandFlipD(prob=0.5, spatial_axis=2, keys=image_keys),
            
            # trans.RandGaussianNoised(keys=image_keys, prob=0.2, mean=0.0, std=0.01),
            # trans.RandAdjustContrastd(keys=image_keys, prob=0.3, gamma=(0.7, 1.5)),
            

        ])

    transforms.extend( [trans.EnsureTypeD(keys=image_keys, dtype="float32")] )

    transforms = Compose(transforms)

    # Create dataset and dataloader
    dataset    = PersistentDataset(data=datalist, transform=transforms, cache_dir=cache_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=(mode=="train"),  num_workers=8)
    return dataloader