# Training and Evaluating AbSViT on Semantic Segmentation

## Step 1. Installing MMSegmentation

The instructions are detailed in the [official doc](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation). 

Specifically, run the following command:

```
pip install -U openmim
mim install mmcv-full
git clone https://github.com/open-mmlab/mmsegmentation.git
cd mmsegmentation
pip install -v -e .
```

## Step 2. Preparing the dataset

Please follow the [offcial doc](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/dataset_prepare.md#prepare-datasets) to prepare ADE20K, Cityscapes, PASCAL VOC, or any dataset you would like to try.


## Step 3. Add AbSViT to MMSegmentation

Copy `models/vit_bottom_up.py` and `models/vit_top_down.py` into `mmsegmentation/mmseg/models/backbones/`.

```
cp models/vit_bottom_up.py mmsegmentation/mmseg/models/backbones/
cp models/vit_top_down.py mmsegmentation/mmseg/models/backbones/
```

Import ViT and AbSViT by adding the following code into `mmsegmentation/mmseg/models/backbones/__init__.py`.

```
from .vit_bottom_up import ViT_Bottom_Up
from .vit_top_down import ViT_Top_Down
```

Also, add `"ViT_Bottom_Up"` and `"ViT_Top_Down"` in the `__all__` list in the same file.

Then, copy `configs/vit_bottom_up/` and `configs/vit_top_down/` into `mmsegmentation/mmseg/configs`.

```
cp -r configs/vit_bottom_up/ mmsegmentation/mmseg/configs
cp -r configs/vit_top_down/ mmsegmentation/mmseg/configs
```


## Step 4. Train and Evaluate

For example, to finetune AbSViT-B on ADE20K for 160k iterations, go into the mmsegmentation directory and run
```
bash tools/dist_train.sh configs/vit_top_down/upernet_vit-top-down-b16_end2end_512x512_160k_ade20k.py 8 --work-dir work_dirs/ --deterministic --options model.backbone.init_cfg.checkpoint=path_to_pretrained_model data.samples_per_gpu=2
```

To learn about the configs, see the [official doc](https://mmsegmentation.readthedocs.io/en/latest/tutorials/config.html).