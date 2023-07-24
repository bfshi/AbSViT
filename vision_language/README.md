# Training and Evaluating AbSViT on Vision-Language Task

This code is modified based on [METER](https://github.com/zdou0830/METER)

## Environment Installation and Data Preparation

See the instructions in [METER](https://github.com/zdou0830/METER).

## Demo on Top-Down Attention

After installing the environment and preparing the VQAv2 data, `demo/visualize_attention.ipynb` shows an example of how top-down attention is adaptive to different tasks (questions).

## Training on VQAv2

Please first download the pre-trained ViT ([here](https://berkeley.box.com/shared/static/6fszey9291pvnkwdpt5ngrhh0rcu1iqu.pth)) or AbSViT([here](https://berkeley.box.com/shared/static/ejf7a2vnzg8pmwty0ih4temm2vgw14u5.pth)). 

To train ViT on VQAv2:

```
python run.py with data_root=<path_to_vqa_data> num_gpus=6 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=24 log_dir=<log_directory> vit_base_patch16_224 text_roberta image_size=224 clip_randaug vit_path=<pretrained_vit> learning_rate=1e-5
```

To train AbSViT on VQAv2:

```
python run.py with data_root=<path_to_vqa_data> num_gpus=6 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=24 log_dir=<log_directory> absvit_base_patch16_224 text_roberta image_size=224 clip_randaug vit_path=<pretrained_absvit> learning_rate=1e-5 lr_mult_feedback=3
```

## Evaluation on VQAv2

To evaluate ViT on VQAv2:

```
python run.py with data_root=<path_to_vqa_data> num_gpus=6 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=24 log_dir=<log_directory> vit_base_patch16_224 text_roberta image_size=224 load_path=<trained_checkpoint> test_only=True
```


To evaluate AbSViT on VQAv2:

```
python run.py with data_root=<path_to_vqa_data> num_gpus=6 num_nodes=1 task_finetune_vqa_clip_bert per_gpu_batchsize=24 log_dir=<log_directory> absvit_base_patch16_224 text_roberta image_size=224 load_path=<trained_checkpoint> test_only=True
```