norm_cfg = dict(type='SyncBN', requires_grad=True)
checkpoint_file = 'path_to_pretrained_model'
model = dict(
    type='EncoderDecoder',
    backbone=dict(
        type='ViT_Top_Down',
        img_size=512,
        prompt_config=dict(deep=True, num_tokens=50, dropout=0.0),
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4,
        qkv_bias=True,
        num_classes=-1,
        drop_path_rate=0.1,
        transfer_type='linear',
        out_indices=(2, 5, 8, 11),
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file,)
    ),
    neck=None,
    decode_head=dict(
        type='UPerHead',
        in_channels=[768, 768, 768, 768],
        in_index=[0, 1, 2, 3],
        pool_scales=(1, 2, 3, 6),
        channels=512,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0)),
    auxiliary_head=dict(
        type='FCNHead',
        in_channels=768,
        in_index=3,
        channels=256,
        num_convs=1,
        concat_input=False,
        dropout_ratio=0.1,
        num_classes=150,
        norm_cfg=norm_cfg,
        align_corners=False,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4)),
    # model training and testing settings
    train_cfg=dict(),
    test_cfg=dict(mode='whole')  # yapf: disable
)

