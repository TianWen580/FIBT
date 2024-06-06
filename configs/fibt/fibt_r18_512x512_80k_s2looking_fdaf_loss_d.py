_base_ = [
    '../_base_/models/fibt_r18.py',
    '../common/standard_512x512_40k_s2looking.py']

crop_size = (512, 512)
model = dict(
    backbone=dict(
        interaction_cfg=(
            None,
            None,
            dict(type='MixExchange', p=1/2),
            dict(type='ChannelExchange', p=1/2),
        )
    ),
    decode_head=dict(
        interaction_cfg=(
            dict(type='ChannelAttnExchange', threshold=0.5, inchannel=512, reduction=16),
            dict(type='TwoBranchesCR', inchannel=512, reduction=16),
            dict(type='TwoBranchesCR', inchannel=256, reduction=16),
            dict(type='TwoBranchesCR', inchannel=128, reduction=16) 
        ),
        is_fdaf=True,
        num_classes=2,
        # sampler=dict(type='mmseg.OHEMPixelSampler', thresh=0.7, min_kept=100000),
        ignore_index=None,
        loss_decode=dict(
            type='mmseg.DiceLoss', use_sigmoid=False, loss_weight=1.0),
    ),
    # test_cfg=dict(mode='slide', crop_size=crop_size, stride=(crop_size[0]//2, crop_size[1]//2)),
)

train_pipeline = [
    dict(type='MultiImgLoadImageFromFile'),
    dict(type='MultiImgLoadAnnotations'),
    dict(type='MultiImgRandomRotFlip', rotate_prob=0.5, flip_prob=0.5, degree=(-20, 20)),
    dict(type='MultiImgRandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='MultiImgExchangeTime', prob=0.5),
    dict(
        type='MultiImgPhotoMetricDistortion',
        brightness_delta=10,
        contrast_range=(0.8, 1.2),
        saturation_range=(0.8, 1.2),
        hue_delta=10),
    dict(type='MultiImgPackSegInputs')
]

train_dataloader = dict(
    dataset=dict(pipeline=train_pipeline))

# optimizer
optimizer=dict(
    type='AdamW', lr=0.001, betas=(0.9, 0.999), weight_decay=0.05)
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=optimizer)

# compile = True # use PyTorch 2.x