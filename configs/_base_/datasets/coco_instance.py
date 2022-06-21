# dataset settings
dataset_type = 'CocoDataset'
data_root = '/data/motionscloud/training/carpart/data/'

classes = ['sli_side_turn_light', 'tyre', 'alloy_wheel', 'hli_head_light', 'hood',
    'fwi_windshield', 'flp_front_license_plate', 'door', 'mirror', 'handle',
    'qpa_quarter_panel', 'fender', 'grille', 'fbu_front_bumper', 'rocker_panel', 'rbu_rear_bumper',
    'pillar', 'roof', 'blp_back_license_plate', 'window', 'rwi_rear_windshield',
    'tail_gate', 'tli_tail_light', 'fbe_fog_light_bezel', 'fli_fog_light', 'fuel_tank_door',
    'lli_low_bumper_tail_light','exhaust']

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1,
    train=dict(
        classes=classes,
        type=dataset_type,
        ann_file=data_root + 'annotations_20220620/train.json',
        img_prefix=data_root + 'images/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotations_20220620/valid.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'annotannotations_20220620ations/valid.json',
        img_prefix=data_root + 'images/',
        pipeline=test_pipeline))
evaluation = dict(metric=['bbox', 'segm'])
