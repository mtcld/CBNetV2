_base_ = ['scratch-copy-paste-HSV-LSJ-merimen-fully-match.py', '../_base_/swa.py']

data = dict(samples_per_gpu=3)
only_swa_training = True
swa_load_from = 'work_dirs/scratch-cp-HSV-LSJ-merimen-fully-match-filter-type/epoch_35.pth'
swa_optimizer = optimizer = dict(
    type='AdamW',
    lr=0.0001,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys=dict(
            absolute_pos_embed=dict(decay_mult=0.0),
            relative_position_bias_table=dict(decay_mult=0.0),
            norm=dict(decay_mult=0.0))))
# swa_optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
work_dir = './work_dirs/swa-scratch-cp-HSV-LSJ-merimen-fully-match-filter-type'


