_base_ = ['carpart_rear.py', '../_base_/swa.py']

data = dict(samples_per_gpu=3)
only_swa_training = True
swa_load_from = 'checkpoints/carpart_phase_2/epoch_24.pth'
swa_optimizer = dict(type='SGD', lr=0.0001, momentum=0.9, weight_decay=0.0001)
swa_optimizer_config = dict(
    _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))
work_dir = './work_dirs/swa_carpart_rear_close'


