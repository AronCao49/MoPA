"""Build optimizers and schedulers"""
import warnings
import torch
from .lr_scheduler import ClipLR


def build_optimizer(optim_cfg, model, weight_decay=False, model_type="2D"):
    name = optim_cfg.TYPE
    if name == '':
        warnings.warn('No optimizer is built.')
        return None
    elif hasattr(torch.optim, name):
        param_base_layer = []
        name_base_layer = []
        param_new_layer = []
        name_new_layer = []
        # if cfg.MODEL_3D.TYPE == "SPVCNN" and cfg.MODEL_3D.SPVCNN.pretrained and model_type == "3D":
        #     for name_param, param in model.named_parameters():
        #         if "linear" in name_param:
        #             param_new_layer.append(param)
        #             name_new_layer.append(name_param)
        #         else:
        #             name_base_layer.append(name_param)
        #             param_base_layer.append(param)
        #     print(">>> Reduce lr_base for pretrained param: {}".format(str(name_base_layer[0:5]) + " ... " + str(name_base_layer[-5:])))
        #     return getattr(torch.optim, name)(
        #         [
        #             {'params':param_base_layer, 'lr':optim_cfg.BASE_LR / 10},
        #             {'params':param_new_layer}
        #         ],
        #         lr=optim_cfg.BASE_LR,
        #         weight_decay=optim_cfg.WEIGHT_DECAY,
        #         **optim_cfg.get(name, dict()),
        #     )
        # else:
        return getattr(torch.optim, name)(
            model.parameters(),
            lr=optim_cfg.BASE_LR,
            weight_decay=optim_cfg.WEIGHT_DECAY,
            **optim_cfg.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of optimizer.')


def build_scheduler(cfg, optimizer):
    name = cfg.SCHEDULER.TYPE
    if name == '':
        warnings.warn('No scheduler is built.')
        return None
    elif hasattr(torch.optim.lr_scheduler, name):
        scheduler = getattr(torch.optim.lr_scheduler, name)(
            optimizer,
            **cfg.SCHEDULER.get(name, dict()),
        )
    else:
        raise ValueError('Unsupported type of scheduler.')

    # clip learning rate
    if cfg.SCHEDULER.CLIP_LR > 0.0:
        print('Learning rate is clipped to {}'.format(cfg.SCHEDULER.CLIP_LR))
        scheduler = ClipLR(scheduler, min_lr=cfg.SCHEDULER.CLIP_LR)

    return scheduler
