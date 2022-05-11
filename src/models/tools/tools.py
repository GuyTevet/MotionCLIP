import torch.nn as nn
from torch.nn.modules.module import ModuleAttributeError


class AutoParams(nn.Module):
    def __init__(self, **kargs):
        try:
            for param in self.needed_params:
                if param in kargs:
                    setattr(self, param, kargs[param])
                else:
                    raise ValueError(f"{param} is needed.")
        except ModuleAttributeError:
            pass
            
        try:
            for param, default in self.optional_params.items():
                if param in kargs and kargs[param] is not None:
                    setattr(self, param, kargs[param])
                else:
                    setattr(self, param, default)
        except ModuleAttributeError:
            pass
        super().__init__()


# taken from joeynmt repo
def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False
