import json
import yaml
import torch
import torch.cuda.amp as amp
from torch.nn.utils import remove_weight_norm
from torch.nn.utils.weight_norm import WeightNorm
from torch.nn.parameter import Parameter, UninitializedParameter
from torch import norm_except_dim


def disable_amp():

    class AutocastOurs(amp.autocast):
        _enabled = False

        def __init__(self):
            self._enabled = False

    class GradScalerOurs(amp.GradScaler):
        _enabled = False

        def __init__(self):
            self._enabled = False

    amp.autocast = AutocastOurs

    amp.GradScaler = GradScalerOurs


def is_leaf_module(module):
    module_list = list(module.modules())

    return bool(len(module_list) == 1)


def rm_weights_norm(model):
    rm_layer = []
    for module in model.modules():
        if is_leaf_module(module):
            try:
                remove_weight_norm(module)
                rm_layer.append(module)
            except:
                pass
    return rm_layer


def rm_ourwn(model):
    rm_layer = []
    for module in model.modules():
        if is_leaf_module(module):
            try:
                _remove_ourwn(module)
                rm_layer.append(module)
            except:
                pass
    return rm_layer


class OurWN(WeightNorm):
    """
	will keep weights as parameters instead of attribute
	"""

    @staticmethod
    def apply(module, name: str, dim: int) -> 'WeightNorm':
        for k, hook in module._forward_pre_hooks.items():
            if isinstance(hook, OurWN) and hook.name == name:
                raise RuntimeError("Cannot register two weight_norm hooks on "
                                   "the same parameter {}".format(name))

        if dim is None:
            dim = -1

        fn = OurWN(name, dim)

        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError('The module passed to `WeightNorm` can\'t have uninitialized parameters. '
                             'Make sure to run the dummy forward before applying weight normalization')
        # remove w from parameter list

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(name + '_g', Parameter(norm_except_dim(weight, 2, dim).data))
        module.register_parameter(name + '_v', Parameter(weight.data))
        module._parameters[name] = torch.nn.Parameter(fn.compute_weight(module).data)

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def __call__(self, module, inputs) -> None:
        module._parameters[self.name] = torch.nn.Parameter(self.compute_weight(module).data)


def appy_ourwn(module, name: str = 'weight', dim: int = 0):
    OurWN.apply(module, name, dim)
    return module


def _remove_ourwn(module, name: str = 'weight'):
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, OurWN) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module
    raise ValueError("weight_norm of '{}' not found in {}".format(name, module))


def get_device(model):
    """
    Function to find which device is model on for pytorch
    """
    return next(model.parameters()).device


def get_input_shape_batch_size_for_data_loader(data_loader):
    """
    Gets input shape of image and batch size from data loader
    :param data_loader: Iterates over data set
    :return: returns batch size and shape of one image
    """
    for _, (images_in_one_batch, _) in enumerate(data_loader):
        # finding shape of a batch
        input_shape = torch.Tensor.size(images_in_one_batch)

        return input_shape[0], (1, input_shape[1], input_shape[2], input_shape[3])

def deep_update_dict(dict_a, dict_b):
    '''
    Deep update dict_a by dict_b
    :param dict_a:
    :param dict_b:
    :return:
    '''
    for k, v in dict_b.items():
        if k in dict_a:
            if isinstance(v, dict):
                deep_update_dict(dict_a[k], v)
            else:
                dict_a[k] = v
        else:
            dict_a[k] = v

def export_prune_sp_config_file(cl,save_path):
    config = cl.args.prune.sp_config_file
    if isinstance(config,str) and len(config)>0:
        with open(save_path, 'w') as outfile:
            data = json.loads(config)
            yaml.dump(data, outfile, default_flow_style=False)
