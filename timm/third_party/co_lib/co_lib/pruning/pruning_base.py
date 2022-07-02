import copy
import numpy as np
import torch
from dotmap import DotMap
from co_lib.common_base import CompressionBase
from co_lib.pruning import sparsity_type_lib
from co_lib.pruning.pruning_utils import *
from co_lib.pruning.default_pruning_config import DEFAULT_PRUNING_CONFIG
from co_lib.utils import deep_update_dict

try:
    from co_adv_lib import adv_sparsity_type
except:
    adv_sparsity_type = None


class PruneBase(CompressionBase):

    def __init__(self, args, model, optimizer, **kwargs):
        super(PruneBase, self).__init__(args=args, model=model, optimizer=optimizer)
        self.sparsity_type_lib = set(dir(sparsity_type_lib))
        self.adv_sparsity_type_lib = set()
        if adv_sparsity_type is not None:
            self.adv_sparsity_type_lib = set(dir(adv_sparsity_type))

        if hasattr(args, 'sp_admm_sparsity_type'):
            self.logger.info("please replace sp_admm_sparsity_type as sparsity_type, in the future will not support sp_admm_sparsity_type")
            args.sparsity_type = args.sp_admm_sparsity_type

        default_args = copy.deepcopy(DEFAULT_PRUNING_CONFIG)
        deep_update_dict(default_args,args)


        self.args = DotMap(default_args, _dynamic=False)


        self.model = model
        self.optimizer = optimizer

        self.configs, self.prune_ratios = load_configs(model, self.args.sp_config_file, self.logger, self.args)

        if self.args.sp_global_weight_sparsity:
            self.prune_ratios = update_prune_ratio(self.args, self.model, self.prune_ratios, self.args.sp_prune_ratios, self.args.sp_prune_threshold)


    def _get_sparsity(self, args, weight, percent, sparsity_type=None):
        if sparsity_type is None:
            sparsity_type = self.args.sparsity_type
        if sparsity_type in self.sparsity_type_lib:
            return eval(f'sparsity_type_lib.{sparsity_type}(args, weight, percent)')
        if sparsity_type in self.adv_sparsity_type:
            return eval(f'adv_sparsity_type.{sparsity_type}(args, weight, percent)')
        raise f"Error : {sparsity_type} not implements  "

    # def after_scheduler_step(self, *args, **kwargs):
    #     epock = kwargs.get('epoch')

    def harden_weights(self, option=None):
        if self.args.sp_no_harden:
            self.logger.info("Not hardening the matrix")
            return

        # if self.args.sp_global_weight_sparsity > 0:
        #     update_prune_ratio(self.args, self.model, self.prune_ratios, self.args.sp_global_weight_sparsity, self.args.sp_prune_threshold)

        for key in self.prune_ratios:
            print("prune_ratios[{}]:{}".format(key, self.prune_ratios[key]))

        # self.logger.info("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
        print("Hardened weight sparsity: name, num_nonzeros, total_num, sparsity")
        first = True
        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:  # ignore layers that do not have rho
                continue
            cuda_pruned_weights = None
            prune_ratio = self.prune_ratios[name]
            if option == None:
                cuda_pruned_weights = self.prune_weight(name, W, prune_ratio, first)  # get sparse model in cuda
                first = False

            elif option == "random":
                _, cuda_pruned_weights = random_pruning(self.args, W, prune_ratio)

            elif option == "l1":
                _, cuda_pruned_weights = L1_pruning(self.args, W, prune_ratio)
            else:
                raise Exception("not implmented yet")
            W.data = cuda_pruned_weights.cuda().type(W.dtype)  # replace the data field in variable

            if self.args.sparsity_type == "block":
                block = eval(self.args.sp_admm_block)
                if block[1] == -1:  # row pruning, need to delete corresponding bias
                    bias_layer = name.replace(".weight", ".bias")
                    with torch.no_grad():
                        bias = self.model.state_dict()[bias_layer]
                        bias_mask = torch.sum(W, 1)
                        bias_mask[bias_mask != 0] = 1
                        bias.mul_(bias_mask)
            elif self.args.sparsity_type == "filter" or self.args.sparsity_type == "filter_CSS":
                if not "downsample" in name:
                    bn_weight_name = name.replace("conv", "bn")
                    bn_bias_name = bn_weight_name.replace("weight", "bias")
                else:
                    bn_weight_name = name.replace("downsample.0", "downsample.1")
                    bn_bias_name = bn_weight_name.replace("weight", "bias")

                print("removing bn {}, {}".format(bn_weight_name, bn_bias_name))
                # bias_layer_name = name.replace(".weight", ".bias")

                with torch.no_grad():
                    bn_weight = self.model.state_dict()[bn_weight_name]
                    bn_bias = self.model.state_dict()[bn_bias_name]
                    # bias = self.model.state_dict()[bias_layer_name]

                    mask = torch.sum(torch.abs(W), (1, 2, 3))
                    mask[mask != 0] = 1
                    bn_weight.mul_(mask)
                    bn_bias.mul_(mask)
            # bias.data.mul_(mask)

            non_zeros = W.detach().cpu().numpy() != 0
            non_zeros = non_zeros.astype(np.float32)
            num_nonzeros = np.count_nonzero(non_zeros)
            total_num = non_zeros.size
            sparsity = 1 - (num_nonzeros * 1.0) / total_num
            print("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))

    # self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_

    def prune_weight(self, name, weight, prune_ratio, first):
        if prune_ratio == 0.0:
            return weight
        # if pruning too many items, just prune everything
        if prune_ratio >= 0.999:
            return weight * 0.0
        if self.args.sparsity_type == "irregular_global":
            res = self.weight_pruning_irregular_global(weight, prune_ratio, first)
        else:
            if (self.args.sp_gs_output_v is not None) and (self.args.sp_gs_output_ptr is not None):
                print("Start to output layer {}".format(name))

            sparsity_type_copy = copy.copy(self.args.sparsity_type)
            sparsity_type_list = (self.args.sparsity_type).split("+")
            if len(sparsity_type_list) != 1:  # multiple sparsity type
                print(sparsity_type_list)
                for i in range(len(sparsity_type_list)):
                    sparsity_type = sparsity_type_list[i]
                    print("* sparsity type {} is {}".format(i, sparsity_type))
                    self.args.sparsity_type = sparsity_type
                    _, weight = self.weight_pruning(name, weight, prune_ratio)
                    self.args.sparsity_type = sparsity_type_copy
                    print(np.sum(weight.detach().cpu().numpy() != 0))
                return weight.to(weight.device).type(weight.dtype)
            else:
                _, res = self.weight_pruning(name, weight, prune_ratio)

        return res.to(weight.device).type(weight.dtype)

    def weight_pruning(self, name, w, prune_ratio, mask_fixed_params=None):
        args = self.args
        configs = self.configs
        torch_weight = w
        weight = w.detach().clone().cpu().numpy()  # convert cpu tensor to numpy
        if mask_fixed_params is not None:
            mask_fixed_params = mask_fixed_params.detach().cpu().numpy()

        percent = prune_ratio * 100

        return self._get_sparsity(args, weight, percent)

    def weight_pruning_irregular_global(self, weight, prune_ratio, first):
        with torch.no_grad():
            if first:
                self.irregular_global_blob = None
                total_size = 0
                for i, (name, W) in enumerate(self.model.named_parameters()):
                    if name not in self.prune_ratios:
                        continue
                    if self.prune_ratios[name] == 0.0:
                        continue
                    total_size += W.numel()
                to_prune = torch.zeros(total_size)
                index_ = 0
                for (name, W) in self.model.named_parameters():
                    if name not in self.prune_ratios:
                        continue
                    if self.prune_ratios[name] == 0.0:
                        continue
                    size = W.numel()
                    to_prune[index_:(index_ + size)] = W.view(-1).abs().clone()
                    index_ += size
                sorted_to_prune, _ = torch.sort(to_prune)
                self.irregular_global_blob = sorted_to_prune

            total_size = self.irregular_global_blob.numel()
            thre_index = int(total_size * prune_ratio)
            global_th = self.irregular_global_blob[thre_index]
            above_threshold = (weight.detach().cpu().float().abs() > global_th).to(weight.device).type(weight.dtype)
            weight = (weight * above_threshold).type(weight.dtype)
            return weight
