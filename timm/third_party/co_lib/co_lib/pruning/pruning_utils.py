import numpy as np
from numpy import linalg as LA
import torch

from skimage.util.shape import view_as_windows

from co_lib.pruning.utils import *


def random_pruning(args, weight, prune_ratio):
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy

    if (args.sp_admm_sparsity_type == "filter"):
        shape = weight.shape
        weight2d = weight.reshape(shape[0], -1)
        shape2d = weight2d.shape
        indices = np.random.choice(shape2d[0], int(shape2d[0] * prune_ratio), replace=False)
        weight2d[indices, :] = 0
        weight = weight2d.reshape(shape)
        expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
        for i in range(shape2d[0]):
            expand_above_threshold[i, :] = i not in indices
        weight = weight2d.reshape(shape)
        expand_above_threshold = expand_above_threshold.reshape(shape)
        return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
    else:
        raise Exception("not implemented yet")


def L1_pruning(args, weight, prune_ratio):
    """
    projected gradient descent for comparison

    """
    percent = prune_ratio * 100
    weight = weight.cpu().detach().numpy()  # convert cpu tensor to numpy
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape
    row_l1_norm = LA.norm(weight2d, 1, axis=1)
    percentile = np.percentile(row_l1_norm, percent)
    under_threshold = row_l1_norm < percentile
    above_threshold = row_l1_norm > percentile
    weight2d[under_threshold, :] = 0
    above_threshold = above_threshold.astype(np.float32)
    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(shape2d[0]):
        expand_above_threshold[i, :] = above_threshold[i]
    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()


def update_prune_ratio(args, model, prune_ratios, global_sparsity, sp_prune_threshold=-1.0):
    # prune layers in prune_ratios only if the sparsity of this layer is < prune_sparsity_threshold
    if sp_prune_threshold > 0:
        for name, W in (model.named_parameters()):
            if (canonical_name(name) in prune_ratios.keys()) or (name in prune_ratios.keys()):
                sp_W = 1 - float(np.sum(W.detach().cpu().numpy() != 0)) / W.data.numel()
                print(name, sp_W)
                if sp_W > sp_prune_threshold:
                    prune_ratios.pop(name, None)
        #print(prune_ratios)
        #exit()

    total_size = 0
    for name, W in (model.named_parameters()):

        if (canonical_name(name) not in prune_ratios.keys()) and (name not in prune_ratios.keys()):
            continue
        total_size += W.data.numel()
    to_prune = np.zeros(total_size)
    index = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) and (name not in prune_ratios.keys()):
            continue
        size = W.data.numel()
        to_prune[index:(index + size)] = W.data.clone().cpu().view(-1).abs().numpy()
        index += size
    #sorted_to_prune = np.sort(to_prune)
    threshold = np.percentile(to_prune, global_sparsity * 100)

    # update prune_ratios key-value pairs
    total_zeros = 0
    for name, W in (model.named_parameters()):
        if (canonical_name(name) not in prune_ratios.keys()) and (name not in prune_ratios.keys()):
            continue
        size = W.data.numel()
        np_W_abs = W.detach().cpu().abs().numpy()

        zero_term = float(np.sum(np.logical_or(np_W_abs < threshold, np_W_abs == 0)))

        new_prune_ratio = zero_term / size

        total_zeros += zero_term

        prune_ratios[name] = min(new_prune_ratio, 0.99)

    print("Updated prune_ratios:")
    for key in prune_ratios:
        print("prune_ratios[{}]:{}".format(key, prune_ratios[key]))
    total_sparsity = total_zeros / total_size
    print("Total sparsity:{}".format(total_sparsity))

    return prune_ratios


def weight_growing(args, name, pruned_weight_np, lower_bound_value, upper_bound_value, update_init_method, mask_fixed_params=None):
    shape = None
    weight1d = None

    if mask_fixed_params is not None:
        mask_fixed_params = mask_fixed_params.detach().cpu().numpy()

    if upper_bound_value == 0:
        print("==> GROW: {}: to DENSE despite the sparsity type is \n".format(name))
        np_updated_mask = np.ones_like(pruned_weight_np, dtype=np.float32)
        updated_mask = torch.from_numpy(np_updated_mask).cuda()
        return updated_mask

    if upper_bound_value == lower_bound_value:
        print("==> GROW: {}: no grow, keep the mask and do finetune \n".format(name))
        non_zeros_updated = pruned_weight_np != 0
        non_zeros_updated = non_zeros_updated.astype(np.float32)
        np_updated_mask = non_zeros_updated
        updated_mask = torch.from_numpy(np_updated_mask).cuda()
        return updated_mask

    if (args.sp_admm_sparsity_type == "irregular"):
        # randomly select and set zero weights to non-zero to restore sparsity
        non_zeros_prune = pruned_weight_np != 0

        shape = pruned_weight_np.shape
        weight1d = pruned_weight_np.reshape(1, -1)[0]
        zeros_indices = np.where(weight1d == 0)[0]
        if args.sp_global_magnitude:
            num_added_zeros = int((lower_bound_value - upper_bound_value) * np.size(weight1d))
        else:
            num_added_zeros = int(np.size(zeros_indices) - upper_bound_value * np.size(weight1d))
        num_added_zeros = num_added_zeros if num_added_zeros < np.size(zeros_indices) else np.size(zeros_indices)
        num_added_zeros = num_added_zeros if num_added_zeros > 0 else 0
        target_sparsity = 1 - (np.count_nonzero(non_zeros_prune) + num_added_zeros) * 1.0 / np.size(pruned_weight_np)
        indices = np.random.choice(zeros_indices, num_added_zeros, replace=False)
        print("==> CALCULATE: all zeros: {}, need grow {} zeros, selected zeros: {} ".format(len(zeros_indices), num_added_zeros, len(indices)))

        # initialize selected weights
        if update_init_method == "weight":
            current_nozero = weight1d[np.nonzero(weight1d)]
            current_mean = np.mean(current_nozero)
            current_std = np.std(current_nozero)
            weight1d[indices] = np.random.normal(loc=current_mean, scale=current_std, size=np.size(indices))

            weight = weight1d.reshape(shape)

            print("==> double check sparsity after updating mask...")
            non_zeros_updated = weight != 0
            non_zeros_updated = non_zeros_updated.astype(np.float32)
            num_nonzeros_updated = np.count_nonzero(non_zeros_updated)
            sparsity_updated = 1 - (num_nonzeros_updated * 1.0) / total_num
            print(("{}: {}, {}, {}\n".format(name, str(num_nonzeros_updated), str(total_num), str(sparsity_updated))))

            # update mask
            # zero_mask = torch.from_numpy(non_zeros_updated).cuda()
            np_updated_zero_one_mask = non_zeros_updated

            # write updated weights back to model
            model.state_dict()[name].data.copy_(torch.from_numpy(weight))
        elif update_init_method == "zero":
            # set selected weights to -1 to get corrrect updated masks
            weight1d[indices] = -1
            weight = weight1d.reshape(shape)
            non_zeros_updated = weight != 0
            non_zeros_updated = non_zeros_updated.astype(np.float32)
            print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, target_sparsity))

            # update mask
            # zero_mask = torch.from_numpy(non_zeros_updated).cuda()
            np_updated_zero_one_mask = non_zeros_updated

            # assign 0 to -1 weight
            weight1d[indices] = 0
            weight = weight1d.reshape(shape)

            # write updated weights back to model
            # self.model.state_dict()[name].data.copy_(torch.from_numpy(weight))
        elif update_init_method == "kaiming":
            assert (False)

        np_updated_mask = np_updated_zero_one_mask
        updated_mask = torch.from_numpy(np_updated_mask).cuda()

        return updated_mask

    elif args.sp_admm_sparsity_type == "N:M-prune-pattern+block":
        shape = pruned_weight_np.shape
        weight2d = copy.copy(pruned_weight_np)
        if len(shape) == 2:
            # assume it is MN format
            pass
        elif len(shape) == 4:
            # assume it is CoCIKhKw format
            # first serialize KhKw to one dimension
            co, ci, kh, kw = weight2d.shape
            weight2d = weight2d.reshape([co, ci, kh * kw])
            # convert from CoCiKhKw to CoKhKwCi format
            weight2d = np.moveaxis(weight2d, 1, -1)
            # merge Ci, Kh, Kw dimension
            weight2d = weight2d.reshape([co, ci * kh * kw])
        elif len(shape) == 3:
            co, ci, kl = weight2d.shape
            weight2d = np.moveaxis(weight2d, 1, -1)
            weight2d = weight2d.reshape([co, ci * kl])
        else:
            assert False, "matrix dim = {}, not equal to 2 (MM), 3 (1d Conv), or 4 (2d Conv)!".format(len(shape))

        assert len(weight2d.shape) == 2, "Now only support 2d matrices"

        block = args.sp_admm_block
        block = eval(block)
        # print(block[0],block[1])
        # exit()
        row_pad_num = (block[0] - weight2d.shape[0] % block[0]) % block[0]
        col_pad_num = (block[1] - weight2d.shape[1] % block[1]) % block[1]
        new_weight2d = np.zeros((weight2d.shape[0] + row_pad_num, weight2d.shape[1] + col_pad_num))
        new_weight2d[:weight2d.shape[0], :weight2d.shape[1]] = weight2d
        new_weight2d = np.sqrt(new_weight2d * new_weight2d)
        '''
        np.set_printoptions(precision=2)
        np.set_printoptions(threshold=sys.maxsize)
        if args.local_rank==0:
            print(weight.shape)
            print(new_weight2d.shape)
            print(new_weight2d[:24,:24])
        '''

        if block[0] == -1:
            block_l = list(block)
            block_l[0] = new_weight2d.shape[0]
            block = tuple(block_l)
        elif block[1] == -1:
            block_l = list(block)
            block_l[1] = new_weight2d.shape[1]
            block = tuple(block_l)
        block_size = block[0] * block[1]
        partitioned_weight2d = view_as_windows(new_weight2d, block, step=block)
        sum2d = np.sum(partitioned_weight2d, axis=(2, 3))
        sum2d = (sum2d != 0).astype(np.float32)
        sum2d_shape = sum2d.shape
        sum1d = sum2d.reshape(1, -1)[0]
        zeros_indices = np.where(sum1d == 0)[0]
        num_added_zeros = int(np.size(zeros_indices) - upper_bound_value * np.size(sum1d))
        indices = np.random.choice(zeros_indices, num_added_zeros, replace=False)
        print("==> CALCULATE: all zeros: {}, need grow {} zeros, selected zeros: {} ".format(len(zeros_indices) * block_size, num_added_zeros * block_size, len(indices) * block_size))

        sum1d[indices] = 1
        sum2d = sum1d.reshape(sum2d_shape)
        sum2d = sum2d * 1.0
        growing_sparsity = np.sum(sum2d == 0) / np.size(sum2d)
        print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, growing_sparsity))

        mask2d = np.kron(sum2d, np.ones(block))
        mask2d = mask2d[:weight2d.shape[0], :weight2d.shape[1]]
        if len(shape) == 2:
            # assume it is MN format
            growing_mask = mask2d
        elif len(shape) == 4:
            # assume it is CoCIKhKw format
            co, ci, kh, kw = pruned_weight_np.shape
            # first separate out Ci, Kh, Kw dimensions
            mask2d = mask2d.reshape([co, kh, kw, ci])
            # convert from CoKhKwCi to CoCiKhKw format
            growing_mask = np.moveaxis(mask2d, -1, 1)
        elif len(shape) == 3:
            co, ci, kl = pruned_weight_np.shape
            mask2d = mask2d.reshape([co, kl, ci])
            growing_mask = np.moveaxis(mask2d, -1, 1)

        assert pruned_weight_np.shape == growing_mask.shape, "Mask shape not equal to weights shape!"

        growing_mask = torch.from_numpy(growing_mask).cuda()
        return growing_mask

    elif (args.sp_admm_sparsity_type == "4:2-H-V-balanced"):
        # data format transponse
        block_row_size = 4
        block_col_size = 4
        if (args.sp_admm_data_format == "NCHW" and len(pruned_weight_np.shape) == 4):
            pruned_weight_np = np.transpose(pruned_weight_np, (0, 3, 1, 2))  # NHWC to NCHW
        if (args.sp_admm_data_format == "NHWC" and len(pruned_weight_np.shape) == 4):
            pruned_weight_np = np.transpose(pruned_weight_np, (0, 2, 3, 1))  # NCHW to NHWC
        weight_abs = np.abs(pruned_weight_np)
        shape = pruned_weight_np.shape
        weight2d = pruned_weight_np.reshape(shape[0], -1)
        weight2d_abs = np.abs(weight2d)
        shape2d = weight2d.shape
        # args.sp_admm_pattern_col_sub * args.sp_admm_pattern_row_sub : select_number sparsity pattern
        pattern_col_num = shape2d[1] // block_col_size
        pattern_col_remainder = shape2d[1] % block_row_size
        pattern_row_num = shape2d[0] // block_col_size
        pattern_row_remainder = shape2d[0] % block_row_size
        weight2d_abs_pad = np.pad(weight2d_abs, ((0, 0 if pattern_row_remainder == 0 else block_row_size - pattern_row_remainder), (0, 0 if pattern_col_remainder == 0 else block_col_size - pattern_col_remainder)), 'constant', constant_values=0)
        weight2d_pad = np.pad(weight2d, ((0, 0 if pattern_row_remainder == 0 else block_row_size - pattern_row_remainder), (0, 0 if pattern_col_remainder == 0 else block_col_size - pattern_col_remainder)), 'constant', constant_values=0)
        shape2d_pad = weight2d_abs_pad.shape
        pattern_col_pad_num = shape2d_pad[1] // block_col_size
        pattern_row_pad_num = shape2d_pad[0] // block_row_size
        #print(weight2d_abs_pad[:10,:10])
        block_mask_rxc = np.random.rand(pattern_row_pad_num, pattern_col_pad_num) < (0.5 - upper_bound_value)  # with prob. threshold, the mask is 1 so we grow that block to dense
        block_mask_all = np.kron(block_mask_rxc, np.ones([block_row_size, block_col_size]))

        weight_mask = weight2d_abs_pad != 0
        growing_mask = (weight_mask + block_mask_all) > 0

        if (args.sp_admm_data_format == "NCHW" and len(pruned_weight_np.shape) == 4):
            pruned_weight_np = np.transpose(growing_mask, (0, 2, 3, 1))  # NCHW to NHWC
        if (args.sp_admm_data_format == "NHWC" and len(pruned_weight_np.shape) == 4):
            pruned_weight_np = np.transpose(growing_mask, (0, 3, 1, 2))  # NHWC to NCHW

        growing_sparsity = np.sum(growing_mask == 0) / np.size(growing_mask)
        print("==> GROW: {}: revise sparse mask to sparsity {}\n".format(name, growing_sparsity))

        growing_mask = torch.from_numpy(growing_mask).cuda()
        return growing_mask


def four_two_pruning(args, weight, pattern='hvb', percent=0.5):
    if args.sp_admm_sparsity_type == "4:2-2:1":
        print("using 4:2-2:1")
        candidate = [[
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ]]
        candidate_4_2 = np.array(candidate)
        candidate_4_2_flatten = candidate_4_2.reshape(36, 16)
    else:
        print("using 4:2-H-V-balanced")
        candidate = [[
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ], [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ], [
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ], [
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ], [
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
        ], [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ], [
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ], [
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ], [
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ], [
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
        ], [
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
        ], [
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], [
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
        ], [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ], [
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
        ], [
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ], [
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
        ], [
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
        ], [
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
        ], [
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
        ], [
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ], [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
        ], [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ], [
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ], [
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
        ], [
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
        ], [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
        ], [
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
        ], [
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ], [
            [0, 1, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ], [
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
        ], [
            [0, 1, 1, 0],
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
        ], [
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
        ], [
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
        ], [
            [0, 1, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
        ], [
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ], [
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 1, 0],
        ], [
            [0, 1, 0, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 0, 1, 0],
        ], [
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ], [
            [0, 1, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ], [
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
        ], [
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
            [0, 1, 0, 1],
        ], [
            [0, 0, 1, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
        ], [
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 1, 1, 0],
        ], [
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
        ], [
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 1, 0, 0],
            [1, 0, 0, 1],
        ], [
            [0, 0, 1, 1],
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 1, 0, 0],
        ], [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 1, 0, 0],
            [1, 0, 1, 0],
        ], [
            [0, 0, 1, 1],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [1, 1, 0, 0],
        ], [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]]
        candidate_4_2 = np.array(candidate)
        candidate_4_2_flatten = candidate_4_2.reshape(90, 16)

    # Assume pytorch use
    # OIHW to OHWI
    shape_before = weight.shape
    if len(weight.shape) == 4 and not args.sp_admm_do_not_permute_conv:
        weight = np.transpose(weight, (0, 2, 3, 1))
    #print("after reshape:", weight[:4,0,0,:4])

    weight_abs = np.abs(weight)
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    weight2d_abs = np.abs(weight2d)
    shape2d = weight2d.shape
    # args.sp_admm_pattern_col_sub * args.sp_admm_pattern_row_sub : select_number sparsity pattern
    pattern_col_num = shape2d[1] // 4
    pattern_col_remainder = shape2d[1] % 4
    pattern_row_num = shape2d[0] // 4
    pattern_row_remainder = shape2d[0] % 4

    weight2d_abs_pad = np.pad(weight2d_abs, ((0, 0 if pattern_row_remainder == 0 else 4 - pattern_row_remainder), (0, 0 if pattern_col_remainder == 0 else 4 - pattern_col_remainder)), 'constant', constant_values=0)
    weight2d_pad = np.pad(weight2d, ((0, 0 if pattern_row_remainder == 0 else 4 - pattern_row_remainder), (0, 0 if pattern_col_remainder == 0 else 4 - pattern_col_remainder)), 'constant', constant_values=0)
    shape2d_pad = weight2d_abs_pad.shape
    pattern_col_pad_num = shape2d_pad[1] // 4
    pattern_row_pad_num = shape2d_pad[0] // 4

    #print(weight2d_abs_pad[:10,:10])
    def check_valid(mat):
        assert mat.shape == (4, 4), 'Matrix not 4x4!'
        row_sum = np.sum(mat != 0, axis=0)
        col_sum = np.sum(mat != 0, axis=1)
        #print(mat, row_sum, col_sum)
        if row_sum[0] == 2 and row_sum[1] == 2 and row_sum[2] == 2 and row_sum[3] == 2 and col_sum[0] == 2 and col_sum[1] == 2 and col_sum[2] == 2 and col_sum[3] == 2:
            return True
        else:
            return False

    block = (4, 4)
    partitioned_weight2d = view_as_windows(weight2d_abs_pad, block, step=block)
    partitioned_weight2d_flatten = partitioned_weight2d.reshape(partitioned_weight2d.shape[0], partitioned_weight2d.shape[1], -1)
    candidate_sum = np.inner(candidate_4_2_flatten, partitioned_weight2d_flatten)
    max_idx_array = np.argmax(candidate_sum, axis=0)

    blocked_mask2d = partitioned_weight2d * 0

    #print(max_idx_array, max_idx_array.shape)
    final_mask = 0 * weight2d_abs_pad
    for i in range(pattern_row_pad_num):
        for j in range(pattern_col_pad_num):
            final_mask[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4] = candidate_4_2[max_idx_array[i][j]]
    weight2d_pad *= final_mask

    weight2d = weight2d_pad
    for i in range(4 - pattern_row_remainder):
        if pattern_row_remainder != 0: weight2d = np.delete(weight2d, shape2d_pad[0] - 1 - i, axis=0)
    for i in range(4 - pattern_col_remainder):
        if pattern_col_remainder != 0: weight2d = np.delete(weight2d, shape2d_pad[1] - 1 - i, axis=1)
    weight = weight2d.reshape(shape)
    # Assume pytorch use OIHW
    # OHWI bach to OIHW
    if len(weight.shape) == 4 and not args.sp_admm_do_not_permute_conv:
        weight = np.transpose(weight, (0, 3, 1, 2))
    shape_after = weight.shape

    non_zeros = weight != 0
    non_zeros = non_zeros.astype(np.float32)
    num_nonzeros = np.count_nonzero(non_zeros)
    total_num = non_zeros.size
    sparsity = 1 - (num_nonzeros * 1.0) / total_num
    print("num_nonzeros ", num_nonzeros, "total_num ", total_num, "sparsity", sparsity)

    return non_zeros, weight


def block_pruning(args, weight, percent, return_block_sums=False):
    print("using block pruning...")
    shape = weight.shape
    weight2d = copy.copy(weight)

    if len(shape) == 2:
        # assume it is MN format
        pass
    elif len(shape) == 4:
        # assume it is CoCIKhKw format
        # first serialize KhKw to one dimension
        co, ci, kh, kw = weight2d.shape
        weight2d = weight2d.reshape([co, ci, kh * kw])
        # convert from CoCiKhKw to CoKhKwCi format
        weight2d = np.moveaxis(weight2d, 1, -1)
        # merge Ci, Kh, Kw dimension
        weight2d = weight2d.reshape([co, ci * kh * kw])
    elif len(shape) == 3:
        co, ci, kl = weight2d.shape
        weight2d = np.moveaxis(weight2d, 1, -1)
        weight2d = weight2d.reshape([co, ci * kl])
    else:
        assert False, "matrix dim = {}, not equal to 2 (MM), 3 (1d Conv), or 4 (2d Conv)!".format(len(shape))

    assert len(weight2d.shape) == 2, "Now only support 2d matrices"

    block = args.sp_admm_block
    block = eval(block)
    row_pad_num = (block[0] - weight2d.shape[0] % block[0]) % block[0]
    col_pad_num = (block[1] - weight2d.shape[1] % block[1]) % block[1]
    new_weight2d = np.zeros((weight2d.shape[0] + row_pad_num, weight2d.shape[1] + col_pad_num))
    new_weight2d[:weight2d.shape[0], :weight2d.shape[1]] = weight2d
    new_weight2d = np.sqrt(new_weight2d * new_weight2d)

    if block[0] == -1:
        block_l = list(block)
        block_l[0] = new_weight2d.shape[0]
        block = tuple(block_l)
    elif block[1] == -1:
        block_l = list(block)
        block_l[1] = new_weight2d.shape[1]
        block = tuple(block_l)
    partitioned_weight2d = view_as_windows(new_weight2d, block, step=block)
    sum2d = np.sum(partitioned_weight2d, axis=(2, 3))
    percentile = np.percentile(sum2d, percent)
    above_threshold = (sum2d > percentile) + 0.0

    # output block index information for CSR
    if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
        mask = copy.copy(above_threshold)
        num_cmf = [0]
        col_indices = []
        for row in mask:
            #print(row)
            num_this_row = np.sum(row)
            col_idx = np.array(np.where(row > 0)).flatten()
            for c in col_idx:
                col_indices.append(c)

            num_cmf.append(num_this_row + num_cmf[-1])

        with open(args.sp_gs_output_ptr + "_" + name + ".txt", 'a') as f:
            f.write("{} {}\n".format(weight2d.shape[0], weight2d.shape[1]))
            f.write("{} {}\n".format(block[0], block[1]))
            for c in col_indices:
                f.write("{} ".format(int(c)))
            f.write("\n")
            for cmf in num_cmf:
                f.write("{} ".format(int(cmf)))

    mask2d = np.kron(above_threshold, np.ones(block))
    mask2d = mask2d[:weight2d.shape[0], :weight2d.shape[1]]

    if len(shape) == 2:
        # assume it is MN format
        pass
    elif len(shape) == 4:
        # assume it is CoCIKhKw format
        co, ci, kh, kw = weight.shape
        # first separate out Ci, Kh, Kw dimensions
        mask2d = mask2d.reshape([co, kh, kw, ci])
        # convert from CoKhKwCi to CoCiKhKw format
        mask2d = np.moveaxis(mask2d, -1, 1)
    elif len(shape) == 3:
        co, ci, kl = weight.shape
        mask2d = mask2d.reshape([co, kl, ci])
        mask2d = np.moveaxis(mask2d, -1, 1)

    assert weight.shape == mask2d.shape, "Mask shape not equal to weights shape!"
    masked_w = weight * mask2d

    if return_block_sums:
        assert len(shape) == 2, "return windowned block masks, now only support weight as 2D matrices!"
        return mask2d, masked_w, sum2d
    else:
        return mask2d, masked_w


def block_interleaved_4_2_pruning(args, weight, percent):
    mask2d, masked_w, sum2d = block_pruning(args, weight, percent, True)
    block_size = eval(args.sp_admm_block)
    percentile = np.percentile(sum2d, percent)
    above_threshold = (sum2d > percentile) + 0.0

    non_zero_block_per_row = np.sum(above_threshold, axis=1)  #number of element > threshold per row

    interleaved_block_multiplier = 16

    non_zero_block_per_row_aligned = (non_zero_block_per_row + interleaved_block_multiplier / 2) // interleaved_block_multiplier * interleaved_block_multiplier

    percentile_per_row = (1 - (non_zero_block_per_row_aligned + 0.0) / sum2d.shape[1]) * 100

    threshold_each_row = []
    for i in range(sum2d.shape[0]):
        threshold_each_row.append(np.percentile(sum2d[i], percentile_per_row[i]))
    threshold_each_row = np.array(threshold_each_row)

    threshold_all = np.repeat(np.expand_dims(threshold_each_row, axis=1), sum2d.shape[1], axis=1)

    above_threshold = (sum2d > threshold_all) + 0.0
    # back to weight mask
    mask2d = np.kron(above_threshold, np.ones(block_size))

    weight_blocked_pruned = weight * mask2d

    cnt = 0
    buffer_i = np.zeros(4)
    buffer_j = np.zeros(4)
    abs_ww = []
    for i in range(weight.shape[0]):
        for j in range(weight.shape[1]):
            if mask2d[i][j] == 1:
                abs_ww.append((abs(weight_blocked_pruned[i][j]), i, j))
                cnt += 1
            if cnt == 4:
                abs_ww.sort()
                weight_blocked_pruned[abs_ww[0][1]][abs_ww[0][2]] = 0
                weight_blocked_pruned[abs_ww[1][1]][abs_ww[1][2]] = 0
                cnt = 0
                abs_ww = []
    mask = (weight_blocked_pruned != 0) + 0.0
    return mask, weight_blocked_pruned


def list_duplicates(seq):
    tally = defaultdict(list)
    for i, item in enumerate(seq):
        tally[item].append(i)
    return ((key, locs) for key, locs in tally.items() if len(locs) > 1)


#    for (name, W) in model.named_parameters():
#        ADMM.ADMM_U[name] = torch.zeros(W.shape).cuda()

ttt = 0


def admm_multi_rho_scheduler(ADMM, name):
    """
    It works better to make rho monotonically increasing
    we increase it by 1.9x every admm epoch
    After 10 admm updates, the rho will be 0.91
    """

    ADMM.rhos[name] *= 2


def admm_adjust_learning_rate(optimizer, epoch, args):
    """ (The pytorch learning rate scheduler)
        Sets the learning rate to the initial LR decayed by 10 every args.sp_admm_update_epoch/3 epochs"""
    """
    For admm, the learning rate change is periodic.
    When epoch is dividable by admm_epoch, the learning rate is reset
    to the original one, and decay every 3 epoch (as the default
    admm epoch is 9)
    """
    admm_epoch = args.sp_admm_update_epoch
    lr = None

    if (epoch) % admm_epoch == 0:
        lr = args.sp_admm_lr
    else:
        admm_epoch_offset = (epoch) % admm_epoch

        admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

        lr = args.sp_admm_lr * (0.1**(admm_epoch_offset // admm_step))

    #print(admm_epoch, args.sp_admm_lr, (epoch) % admm_epoch, lr)
    #input('?')

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def threshold_mask(param, threshold):
    """Create a threshold mask for the provided parameter tensor using
    magnitude thresholding.
    Arguments:
        param: a parameter tensor which should be pruned.
        threshold: the pruning threshold.
    Returns:
        prune_mask: The pruning mask.
    """
    return torch.gt(torch.abs(param), threshold).type(param.type())


def zero_masking(args, config, model):
    masks = {}
    for name, W in model.named_parameters():  ## no gradient for weights that are already zero (for progressive pruning and sequential pruning)
        if name in config.prune_ratios:
            w_temp = W.cpu().detach().numpy()
            indices = (w_temp != 0)
            indices = indices.astype(np.float32)
            masks[name] = torch.from_numpy(indices).cuda()
    config.zero_masks = masks


'''
def masking(args, config, model):
    masks = {}
    for name, W in model.named_parameters():
        if name in config.prune_ratios:
            above_threshold, pruned_weight = weight_pruning(args, W, config.prune_ratios[name])
            W.data = pruned_weight
            masks[name] = above_threshold

    config.masks = masks
'''


def generate_mask(model):
    masks = {}
    for name, W in (model.named_parameters()):
        weight = W.cpu().detach().numpy()
        non_zeros = weight != 0
        non_zeros = non_zeros.astype(np.float32)
        zero_mask = torch.from_numpy(non_zeros).cuda()
        W = torch.from_numpy(weight).cuda()
        W.data = W
        masks[name] = zero_mask
    return masks


def update_subarray_bucket_mask(sum_array, mask, threshold, num_buckets, start_idx, end_idx, elem_per_row, args=None, name=None):

    def get_value(v):
        return v[2]

    def get_max_pack(sorted_bucket_values, num_rows, elem_per_row):
        one_pack = [[None for i in range(elem_per_row)] for j in range(num_rows)]
        col_indices = [0] * num_rows
        occupied_bucket = [None] * num_buckets
        num = 0
        while num < num_rows * elem_per_row:
            max_row = 0
            max_value = None
            max_idx = -1
            for j in range(num_rows):
                if col_indices[j] >= len(one_pack[j]):
                    continue
                for k in range(len(sorted_bucket_values[j]) - 1, -1, -1):
                    item = sorted_bucket_values[j][k]
                    col = item[1]
                    bucket = col % num_buckets
                    if occupied_bucket[bucket] == None:
                        if max_value is None or (max_value[2] < item[2]):
                            max_row = j
                            max_value = item
                            max_idx = k
                        break
            if max_value is None:
                break
            sorted_bucket_values[max_row].pop(max_idx)
            assert one_pack[max_row][col_indices[max_row]] == None
            one_pack[max_row][col_indices[max_row]] = max_value
            col_indices[max_row] = col_indices[max_row] + 1
            num = num + 1
            bucket = max_value[1] % num_buckets
            assert occupied_bucket[bucket] == None
            occupied_bucket[bucket] = True
        return one_pack

    assert (len(sum_array.shape) == 2)

    num_items = np.count_nonzero(np.abs(sum_array[start_idx:end_idx]) > threshold)  # np.sum(ref_mask[start_idx: end_idx])
    # whether to add one more elements or reduce one element?
    if num_items // num_buckets > 8:
        num_items = ((num_items + num_buckets // 2) // num_buckets) * num_buckets

    sorted_bucket_values = []
    num_rows = end_idx - start_idx

    assert num_rows * elem_per_row <= num_buckets
    sorted_bucket_values = [[]] * num_rows

    for i in range(num_rows):
        bucket_values = []
        for j in range(sum_array.shape[1]):
            bucket_values.append((start_idx + i, j, sum_array[start_idx + i][j]))
        sorted_bucket_values[i] = sorted(bucket_values, key=get_value)

    num_packs = 0
    while num_items > 0:
        one_pack = get_max_pack(sorted_bucket_values, num_rows, elem_per_row)
        #print(one_pack)
        if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
            one_line_v = []
            one_line_col_idx = []
            for one_row in one_pack:
                for z in one_row:
                    one_line_v.append(z[2])
                    one_line_col_idx.append(z[1])
            while len(one_line_v) < args.sp_admm_buckets_num:
                one_line_v.append(0.0)
                one_line_col_idx.append(sum_array.shape[1])
                print("Some bucket not filled!")

        num_packs += 1
        #print(one_line_v)
        #print(one_line_col_idx)
        #print(num_packs)

        if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
            with open(args.sp_gs_output_v + "_" + name + ".txt", 'a') as f:
                for v in one_line_v:
                    f.write("{} ".format(v))
                f.write("\n")
                for col in one_line_col_idx:
                    f.write("{} ".format(col))
                f.write("\n")

        #input("?")
        num_items = num_items - num_buckets
        for i in range(num_rows):
            for j in range(elem_per_row):
                elem = one_pack[i][j]
                if elem != None:
                    row = elem[0]
                    col = elem[1]
                    mask[row][col] = 1
                else:
                    # import pdb; pdb.set_trace()
                    print("Error, one number cannot be found after exhausting all values")
    #if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
    #    with open(args.sp_gs_output_ptr+"_"+name+".txt",'a') as f:
    #        f.write("{} ".format(num_packs))
    return num_packs


def update_tiling_mask(sum_array, mask, threshold, num_buckets, elem_per_row, args, name):

    def get_value(v):
        return v[2]

    num_rows = sum_array.shape[0]
    num_cols = sum_array.shape[1]
    tmp_mask = np.abs(sum_array) > threshold
    num_items = np.sum(tmp_mask)

    row_nums = np.sum(tmp_mask, 1)
    num_bucket_rows = num_buckets // elem_per_row
    # stable sort to find the rows to put together
    # currently just sort the number of entries from small to large
    dtype = [('row', int), ('num', int)]
    values = [(i, row_nums[i]) for i in range(num_rows)]
    rows = np.array(values, dtype=dtype)
    sorted_rows = np.sort(rows, kind="stable", order="num")

    new_sum_array = np.zeros(sum_array.shape)
    for i in range(num_rows):
        new_sum_array[i] = sum_array[sorted_rows[i]['row']]
    new_mask = np.zeros(sum_array.shape)
    start_row = 0
    while (start_row < num_rows):
        end_row = start_row + num_bucket_rows if start_row + num_bucket_rows < num_rows else num_rows
        update_subarray_bucket_mask(new_sum_array, new_mask, threshold, num_buckets, start_row, end_row, elem_per_row, args, name)
        start_row = end_row

    for i in range(num_rows):
        mask[sorted_rows[i]['row']] = new_mask[i]


def update_bucket_mask(sum_array, mask, threshold, num_buckets, elem_per_row, tile_str, args=None, name=None):
    assert len(sum_array.shape) == 2

    if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
        with open(args.sp_gs_output_ptr + "_" + name + ".txt", 'a') as f:
            f.write("{} {}\n".format(sum_array.shape[0], sum_array.shape[1]))
            f.write("{}\n".format(num_buckets))
            f.write("{}\n".format(elem_per_row))
        with open(args.sp_gs_output_ptr + "_" + name + "_cmf.txt", 'a') as f:
            f.write("{} {}\n".format(sum_array.shape[0], sum_array.shape[1]))
            f.write("{}\n".format(num_buckets))
            f.write("{}\n".format(elem_per_row))

    if tile_str:
        tile = eval(tile_str)
        tile_row = tile[0] if tile[0] > 0 else sum_array.shape[0]
        tile_col = tile[1] if tile[1] > 0 else sum_array.shape[1]
        for i in range(0, sum_array.shape[0], tile_row):
            for j in range(0, sum_array.shape[1], tile_col):
                srow = i
                erow = i + tile_row if i + tile_row < sum_array.shape[0] else sum_array.shape[0]
                scol = j
                ecol = j + tile_col if j + tile_col < sum_array.shape[1] else sum_array.shape[1]
                new_sum_array = sum_array[srow:erow, scol:ecol]
                new_mask = mask[srow:erow, scol:ecol]
                update_tiling_mask(new_sum_array, new_mask, threshold, num_buckets, elem_per_row, args, name)
                mask[srow:erow, scol:ecol] = new_mask
    else:
        start_idx = 0
        cmf_packs = 0
        cmf_list = []
        cdf_list = []
        while start_idx < sum_array.shape[0]:
            end_idx = start_idx + num_buckets // elem_per_row if start_idx + num_buckets // elem_per_row < sum_array.shape[0] else sum_array.shape[0]
            num_packs = update_subarray_bucket_mask(sum_array, mask, threshold, num_buckets, start_idx, end_idx, elem_per_row, args, name)
            cmf_packs += num_packs
            cmf_list.append(cmf_packs)
            cdf_list.append(num_packs)
            start_idx = end_idx

        if (args.sp_gs_output_v is not None) and (args.sp_gs_output_ptr is not None):
            with open(args.sp_gs_output_ptr + "_" + name + ".txt", 'a') as f:
                for i in cdf_list:
                    f.write("{} ".format(i))
            with open(args.sp_gs_output_ptr + "_" + name + "_cmf.txt", 'a') as f:
                for i in cmf_list:
                    f.write("{} ".format(i))
