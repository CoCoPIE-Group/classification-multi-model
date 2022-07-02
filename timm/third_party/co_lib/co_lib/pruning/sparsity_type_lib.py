import torch
import numpy as np
from numpy import linalg as LA
from skimage.util.shape import view_as_windows


def irregular(args, weight, percent):
    weight_abs = np.abs(weight)  # a buffer that holds weights with absolute values
    percentile = np.percentile(weight_abs, percent)  # get a value for this percentitle
    under_threshold = weight_abs < percentile
    above_threshold = weight_abs > percentile
    above_threshold = above_threshold.astype(np.float32)  # has to convert bool to float32 for numpy-tensor conversion
    # weight[under_threshold] = 0
    ww = weight * above_threshold
    return torch.from_numpy(above_threshold), torch.from_numpy(ww)


def block_punched(args, weight, percent):
    shape = weight.shape
    weight2d = weight.reshape(shape[0], -1)
    shape2d = weight2d.shape

    block_rows, block_cols = eval(args.sp_admm_block)

    # block_rows = 8  # this is the block size, it could be 16 or 8
    # block_cols = 4
    kernel_s1d = shape[2] * shape[3]
    length_x = kernel_s1d * block_cols  # kernel size = 3

    if shape2d[0] % block_rows != 0 or shape2d[1] % length_x != 0:
        print("the layer size is not divisible")
        # return torch.from_numpy(np.array([])).cuda(), torch.from_numpy(weight).cuda()
        raise SyntaxError("block_size error")

    cross_f = int(shape2d[0] / block_rows)
    cross_x = int(shape2d[1] / length_x)
    # **************************************************************************************************
    _block = (block_rows, length_x)
    partitioned_weight2d = view_as_windows(weight2d, _block, step=_block)
    npw = LA.norm(partitioned_weight2d, 2, axis=2)
    npw = npw.reshape(cross_f, cross_x, -1, kernel_s1d)
    npw = npw.sum(axis=-2)
    l2_norm_record = npw.reshape(cross_f, cross_x * kernel_s1d)

    percentile = np.percentile(l2_norm_record, percent)
    # under_threshold = l2_norm_record <= percentile
    above_threshold = l2_norm_record > percentile

    expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    for i in range(cross_f):
        for j in range(cross_x):
            # block = weight2d[i*block_rows : (i+1)*block_rows, j*length_x : (j+1)*length_x]
            for k in range(kernel_s1d):
                if above_threshold[i, kernel_s1d * j + k]:
                    for c in range(block_cols):
                        expand_above_threshold[i * block_rows:(i + 1) * block_rows, j * length_x + k + kernel_s1d * c] = 1

    weight = weight.reshape(shape)
    expand_above_threshold = expand_above_threshold.reshape(shape)
    weight = weight * expand_above_threshold
    # **************************************************************************************************
    # l2_norm_record = np.zeros((cross_f, cross_x * kernel_s1d))
    # for i in range(cross_f):
    #     for j in range(cross_x):
    #         block = weight2d[i * block_rows: (i + 1) * block_rows, j * length_x: (j + 1) * length_x]
    #         block_l2_norm = LA.norm(block, 2, axis=0)
    #         for k in range(kernel_s1d):
    #             for c in range(block_cols):
    #                 l2_norm_record[i, j * kernel_s1d + k] += block_l2_norm[k + c * kernel_s1d]  # there are 4 channels in every block
    # percentile = np.percentile(l2_norm_record, percent)
    # above_threshold = l2_norm_record > percentile
    # expand_above_threshold = np.zeros(shape2d, dtype=np.float32)
    # temp_mat_inexpand_0 = np.zeros(block_rows)
    # temp_mat_inexpand_1 = np.ones(block_rows)
    # for i in range(cross_f):
    #     for j in range(cross_x):
    #         # block = weight2d[i*block_rows : (i+1)*block_rows, j*length_x : (j+1)*length_x]
    #         for k in range(kernel_s1d):
    #             if above_threshold[i, kernel_s1d * j + k]:
    #                 for c in range(block_cols):
    #                     expand_above_threshold[i * block_rows: (i + 1) * block_rows,
    #                     j * length_x + k + kernel_s1d * c] = temp_mat_inexpand_1
    #             else:
    #                 for c in range(block_cols):
    #                     weight2d[i * block_rows: (i + 1) * block_rows, j * length_x + k + kernel_s1d * c] = temp_mat_inexpand_0
    # weight = weight.reshape(shape)
    # weight = weight * expand_above_threshold
    return torch.from_numpy(expand_above_threshold).cuda(), torch.from_numpy(weight).cuda()
