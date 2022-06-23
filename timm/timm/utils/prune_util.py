import numpy as np

def print_sparsity(model=None, show_sparse_only=False, compressed_view=False):
    if show_sparse_only:
        print("The sparsity of all params (>0.01): num_nonzeros, total_num, sparsity")
        total_nz = 0
        total = 0
        for (name, W) in model.named_parameters():
            #print(name, W.shape)
            non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
            num_nonzeros = np.count_nonzero(non_zeros)
            total_num = non_zeros.size
            sparsity = 1 - (num_nonzeros * 1.0) / total_num
            if sparsity > 0.01:
                print("{}, {}, {}, {}, {}".format(name, non_zeros.shape, num_nonzeros, total_num, sparsity))
                total_nz += num_nonzeros
                total += total_num
        if total > 0:
            print("Overall sparsity for layers with sparsity >0.01: {}".format(1 - float(total_nz)/total))
        else:
            print("All layers are dense!")
        return

    if compressed_view is True:
        total_w_num = 0
        total_w_num_nz = 0
        for (name, W) in model.named_parameters():
            if "weight" in name:
                non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
                num_nonzeros = np.count_nonzero(non_zeros)
                total_w_num_nz += num_nonzeros
                total_num = non_zeros.size
                total_w_num += total_num

        sparsity = 1 - (total_w_num_nz * 1.0) / total_w_num
        print("The sparsity of all params with 'weights' in its name: num_nonzeros, total_num, sparsity")
        print("{}, {}, {}".format(total_w_num_nz, total_w_num, sparsity))
        return

    print("The sparsity of all parameters: name, num_nonzeros, total_num, shape, sparsity")
    for (name, W) in model.named_parameters():
        non_zeros = W.detach().cpu().numpy().astype(np.float32) != 0
        num_nonzeros = np.count_nonzero(non_zeros)
        total_num = non_zeros.size
        sparsity = 1 - (num_nonzeros * 1.0) / total_num
        print("{}: {}, {}, {}, [{}]".format(name, str(num_nonzeros), str(total_num), non_zeros.shape, str(sparsity)))
