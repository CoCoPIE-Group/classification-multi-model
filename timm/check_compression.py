import numpy as np
import torch
from timm.utils import torch_utils


def test_sparsity(model):

    # --------------------- total sparsity --------------------
    total_zeros = 0
    total_nonzeros = 0

    for name, weight in model.named_parameters():
        if (len(weight.size()) == 4):# and "shortcut" not in name):  # only consider conv layers
            zeros = np.sum(weight.cpu().detach().numpy() == 0)
            total_zeros += zeros
            non_zeros = np.sum(weight.cpu().detach().numpy() != 0)
            total_nonzeros += non_zeros

    comp_ratio = float((total_zeros + total_nonzeros)) / float(total_nonzeros)

    print("ONLY consider CONV layers: ")
    print("total number of zeros: {}, non-zeros: {}, zero sparsity is: {:.4f}".format(
        total_zeros, total_nonzeros, total_zeros / (total_zeros + total_nonzeros)))
    print("only consider conv layers, compression rate is: {:.4f}".format(
        (total_zeros + total_nonzeros) / total_nonzeros))
    print("===========================================================================\n\n")
    return comp_ratio




if __name__ == '__main__':
    import timm
    from timm.models import create_model, apply_test_time_pool, load_checkpoint

    model = timm.create_model('efficientnet_b0', pretrained=True)
    print(model)
    n_po, macso = torch_utils.model_info(model, verbose=False)

    # model = timm.create_model('efficientnet_b1_pruned', pretrained=True)
    # model = timm.create_model('efficientnet_b0', pretrained=True)
    model = create_model(
        "efficientnet_b0_c07",
        pretrained=False,
        num_classes=None,
        in_chans=3,
        global_pool=None,
        scriptable=False)

    print("Check 8x prunned model: ")
    # state_dict = torch.load('./output/train/effnet_b0_u0.6/model_best.pth.tar')  # nonuniform
    # load_checkpoint(model, './output/checkpoint-161.pth.tar', use_ema=False)
    n_p8x, macs8x = torch_utils.prunned_model_info(model, input_size=224)
    print("parameters compression rate: %g, flops compression rate: %g" % (n_po / n_p8x, macso / macs8x))

    test_sparsity(model)