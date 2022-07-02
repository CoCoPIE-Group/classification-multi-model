import copy
import numpy as np
import torch

from ..utils import get_device

try:
    import libpymo
    from aimet_torch.quantsim import QuantizationSimModel
    from aimet_torch.onnx_utils import OnnxExportApiArgs
    from aimet_torch.batch_norm_fold import fold_all_batch_norms
    from aimet_torch.cross_layer_equalization import equalize_bn_folded_model



    def aimet_export_onnx(model, path, filename_prefix, dummy_input, onnx_export_args=OnnxExportApiArgs(), propagate_encodings=False):
        """
        This method exports out the quant-sim model so it is ready to be run on-target.

        Specifically, the following are saved

        1. The sim-model is exported to a regular PyTorch model without any simulation ops
        2. The quantization encodings are exported to a separate JSON-formatted file that can then be imported by the on-target runtime (if desired)
        3. Optionally, An equivalent model in ONNX format is exported. In addition, nodes in the ONNX model are named the same as the corresponding PyTorch module names. This helps with matching ONNX node to their quant encoding from #2.

        :param path: path where to store model pth and encodings
        :param filename_prefix: Prefix to use for filenames of the model pth and encodings files
        :param dummy_input: Dummy input to the model. Used to parse model graph. It is required for the dummy_input to be placed on CPU.
        :param onnx_export_args: optional export argument with onnx specific overrides if not provide export via torchscript graph
        :param propagate_encodings: If True, encoding entries for intermediate ops (when one PyTorch ops results in multiple ONNX nodes) are filled with the same BW and data_type as the output tensor for that series of ops.
        :return: None

        """
        model.eval()
        model_to_export = copy.deepcopy(model).cpu()
        dummy_input = dummy_input.to('cpu')
        all_modules_in_model_to_export = [module for module in model_to_export.modules()]
        QuantizationSimModel._remove_quantization_wrappers(model_to_export, all_modules_in_model_to_export)

        QuantizationSimModel.export_onnx_model_and_encodings(path, filename_prefix, model_to_export, model, dummy_input, onnx_export_args, propagate_encodings)

        model.train()


    def fold_layers(model, input_shapes):
        """
        model should in cpu!
        :param model:
        :param input_shapes:
        :return:
        """
        assert 'cpu' in get_device(model).type, "fold_layers required model on cpu"
        folded_pairs = fold_all_batch_norms(model, input_shapes)
        return folded_pairs


    def cross_layer_equalization(model, input_shapes, folded_pairs):
        assert 'cpu' in get_device(model).type, "cross_layer_equalization required model on cpu"
        equalize_bn_folded_model(model, input_shapes, folded_pairs)
except ImportError:
    Warning('Aimet installation was incomplete. Co_lib may not use quantized function')