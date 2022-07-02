import copy
import numpy as np
import torch


from dotmap import DotMap

from co_lib.quantization import QuantizationBase
from co_lib.quantization.register_quantization import register_quantization
from ..utils import disable_amp, rm_weights_norm, appy_ourwn, get_input_shape_batch_size_for_data_loader, get_device

try:
    import libpymo
    from .aimet_utils import fold_layers, cross_layer_equalization
    from aimet_torch import quantsim as qsim
    from aimet_torch.quantsim import QcQuantizeWrapper
    from aimet_torch import utils
    from aimet_torch.bias_correction import forward_pass, find_all_conv_bn_with_activation, call_analytical_mo_correct_bias, get_output_data, call_empirical_mo_correct_bias
    from aimet_torch.quantsim import QuantParams


    @register_quantization(1)
    class AimetQt(QuantizationBase):

        __method_name__ = "qt_aimet"

        default_args = {'qat': False, 'fold_layers': True, 'cross_layer_equalization': False, 'bias_correction': True, 'rounding_mode': 'nearest', 'num_quant_samples': 1000, 'num_bias_correct_samples': 1000, 'weight_bw': 8, 'act_bw': 8, 'quant_scheme': 'tf_enhanced', 'layers_to_ignore': [], 'auto_add_bias': False, 'perform_only_empirical_bias_corr': True}

        def __init__(self, *args, **kwargs):
            super(AimetQt, self).__init__(*args, **kwargs)
            self.args = kwargs.get('args', None)
            self.model = kwargs.get('model', None)
            self.data_loader = kwargs.get('data_loader', None)
            self.default_args.update(self.args)
            self.args = DotMap(self.default_args, _dynamic=False)

        @staticmethod
        def argument_parser(parser):
            qt_args = parser.add_argument_group('Aimet group')
            qt_args.add_argument('--qat', type=bool, default=False, help="enable qat")
            qt_args.add_argument('--fold_layers', type=bool, default=True, help="update admm after how many minibatches")
            qt_args.add_argument('--cross_layer_equalization', type=float, default=0.001, help="apply cross_layer_equalization")
            qt_args.add_argument('--bias_correction', type=str, default='gather_scatter', help="apply bias_correction")

        def init(self, *args, **kwargs):
            model = self.model
            data_loader = self.data_loader
            num_quant_samples = self.args['num_quant_samples']
            num_bias_correct_samples = self.args['num_bias_correct_samples']
            weight_bw = self.args['weight_bw']
            act_bw = self.args['act_bw']
            quant_scheme = self.args['quant_scheme']
            rounding_mode = self.args['rounding_mode']
            layers_to_ignore = self.args['layers_to_ignore']
            logger = self.logger
            perform_only_empirical_bias_corr = self.args['perform_only_empirical_bias_corr']

            # set model to eval
            model.eval()

            # Find batch size and shape of input tensor
            batch_size, input_shape = get_input_shape_batch_size_for_data_loader(data_loader)

            #disable amp
            disable_amp()

            #remove weights norm
            rm_layer = rm_weights_norm(self.model)

            if self.args['cross_layer_equalization'] or self.args['fold_layers']:
                device = get_device(model)
                model.cpu()
                folded_pairs = fold_layers(model, input_shape)
                model.to(device=device)

                if self.args['cross_layer_equalization']:
                    device = get_device(model)
                    model.cpu()
                    cross_layer_equalization(model, input_shape, folded_pairs)
                    model.to(device=device)

            if self.args['qat']:
                #apply our weights norm
                for module in rm_layer:
                    appy_ourwn(module)

            # set aimet params
            quant_params = QuantParams(weight_bw=weight_bw, act_bw=act_bw, round_mode=rounding_mode, quant_scheme=quant_scheme)

            # Rounding up number of samples to batch size
            n_batches_bias_correction = int(np.ceil(num_bias_correct_samples / batch_size))
            n_batches_quantization = int(np.ceil(num_quant_samples / batch_size))

            data_loader_n_samples_bias_corr = utils.IterFirstX(data_loader, n_batches_bias_correction)
            data_loader_n_samples_quant = utils.IterFirstX(data_loader, n_batches_quantization)

            def pass_data_through_model(model, early_stopping_iterations=None, use_cuda=False):
                # pylint: disable=unused-argument
                # forward pass for given number of batches for model
                for (images_in_one_batch, _) in data_loader_n_samples_quant:
                    forward_pass(model, images_in_one_batch)

            ordered_conv_linear_nodes = utils.get_ordered_lists_of_conv_fc(model, input_shape)

            conv_bn_dict = find_all_conv_bn_with_activation(model, input_shape)

            # Create a copy of the model as reference model
            model_copy = copy.deepcopy(model)

            if self.args['auto_add_bias'] or self.args['bias_correction']:
                # Add bias for all the layers whose bias is None
                for name, module in ordered_conv_linear_nodes:
                    if module.bias is None:
                        if isinstance(module, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                            output_size = module.out_channels
                        elif isinstance(module, torch.nn.Linear):
                            output_size = module.out_features
                        module.bias = torch.nn.Parameter(torch.zeros(output_size))
                        module.bias.data = module.bias.data.to(device=module.weight.device)

            # Quantize full model
            dummy_tensors = utils.create_rand_tensors_given_shapes(input_shape)
            dummy_tensors = [tensor.to(utils.get_device(model)) for tensor in dummy_tensors]
            q = qsim.QuantizationSimModel(model=model, quant_scheme=quant_params.quant_scheme, rounding_mode=quant_params.round_mode, default_output_bw=quant_params.act_bw, default_param_bw=quant_params.weight_bw, in_place=True, dummy_input=dummy_tensors, config_file=quant_params.config_file)

            # make sure  model got updated in-place before we use it for bc updates
            assert (q.model is model)

            if self.args['bias_correction']:
                # updates to skip_output_activation and layers_to_ignore
                for name, module in model.named_modules():
                    # Skip all layer's output quantization
                    if isinstance(module, QcQuantizeWrapper):
                        module.output_quantizers[0].enabled = False

            q.compute_encodings(pass_data_through_model, None)

            # For first conv layer, perform analytical bc if perform_only_empirical_bias_corr is set to False
            # and layer is not marked to be ignored during bc.
            if self.args['bias_correction']:
                if not perform_only_empirical_bias_corr:
                    module_name, module = ordered_conv_linear_nodes[0]
                    if module not in layers_to_ignore:
                        logger.info('Correcting layer %s using Analytical Bias Correction', module_name)
                        quantize_layer = utils.get_layer_by_name(model, module_name)
                        call_analytical_mo_correct_bias(quantize_layer, None, None)
                        logger.info('Corrected bias for the layer')
                        ordered_conv_linear_nodes.pop(0)

                for module_name, module in ordered_conv_linear_nodes:
                    # Ignore all layers which are skipped by user
                    if module in layers_to_ignore:
                        continue
                    else:
                        # make sure module is in the model used by qsim.
                        assert (module in list(q.model.modules()))
                        # Analytical Bias Correction is only done for Conv layers
                        reference_layer = utils.get_layer_by_name(model_copy, module_name)
                        quantize_layer = utils.get_layer_by_name(model, module_name)

                        if module in conv_bn_dict.keys():

                            bn_layer_info = conv_bn_dict[module]

                            if perform_only_empirical_bias_corr or bn_layer_info is None or bn_layer_info.input_bn is None:
                                logger.info('Correcting layer %s using Empirical Bias Correction', module_name)
                                bias_correction = libpymo.BiasCorrection()

                                # Get output from quantized model and reference model

                                for images_in_one_batch, _ in data_loader_n_samples_bias_corr:
                                    reference_output_batch = get_output_data(reference_layer, model_copy, images_in_one_batch)
                                    quantized_model_output_batch = get_output_data(quantize_layer, model, images_in_one_batch)

                                    if isinstance(reference_layer, torch.nn.Linear):
                                        extended_shape = np.concatenate((reference_output_batch.shape, np.array([1, 1])))
                                        reference_output_batch = reference_output_batch.reshape(extended_shape)
                                        quantized_model_output_batch = quantized_model_output_batch.reshape(extended_shape)

                                    bias_correction.storePreActivationOutput(reference_output_batch)
                                    bias_correction.storeQuantizedPreActivationOutput(quantized_model_output_batch)

                                call_empirical_mo_correct_bias(module, bias_correction)

                            else:
                                logger.info('Correcting layer %s using Analytical Bias Correction', module_name)
                                call_analytical_mo_correct_bias(quantize_layer, bn_layer_info.input_bn, bn_layer_info.in_activation_type)

                            logger.info('Corrected bias for the layer')

            if self.args['bias_correction']:
                # set output_quantizers back to true
                # updates to skip_output_activation and layers_to_ignore
                for name, module in model.named_modules():
                    # Skip all layer's output quantization
                    if isinstance(module, QcQuantizeWrapper):
                        module.output_quantizers[0].enabled = True
                q.compute_encodings(pass_data_through_model, None)
            # set model to train
            model.train()
except ImportError:
    Warning('Aimet installation was incomplete. Co_lib may not use quantized function')
