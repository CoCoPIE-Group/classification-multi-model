import sys
import logging
import types
import numpy as np
import torch
from co_lib.pruning import PruneBase
from co_lib.pruning import utils as utils_pr
from co_lib.pruning.register_pruning import register_pruning


@register_pruning(1)
class Magnitude(PruneBase):
    __method_name__ = "sp_retrain"

    def __init__(self, *args, **kwargs):
        super(Magnitude, self).__init__(*args, **kwargs)
        # args = kwargs.get('args', None)
        # model = kwargs.get('model', None)
        # optimizer = kwargs.get('optimizer', None)
        pre_defined_mask = kwargs.get('args', None)
        # self.args = args
        # # we assume the model does not change during execution
        # self.model = model
        # self.optimizer = optimizer
        self.pattern = self.args.retrain_mask_pattern
        self.pre_defined_mask = pre_defined_mask  # as model's state_dict
        self.sparsity = self.args.retrain_mask_sparsity
        self.seed = self.args.retrain_mask_seed
        self.sp_mask_update_freq = self.args.sp_mask_update_freq
        self.update_init_method = self.args.sp_update_init_method
        self.__optimizer = None


        self.logger.info("Command line:")
        self.logger.info(' '.join(sys.argv))
        self.logger.info("Args:")
        self.logger.info(args)

        self.masks = {}
        self.masked_layers = {}
        # self.configs, self.prune_ratios = utils_pr.load_configs(model, args.sp_config_file, self.logger, args)

        if "masked_layers" in self.configs:
            self.masked_layers = self.configs['masked_layers']
        else:
            for name, W in (self.model.named_parameters()):
                self.masked_layers[utils_pr.canonical_name(name)] = None

        if "fixed_layers" in self.configs:
            self.fixed_layers = self.configs['fixed_layers']
        else:
            self.fixed_layers = None
        self.fixed_layers_save = {}

        if "upper_bound" in self.configs:
            self.upper_bound = self.configs['upper_bound']
        else:
            self.upper_bound = None
        if "lower_bound" in self.configs:
            self.lower_bound = self.configs['lower_bound']
        else:
            self.lower_bound = None
        if "mask_update_decay_epoch" in self.configs:
            self.mask_update_decay_epoch = self.configs['mask_update_decay_epoch']
        else:
            self.mask_update_decay_epoch = None

        if "seq_gap_layer_indices" in self.configs:
            self.seq_gap_layer_indices = self.configs['seq_gap_layer_indices']
            self.all_part_name_list = []
        else:
            self.seq_gap_layer_indices = None

    @staticmethod
    def argument_parser(parser):
        parser.add_argument(
            '--retrain-mask-pattern',
            type=str,
            default='weight',
            help="retrain mask pattern",
        )
        parser.add_argument(
            '--sp-update-init-method',
            type=str,
            default='weight',
            help="mask update initialization method",
        )
        parser.add_argument(
            '--sp-mask-update-freq',
            type=int,
            default=10,
            help="how many epochs to update sparse mask",
        )
        parser.add_argument(
            '--retrain-mask-sparsity',
            type=float,
            default=-1.0,
            help="sparsity of a retrain mask, used when retrain-mask-pattern is set to NOT being 'weight' ",
        )
        parser.add_argument(
            '--retrain-mask-seed',
            type=int,
            default=None,
            help="seed to generate a random mask",
        )
        parser.add_argument(
            '--sp-prune-before-retrain',
            action='store_true',
            help="Prune the loaded model before retrain, in case of loading a dense model",
        )
        parser.add_argument(
            '--output-compressed-format',
            action='store_true',
            help="output compressed format",
        )
        parser.add_argument(
            "--sp-grad-update",
            action="store_true",
            help="enable grad update when training in random GaP",
        )
        parser.add_argument(
            "--sp-grad-decay",
            type=float,
            default=0.98,
            help="The decay number for gradient",
        )
        parser.add_argument(
            "--sp-grad-restore-threshold",
            type=float,
            default=-1,
            help="When the decay",
        )
        parser.add_argument(
            "--sp-global-magnitude",
            action="store_true",
            help="Use global magnitude to prune models",
        )
        parser.add_argument(
            '--sp-pre-defined-mask-dir',
            type=str,
            default=None,
            help="using another sparse model to init sparse mask",
        )

    @classmethod
    def build(cls, *args, **kwargs):
        '''
        Used to determine whether to create a class, if not return None
        :param args:
        :param kwargs:
        :return:
        '''

        return cls(*args, **kwargs)

    def init(self, *args, **kwargs):
        self.harden_weights()
        self.generate_mask(self.pre_defined_mask)
        self.generate_masked_optimizer()

    def generate_masked_optimizer(self):
        assert (self.__optimizer is None), "__optimizer is already defined. Please change!"
        self.__optimizer = self.optimizer
        self.__optimizer.__step = self.optimizer.step

        def __step(opt_self, *args, **kwargs):
            # prune gradients before step method
            with torch.no_grad():
                for name, W in (self.model.named_parameters()):
                    if name in self.masks:
                        W.grad.mul_((self.masks[name] != 0).type(W.grad.dtype))
            # call original optimizer step method
            rval = opt_self.__step(*args, **kwargs)
            # prune parameters after step method
            with torch.no_grad():
                for name, W in (self.model.named_parameters()):
                    if name in self.masks:
                        # dtype = W.dtype
                        W.mul_((self.masks[name] != 0).type(W.dtype))
            return rval

        self.__optimizer.step = types.MethodType(__step, self.__optimizer)

    def generate_mask(self, pre_defined_mask=None):
        masks = {}
        # import pdb; pdb.set_trace()
        if self.pattern == 'weight':

            with torch.no_grad():
                for name, W in (self.model.named_parameters()):

                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    # self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        # self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask

                    # bias.data.mul_(mask)
            # for name in masks:
            #    print("Current mask includes:", name)
            # if 'weight' in name:
            #    print(name, (np.sum(non_zeros) + 0.0) / np.size(non_zeros) )
            # exit()

        elif self.pattern == 'random':
            if self.seed is not None:
                print("Setting the random mask seed as {}".format(self.seed))
                np.random.seed(self.seed)

            with torch.no_grad():
                # self.sparsity (args.retrain_mask_sparsity) will override prune ratio config file
                if self.sparsity > 0:
                    sparsity = self.sparsity

                    for name, W in (self.model.named_parameters()):
                        if 'weight' in name and 'bn' not in name:
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size * (1 - sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        self.masks[name] = zero_mask

                else:  # self.sparsity < 0

                    for name, W in (self.model.named_parameters()):
                        if (utils_pr.canonical_name(name) not in self.prune_ratios.keys()) and (name not in self.prune_ratios.keys()):
                            continue
                        if name in self.prune_ratios:
                            # Use prune_profiles[] to indicate which layers to random masked
                            sparsity = self.prune_ratios[name]
                            '''
                            if sparsity < 0.001:
                                continue
                            '''
                            non_zeros = np.zeros(W.data.shape).flatten()
                            non_zeros[:int(non_zeros.size * (1 - sparsity))] = 1

                            np.random.shuffle(non_zeros)

                            non_zeros = np.reshape(non_zeros, W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()
                        else:
                            non_zeros = np.ones(W.data.shape)
                            non_zeros = non_zeros.astype(np.float32)
                            zero_mask = torch.from_numpy(non_zeros).cuda()

                        self.masks[name] = zero_mask

                # # DEBUG:
                DEBUG = False
                if DEBUG:
                    for name, W in (self.model.named_parameters()):
                        m = self.masks[name].detach().cpu().numpy()
                        total_ones = np.sum(m)
                        total_size = np.size(m)
                        print(name, m.shape, (total_ones + 0.0) / total_size)

                # exit()
        # TODO
        elif self.pattern == 'regular':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    if 'weight' in name and 'bn' not in name:

                        ouputSize, inputSize = W.data.shape[0], W.data.shape[1]
                        non_zeros = np.zeros(W.data.shape)
                        non_zeros = np.squeeze(non_zeros)

                        if 'sa1.conv_blocks.0.0.weight' in name or 'sa1.conv_blocks.1.0.weight' in name or 'sa1.conv_blocks.2.0.weight' in name:
                            non_zeros[::self.args.mask_sample_rate, ::] = 1

                        else:
                            non_zeros[::self.args.mask_sample_rate, ::self.args.mask_sample_rate] = 1

                        non_zeros = np.reshape(non_zeros, W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()

                    else:
                        non_zeros = 1 - np.zeros(W.data.shape)
                        non_zeros = non_zeros.astype(np.float32)
                        zero_mask = torch.from_numpy(non_zeros).cuda()
                    self.masks[name] = zero_mask
        elif self.pattern == 'global_weight':
            with torch.no_grad():
                all_w = []
                all_name = []
                print('Concatenating all weights...')
                for name, W in self.model.named_parameters():
                    if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                        continue
                    all_w.append(W.detach().cpu().numpy().flatten())
                    all_name.append(name)
                np_w = all_w[0]
                for i in range(1, len(all_w)):
                    np_w = np.append(np_w, all_w[i])

                # print(np_w.shape)
                print("All weights concatenated!")
                print("Start sorting all the weights...")
                np_w = np.sort(np.abs(np_w))
                print("Sort done!")
                L = len(np_w)
                # print(np_w)
                if self.args.retrain_mask_sparsity >= 0.0:
                    thr = np_w[int(L * self.args.retrain_mask_sparsity)]

                    for name, W in self.model.named_parameters():
                        if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                            continue

                        np_mask = np.abs(W.detach().cpu().numpy()) > thr
                        print(name, np.size(np_mask), np.sum(np_mask), float(np.sum(np_mask)) / np.size(np_mask))

                        self.masks[name] = torch.from_numpy(np_mask).cuda()

                    total_non_zero = 0
                    total_size = 0
                    with open('gw_sparsity.txt', 'w') as f:
                        for name, W in sorted(self.model.named_parameters()):
                            if (utils_pr.canonical_name(name) not in self.prune_ratios) and (name not in self.prune_ratios):
                                continue
                            np_mask = self.masks[name].detach().cpu().numpy()
                            sparsity = 1.0 - float(np.sum(np_mask)) / np.size(np_mask)
                            if sparsity < 0.5:
                                sparsity = 0.0

                            if sparsity < 0.5:
                                total_non_zero += np.size(np_mask)
                            else:
                                total_non_zero += np.sum(np_mask)
                            total_size += np.size(np_mask)

                            f.write("{}: {}\n".format(name, sparsity))
                    print("Thr:{}".format(thr))
                    print("{},{},{}".format(total_non_zero, total_size, float(total_non_zero) / total_size))
                    exit()

        elif self.pattern == 'none':
            with torch.no_grad():
                for name, W in self.model.named_parameters():
                    non_zeros = np.ones(W.data.shape)
                    non_zeros = non_zeros.astype(np.float32)
                    zero_mask = torch.from_numpy(non_zeros).cuda()
            self.masks[name] = zero_mask

        elif self.pattern == "pre_defined":
            assert pre_defined_mask is not None, "\n\n * Error, pre_defined sparse mask model must be declared!"
            with torch.no_grad():
                for name, W in pre_defined_mask.items():
                    if (utils_pr.canonical_name(name) not in self.masked_layers) and (name not in self.masked_layers):
                        continue

                    weight = W.cpu().detach().numpy()
                    non_zeros = weight != 0
                    non_zeros = non_zeros.astype(np.float32)
                    num_nonzeros = np.count_nonzero(non_zeros)
                    total_num = non_zeros.size
                    sparsity = 1 - (num_nonzeros * 1.0) / total_num
                    # self.logger.info("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity)))
                    print(("{}: {}, {}, {}".format(name, str(num_nonzeros), str(total_num), str(sparsity))))
                    if sparsity < 0.1:
                        # self.logger.info("{}: sparsity too low, skip".format(name))
                        print("{}: sparsity too low, skip".format(name))
                        continue
                    zero_mask = torch.from_numpy(non_zeros).cuda()

                    self.masks[name] = zero_mask

        else:
            print("mask pattern not recognized!")
            exit()

        return self.masks
