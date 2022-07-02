from co_lib.pruning import PruneBase
from co_lib.pruning.pruning_utils import *
from co_lib.pruning.register_pruning import register_pruning


# @PruneOptimizer.compression_zoom_register(0)
@register_pruning(1)
class Admm(PruneBase):
    __method_name__ = "sp_admm"

    def __init__(self, *args, **kwargs):
        super(Admm, self).__init__(*args, **kwargs)
        # args = kwargs.get('args', None)
        # self.args = args
        # model = kwargs.get('model', None)
        # logger = kwargs.get('logger', None)
        initialize = kwargs.get('initialize', None)
        # this is to keep in CPU
        self.ADMM_U = {}
        self.ADMM_Z = {}
        # this is the identical copy in GPU. We do this separation
        # because in some cases in GPU run out of space if modified
        # directly
        self.ADMM_U_GPU = {}
        self.ADMM_Z_GPU = {}
        self.rhos = {}
        self.rho = None

        # assert args.sp_config_file is not None, "Config file must be specified for ADMM"
        self.logger.info("Initializing ADMM pruning algorithm")

        if self.args.sp_admm_update_epoch is not None:
            self.update_epoch = self.args.sp_admm_update_epoch
        elif 'admm_update_epoch' in self.configs:
            self.update_epoch = self.configs["admm_update_epoch"]
        elif self.args.train_epochs:
            self.update_epoch = int(self.args.train_epochs / 5)
        else:
            self.update_epoch = None
        if self.args.sp_admm_update_batch is not None:
            self.update_batch = self.args.sp_admm_update_batch
        elif 'admm_update_batch' in self.configs:
            self.update_batch = self.configs["admm_update_batch"]
        else:
            self.update_batch = None

        assert (self.update_epoch is None and self.update_batch is not None) or (self.update_epoch is not None and self.update_batch is None)

        assert self.prune_ratios is not None
        if 'rho' in self.configs:
            self.rho = self.configs['rho']
        else:
            assert self.args.sp_admm_rho is not None
            self.rho = self.args.sp_admm_rho
        self.logger.info("ADMM rho is set to {}".format(str(self.rho)))

        if self.args.sp_load_prune_params is not None:
            self.prune_load_params()
        elif initialize and ('admm_initialize' not in args or args.admm_initialize):
            self.init()

        if self.args.admm_debug:
            self.admm_debug = True
        else:
            self.admm_debug = False

    @staticmethod
    def argument_parser(parser):
        admm_args = parser.add_argument_group('Admm arguments')
        update_freq = admm_args.add_mutually_exclusive_group()

        update_freq.add_argument(
            '--sp-admm-update-epoch',
            type=int,
            help="how often we do admm update",
        )
        update_freq.add_argument(
            '--sp-admm-update-batch',
            type=int,
            help="update admm after how many minibatches",
        )
        admm_args.add_argument(
            '--sp-admm-rho',
            type=float,
            default=0.001,
            help="define rho for ADMM, overrides the rho specified in config file",
        )
        admm_args.add_argument(
            '--sp-admm-sparsity-type',
            type=str,
            default='gather_scatter',
            help="define sp_admm_sparsity_type: [irregular, block_punched, irregular_global, column, filter]",
        )

        admm_args.add_argument(
            '--sp-admm-lr',
            type=float,
            default=0.001,
            help="define learning rate for ADMM, reset to sp_admm_lr every time U,Z is updated. \
                  Overrides the learning rate of the outside training loop",
        )
        admm_args.add_argument(
            '--admm-debug',
            dest='admm_debug',
            action='store_true',
            help='debug mode of admm, print out some values (e.g., loss)',
        )

        admm_args.add_argument(
            '--sp-global-weight-sparsity',
            type=float,
            default=-1,
            help="Use global weight magnitude to prune, override the --sp-config-file",
        )
        admm_args.add_argument(
            '--sp-prune-threshold',
            type=float,
            default=-1.0,
            help="Used with --sp-global-weight-sparsity. \
                  Threshold of sparsity to prune. \
                  For example, if set this threshold = 0.1, then only prune layers with sparsity > 0.1 in a global fashion",
        )
        admm_args.add_argument(
            '--sp-block-irregular-sparsity',
            type=str,
            default="(0,0)",
            help="blocked and irregular sparsity in block + irregular sparse pattern",
        )
        admm_args.add_argument(
            '--sp-block-permute-multiplier',
            type=int,
            default=2,
        )

        # the following is for gather/scatter sparsity type
        admm_args.add_argument(
            '--sp-admm-block',
            default="(8,4)",
        )
        admm_args.add_argument(
            '--sp-admm-buckets-num',
            type=int,
            default=16,
        )
        # this is not needed, should be calculated
        # admm_args.add_argument('--sp-admm-bucket-axis', type=int, default=1)
        admm_args.add_argument(
            '--sp-admm-elem-per-row',
            type=int,
            default=1,
        )
        admm_args.add_argument(
            '--sp-admm-tile',
            type=str,
            default=None,
            help="in the form of (x,y) e.g. (256,256) \
                  x is the number of rows in a tile, -1 means all rows \
                  y is the number of cols in a tile, -1 means all cols",
        )

        # the following is for M:N pruning sparsity type
        admm_args.add_argument(
            '--sp-admm-select-number',
            type=int,
            default=4,
        )
        admm_args.add_argument(
            '--sp-admm-pattern-row-sub',
            type=int,
            default=1,
        )
        admm_args.add_argument(
            '--sp-admm-pattern-col-sub',
            type=int,
            default=4,
        )
        admm_args.add_argument(
            '--sp-admm-data-format',
            type=str,
            default=None,
            help="define sp_admm_format: [NHWC,NCHW]",
        )
        admm_args.add_argument(
            '--sp-admm-do-not-permute-conv',
            default=False,
            action='store_true',
            help="Do not permute conv filters",
        )

        # output compressed format
        admm_args.add_argument(
            '--sp-gs-output-v',
            type=str,
            default=None,
            help="output compressed format of a gs pattern",
        )
        admm_args.add_argument(
            '--sp-gs-output-ptr',
            type=str,
            default=None,
            help="output compressed format of a gs pattern",
        )

    def init(self, *args, **kwargs):
        first = True

        for key in self.prune_ratios:
            print("prune_ratios[{}]:{}".format(key, self.prune_ratios[key]))

        for (name, W) in self.model.named_parameters():
            if name not in self.prune_ratios:
                continue
            self.rhos[name] = self.rho
            prune_ratio = self.prune_ratios[name]

            self.logger.info("ADMM initialzing {}".format(name))
            updated_Z = self.prune_weight(name, W, prune_ratio, first)  # Z(k+1) = W(k+1)+U(k)  U(k) is zeros her
            # print("Done")

            first = False
            self.ADMM_Z[name] = updated_Z.detach().cpu().float()
            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)
            self.ADMM_U[name] = torch.zeros(W.shape).detach().cpu().float()
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)

        if (self.args.output_compressed_format) and (self.args.sp_gs_output_v is not None) and (self.args.sp_gs_output_ptr is not None):
            print("Compressed format output done!")
            exit()

    def before_each_train_epoch(self, *args, **kwargs):
        epoch = kwargs.get('epoch', 0)
        batch_idx = kwargs.get('batch_idx', epoch)
        if ((self.update_epoch is not None) and ((epoch == 0) or (epoch % self.update_epoch != 0))) or ((self.update_batch is not None) and ((batch_idx == 0) or (batch_idx % self.update_batch != 0))):
            return

        # this is to avoid the bug that GPU memory overflow
        for key in self.ADMM_Z:
            del self.ADMM_Z_GPU[key]
        for key in self.ADMM_U:
            del self.ADMM_U_GPU[key]
        first = True
        for i, (name, W) in enumerate(self.model.named_parameters()):
            if name not in self.prune_ratios:
                continue
            Z_prev = None
            W_CPU = W.detach().cpu().float()

            admm_z = W_CPU + self.ADMM_U[name]  # Z(k+1) = W(k+1)+U[k]

            updated_Z = self.prune_weight(name, admm_z, self.prune_ratios[name], first)  # equivalent to Euclidean Projection
            first = False
            self.ADMM_Z[name] = updated_Z.detach().cpu().float()

            self.ADMM_U[name] = (W_CPU - self.ADMM_Z[name] + self.ADMM_U[name]).float()  # U(k+1) = W(k+1) - Z(k+1) +U(k)

            self.ADMM_Z_GPU[name] = self.ADMM_Z[name].detach().to(W.device).type(W.dtype)
            self.ADMM_U_GPU[name] = self.ADMM_U[name].detach().to(W.device).type(W.dtype)

    def after_scheduler_step(self, *args, **kwargs):

        epoch = kwargs.get('epoch', -1)
        args = self.args
        optimizer = self.optimizer
        admm_epoch = args.sp_admm_update_epoch
        lr = None

        if (epoch) % admm_epoch == 0:
            lr = args.sp_admm_lr
        else:
            admm_epoch_offset = (epoch) % admm_epoch

            admm_step = admm_epoch / 3  # roughly every 1/3 admm_epoch.

            lr = args.sp_admm_lr * (0.1**(admm_epoch_offset // admm_step))

        # print(admm_epoch, args.sp_admm_lr, (epoch) % admm_epoch, lr)
        # input('?')

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def update_loss(self, loss):
        assert loss is not None, "loss can't be None"
        _, _, combined_loss = self.update_combined_loss(loss)
        return combined_loss

    def update_combined_loss(self, ce_loss):
        admm_loss = {}
        for i, (name, W) in enumerate(self.model.named_parameters()):  # initialize Z (for both weights and bias)
            if name not in self.prune_ratios:
                continue
            if self.prune_ratios[name] == 0.0:
                continue
            admm_loss[name] = (0.5 * self.rhos[name] * (torch.norm(W.float() - self.ADMM_Z_GPU[name].float() + self.ADMM_U_GPU[name].float(), p=2)**2)).float()

        total_admm_loss = 0
        for k, v in admm_loss.items():
            total_admm_loss += v
        mixed_loss = total_admm_loss + ce_loss

        if self.admm_debug:
            ce_loss_np = ce_loss.data.cpu().numpy()
            _admm_loss_np = total_admm_loss.data.cpu().numpy()
            print("ce_loss:{}, admm_loss:{}, mixed_loss:{}".format(ce_loss_np, _admm_loss_np, mixed_loss.data.cpu().numpy()))

        return ce_loss, admm_loss, mixed_loss
