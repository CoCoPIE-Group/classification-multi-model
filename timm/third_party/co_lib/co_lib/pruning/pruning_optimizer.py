import numpy as np
import torch

from co_lib.optimizer_base import SpOptimizerBase, OptimizerBase
from co_lib.pruning.register_pruning import PRUNING_ZOOM
from co_lib.register_optimizer import register_optimizer
try:
    from co_adv_lib import ADV_PRUNING_ZOOM
except:
    ADV_PRUNING_ZOOM = []


@register_optimizer(1)
class PruneOptimizer(SpOptimizerBase):
    # compression_zoom = [[Admm, 0], [Magnitude, 0]]
    compression_zoom = PRUNING_ZOOM + ADV_PRUNING_ZOOM

    def _init(self, args=None, model=None, optimizer=None, logger=None, **kwargs):
        if isinstance(args, dict):
            self.args = args.get('prune', None)
        super(PruneOptimizer, self)._init(args, model, optimizer, logger, **kwargs)

    def _add_algorithm_to_pipline(self, algorithm, priority=0):
        algorithm = algorithm.build(args=self.args, model=self.model, optimizer=self.optimizer)
        super()._add_algorithm_to_pipline(algorithm, priority)

    @classmethod
    def argument_parser(cls, parser):
        main_prune_parse_arguments(parser)
        prune_base_parse_arguments(parser)
        utils_prune_parse_arguments(parser)
        super().argument_parser(parser)


# TODO RM UNUSED parser
def prune_base_parse_arguments(parser):
    prune_retrain = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--sp-backbone",
        action="store_true",
        help="enable sparse backboen training",
    )

    prune_retrain.add_argument(
        '--sp-retrain',
        action='store_true',
        help="Retrain a pruned model",
    )
    prune_retrain.add_argument(
        '--sp-admm',
        action='store_true',
        default=False,
        help="for admm pruning",
    )
    prune_retrain.add_argument(
        '--sp-admm-multi',
        action='store_true',
        default=False,
        help="for multi-level admm pruning",
    )
    prune_retrain.add_argument(
        '--sp-retrain-multi',
        action='store_true',
        help="For multi-level retrain a pruned model",
    )

    parser.add_argument(
        '--sp-prune-ratios',
        type=float,
        help="for admm prune ratios, override the prune ratios in config file",
    )
    parser.add_argument(
        '--sp-config-file',
        type=str,
        help="define config file",
    )
    parser.add_argument(
        '--sp-subset-progressive',
        action='store_true',
        help="ADMM from a sparse model",
    )
    parser.add_argument(
        '--sp-admm-fixed-params',
        action='store_true',
        help="ADMM from a sparse model, with a fixed subset of parameters",
    )
    parser.add_argument(
        '--sp-no-harden',
        action='store_true',
        help="Do not harden the pruned matrix",
    )
    parser.add_argument(
        '--nv-sparse',
        action='store_true',
        help="use nv's sparse library ASP",
    )
    parser.add_argument(
        '--sp-load-prune-params',
        type=str,
        help="Load the params used in pruning only",
    )
    parser.add_argument(
        '--sp-store-prune-params',
        type=str,
        help="Store the params used in pruning only",
    )
    parser.add_argument(
        '--generate-rand-seq-gap-yaml',
        action='store_true',
        help="whether to generate a set of randomly selected sequential GaP yamls",
    )


def main_prune_parse_arguments(parser):
    parser.add_argument(
        '--sp-store-weights',
        type=str,
        help="store the final weights, "
        "maybe used by the next stage pruning",
    )
    parser.add_argument(
        "--sp-lars",
        action="store_true",
        help="enable LARS learning rate scheduler",
    )
    parser.add_argument(
        '--sp-lars-trust-coef',
        type=float,
        default=0.001,
        help="LARS trust coefficient",
    )


def utils_prune_parse_arguments(parser):
    admm_args = parser.add_argument_group('Multi level admm arguments')
    admm_args.add_argument(
        '--sp-load-frozen-weights',
        type=str,
        help='the weights that are frozen '
        'throughout the pruning process',
    )
