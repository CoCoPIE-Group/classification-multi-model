
from co_lib.optimizer_base import SpOptimizerBase, OptimizerBase
from co_lib.register_optimizer import register_optimizer

try:
    from co_lib.quantization.register_quantization import QUANTIZATION_ZOOM
    try:
        from co_adv_lib import ADV_QUANTIZATION_ZOOM
    except:
        ADV_QUANTIZATION_ZOOM = []


    @register_optimizer(1)
    class QuantizationOptimizer(SpOptimizerBase):
        compression_zoom = QUANTIZATION_ZOOM + ADV_QUANTIZATION_ZOOM

        def _init(self, args=None, model=None, optimizer=None, logger=None, **kwargs):
            if isinstance(args, dict) and 'quantization' in args:
                self.args = args.get('quantization', None)
            super(QuantizationOptimizer, self)._init(args, model, optimizer, logger, **kwargs)

        def _add_algorithm_to_pipline(self, algorithm, priority=0):
            algorithm = algorithm.build(args=self.args, model=self.model, optimizer=self.optimizer, data_loader=self.data_loader)
            super()._add_algorithm_to_pipline(algorithm, priority)
except:
    Warning('Aimet installation was incomplete. Co_lib may not use quantized function')
