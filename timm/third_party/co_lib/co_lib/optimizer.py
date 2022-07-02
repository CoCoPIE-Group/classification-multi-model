import sys
import argparse
from dotmap import DotMap
from co_lib import OptimizerBase
from co_lib.register_optimizer import COMPRESSION_ZOOM
try:
    from co_adv_lib import ADV_COMPRESSION_ZOOM
except:
    ADV_COMPRESSION_ZOOM = []


class CoLib(OptimizerBase):
    compression_zoom = COMPRESSION_ZOOM + ADV_COMPRESSION_ZOOM

    def init(self, args=None, model=None, optimizer=None, logger=None, data_loader=None):
        # re init co lib
        self._re_init()
        class_name = self.__class__.__name__
        self.logger.info(f'Start {class_name}.init')
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        if isinstance(args, dict):
            args = DotMap(args, _dynamic=False)
        self._init(args=args, model=model, optimizer=optimizer, logger=logger, data_loader=data_loader)
        self._call('init', args=args, model=model, optimizer=optimizer, logger=logger, data_loader=data_loader)
        self.review_colb_state_info()


    def review_colb_state_info(self):
        compression_zoom = {i[0].__name__:[j[0].__name__ for j in i[0].compression_zoom] for i in self.compression_zoom}
        self.logger.info(f'available compression algrithm is {compression_zoom} ')
        self.logger.info(f'If you find some algorithm was not shown here, please check whether fully installed the co_lib or include all co_lib related package ')

        # applied pipline checked
        new_piplines = []
        for items in self.piplines:
            algorithm,priority = items
            if len(algorithm.piplines)>0:
                new_piplines.append(items)
        self.piplines = new_piplines
        piplines = {i[0].__class__.__name__: [j[0].__class__.__name__ for j in i[0].piplines] for i in self.piplines}
        self.logger.info(f'Applied algorithm is {piplines} ')


    def _re_init(self):
        self.piplines = []
        self.init_state = False



    def _add_algorithm_to_pipline(self, algorithm, priority=0):
        algorithm = algorithm.build()
        if algorithm is not None:
            if issubclass(algorithm.__class__, OptimizerBase):
                self.piplines.append([algorithm, priority])
            else:
                self.logger.warning(f"algorithm type has problems, unknown {algorithm.__class__.__name__}")


Co_Lib = CoLib()
