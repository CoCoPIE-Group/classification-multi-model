'''
base optimizer class
'''
import logging

from co_lib.common_base import CompressionBase


def iterative_call(func):

    def wrap(self, *args, **kwargs):
        class_name = self.__class__.__name__
        func_name = func.__name__
        self.logger.info(f'Start {class_name}.{func_name}')
        func(self, *args, **kwargs)
        self._call(func_name, *args, **kwargs)

    return wrap


class OptimizerBase:
    # [[object:CompressionBase,priority]] default priority is 0
    compression_zoom = []

    def __init__(self, *args, **kwargs):
        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.piplines = []
        self.init_state = False

    @classmethod
    def compression_zoom_register(cls, priority=0):

        def wrap(_class):
            cls.compression_zoom.append(_class, priority)

        return wrap

    def _init_logging(self, logger):
        if logger is None:
            logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
            self.logger = logging.getLogger(self.__class__.__name__)
        else:
            self.logger = logger

    def _init_piplines(self):
        self.init_state = True
        for compression_algorithm in self.compression_zoom:
            self._add_algorithm_to_pipline(*compression_algorithm)
        self._sort_pipline()

    @iterative_call
    def init(self, *args, **kwargs):
        self._init(**kwargs)

    def _init(self, args=None, model=None, optimizer=None, logger=None, **kwargs):
        assert args is not None, "args can't be none"
        assert model is not None, "args can't be none"
        assert optimizer is not None, "args can't be none"
        data_loader = kwargs.get('data_loader', None)
        if not hasattr(self, 'data_loader'):
            self.data_loader = data_loader
        if not hasattr(self, 'args'):
            self.args = args
        if not hasattr(self, 'model'):
            self.model = model
        if not hasattr(self, 'optimizer'):
            self.optimizer = optimizer

        self._init_logging(logger)
        self._init_piplines()

    @classmethod
    def argument_parser(cls, parser):
        for algorithm in cls.compression_zoom:
            algorithm, _ = algorithm
            algorithm.argument_parser(parser)

    @iterative_call
    def before_each_train_epoch(self, *args, **kwargs):
        pass

    @iterative_call
    def after_scheduler_step(self, *args, **kwargs):
        pass

    def update_loss(self, loss):
        class_name = self.__class__.__name__
        # self.logger.info(f'Start {class_name}.update_loss')
        for algorithm in self.piplines:
            algorithm, _ = algorithm
            if getattr(algorithm, "update_loss", None) is not None:
                loss = algorithm.update_loss(loss)
        return loss

    @iterative_call
    def before_optimizer_step(self, *args, **kwargs):
        pass

    @iterative_call
    def after_optimizer_step(self, *args, **kwargs):
        pass

    def _call(self, func, *args, **kwargs):
        for algorithm, _ in self.piplines:
            self_class_name = self.__class__.__name__
            if self_class_name not in kwargs:
                kwargs[self_class_name] = self
            eval(f'algorithm.{func}(*args, **kwargs)')

    def _add_algorithm_to_pipline(self, algorithm, priority=0):
        pass

    def _sort_pipline(self):
        self.piplines = sorted(self.piplines, key=lambda x: x[1], reverse=True)

    def add_compression_algorithm(self, compression_class, priority=0):
        self.compression_zoom.append(compression_class)

        # if the optimizer already initialized, directly add this algorithm
        if self.init_state:
            self._add_algorithm_to_pipline(compression_class, priority)
            self._sort_pipline()


class SpOptimizerBase(OptimizerBase):
    piplines = []

    @iterative_call
    def init(self, *args, **kwargs):
        if 'CocoLib' in kwargs and hasattr(kwargs['CocoLib'], 'logger') and kwargs['CocoLib'].logger is not None:
            kwargs['logger'] = kwargs['CocoLib'].logger
        self._init(**kwargs)

    def _init_piplines(self):
        self.init_state = True
        for compression_algorithm in self.compression_zoom:
            _algorithm, _priority = compression_algorithm
            if _algorithm.__method_name__ in self.args and self.args[_algorithm.__method_name__]:
                self._add_algorithm_to_pipline(*compression_algorithm)

        # for compression_algorithm in self.compression_zoom:
        # 	self._add_algorithm_to_pipline(*compression_algorithm)
        self._sort_pipline()

    def _add_algorithm_to_pipline(self, algorithm, priority=0):
        if algorithm is not None:
            if issubclass(algorithm.__class__, CompressionBase):
                self.piplines.append([algorithm, priority])
            else:
                self.logger.warning("algorithm type has problems")

    @classmethod
    def build(cls):
        '''
        Used to determine whether to create a class, if not return None
        :param args:
        :param kwargs:
        :return:
        '''

        return cls()
