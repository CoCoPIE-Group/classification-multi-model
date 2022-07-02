import logging


class CompressionBase:

    def __init__(self, *args, **kwargs):

        logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)


    @staticmethod
    def argument_parser(parser):
        pass

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
        pass

    def before_each_train_epoch(self, *args, **kwargs):
        pass

    def after_scheduler_step(self, *args, **kwargs):
        pass

    def update_loss(self, loss):
        return loss

    def after_optimizer_step(self, *args, **kwargs):
        pass
