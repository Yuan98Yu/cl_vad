from abc import ABC, abstractmethod


class Trainer(ABC):
    @abstractmethod
    def train(self, data_loader, logger):
        pass

    @abstractmethod
    def load_ckpt(self, args):
        pass

    @abstractmethod
    def save_ckpt(self, save_dict,
                  file_name=None, save_path=None, save_dir=None):
        pass

    @abstractmethod
    def __create_optimizer(self, opt_config):
        pass

    @abstractmethod
    def __create_criterion(self, gpu):
        pass

    @abstractmethod
    def __create_schedular(self, optimizer, opt_config):
        pass
