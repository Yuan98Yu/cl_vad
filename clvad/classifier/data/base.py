from abc import ABC, abstractmethod


class BaseADDataset(ABC):
    """Anomaly detection dataset base class."""

    def __init__(self):
        super().__init__()
        pass

    def __repr__(self):
        return self.__class__.__name__
