from abc import ABC, abstractmethod

class BaseAlgorithm(ABC):
    @abstractmethod
    def process(self, image):
        pass

    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def plot_graph(self, image):
        pass
