from Layers.Base import BaseLayer
import numpy as np


class Dropout(BaseLayer):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.mask = 0

    def forward(self, input_tensor):
        if self.testing_phase:
            return input_tensor
        else:
            self.mask = np.random.binomial(1, self.probability, size=input_tensor.shape) / self.probability
            return input_tensor * self.mask

    def backward(self, error_tensor):
        return error_tensor * self.mask if self.mask is not None else error_tensor