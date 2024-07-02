import numpy as np
from Layers import Base, Helpers
import copy

class BatchNormalization(Base.BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.trainable = True
        self.channels = channels
        self.initialize()
        self._optimizer = None
        self.moving_mean = None
        self.moving_var = None
        self.decay = 0.8

    def initialize(self, weights_initializer=None, bias_initializer=None):
        self.gamma = np.ones(self.channels)
        self.beta = np.zeros(self.channels)

    def forward(self, input_tensor):
        input_data = input_tensor
        is_conv = False
        if input_data.ndim == 4:
            is_conv = True
            input_data = self.reformat(input_data)
        self.input_data = input_data

        if self.testing_phase is True:
            if self.moving_mean is None or self.moving_var is None:
                print("Train the model before testing")
            self.mean = self.moving_mean
            self.variance = self.moving_var
        else:
            self.mean = np.mean(input_data, axis=0)
            self.variance = np.var(input_data, axis=0)
            if self.moving_mean is None:
                self.moving_mean = copy.deepcopy(self.mean)
                self.moving_var = copy.deepcopy(self.variance)
            else:
                self.moving_mean = self.moving_mean * self.decay + self.mean * (1 - self.decay)
                self.moving_var = self.moving_var * self.decay + self.variance * (1 - self.decay)

        self.normalized_input = (input_data - self.mean) / np.sqrt(self.variance + np.finfo(float).eps)
        output = self.gamma * self.normalized_input + self.beta
        if is_conv:
            output = self.reformat(output)
        return output

    def backward(self, error_tensor):
        error_data = error_tensor
        is_conv = False
        if error_data.ndim == 4:
            is_conv = True
            error_data = self.reformat(error_data)

        grad_gamma = np.sum(error_data * self.normalized_input, axis=0)
        grad_beta = np.sum(error_data, axis=0)
        grad_input = Helpers.compute_bn_gradients(error_data, self.input_data, self.gamma, self.mean, self.variance)

        if self._optimizer is not None:
            self._optimizer.weight.calculate_update(self.gamma, grad_gamma)
            self._optimizer.bias.calculate_update(self.beta, grad_beta)

        if is_conv:
            grad_input = self.reformat(grad_input)
        self.gradient_weights = grad_gamma
        self.gradient_bias = grad_beta
        return grad_input

    def reformat(self, tensor):
        if tensor.ndim == 4:
            self.reformat_shape = tensor.shape
            batch_size, channels, height, width = tensor.shape
            tensor = tensor.reshape(batch_size, channels, height * width)
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(batch_size * height * width, channels)
            return tensor
        else:
            batch_size, channels, height, width = self.reformat_shape
            tensor = tensor.reshape(batch_size, height * width, channels)
            tensor = tensor.transpose(0, 2, 1)
            tensor = tensor.reshape(batch_size, channels, height, width)
            return tensor

    @property
    def weights(self):
        return self.gamma

    @weights.setter
    def weights(self, gamma):
        self.gamma = gamma

    @property
    def bias(self):
        return self.beta

    @bias.setter
    def bias(self, beta):
        self.beta = beta

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer.weight = copy.deepcopy(optimizer)
        self._optimizer.bias = copy.deepcopy(optimizer)