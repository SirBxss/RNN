import numpy as np


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        grad = self.alpha * np.sign(weights)
        print(f"Regularizer Gradient: {grad}")
        return grad

    def norm(self, weights):
        norm_value = self.alpha * np.sum(np.abs(weights))
        print(f"Regularizer Norm: {norm_value}")
        return norm_value


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha  # Regularization weight

    def calculate_gradient(self, weights):
        # Calculate the gradient of L2 norm
        return self.alpha * weights

    def norm(self, weights):
        # Calculate the L2 norm of the weights
        return self.alpha * np.sum(weights ** 2)
