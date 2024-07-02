import numpy as np


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha  # Regularization weight

    def calculate_gradient(self, weights):
        # Calculate the subgradient of L1 norm
        return self.alpha * np.sign(weights)

    def norm(self, weights):
        # Calculate the L1 norm of the weights
        return self.alpha * np.sum(np.abs(weights))


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha  # Regularization weight

    def calculate_gradient(self, weights):
        # Calculate the gradient of L2 norm
        return self.alpha * weights

    def norm(self, weights):
        # Calculate the L2 norm of the weights
        return self.alpha * np.sum(weights ** 2)
