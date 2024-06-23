import numpy as np
from Layers import BatchNormalization
from Optimization import *


def test_batch_normalization_updates():
    input_tensor = np.random.randn(10, 5)
    channels = input_tensor.shape[-1]
    layer = BatchNormalization(1, channels)
    layer.optimizer = Optimizers.Sgd(0.01)  # Assuming this is the correct usage in your context

    initial_output = layer.forward(input_tensor)
    print("Initial output calculated.")

    for _ in range(10):
        error_tensor = -layer.forward(input_tensor)
        layer.backward(error_tensor)

    new_output = layer.forward(input_tensor)
    print("New output calculated after training.")

    assert np.sum(np.power(new_output, 2)) < np.sum(np.power(initial_output, 2)), "Output did not decrease."


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    test_batch_normalization_updates()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
