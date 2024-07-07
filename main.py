import numpy as np
from Layers import BatchNormalization
from Optimization import *
from Layers import *
import NeuralNetwork

class test:
    def test_regularization_loss(self):
        import random
        fcl = FullyConnected.FullyConnected(4, 3)
        rnn = RNN.RNN(4, 4, 3)
        tolerance = 1e-5

        for layer in [fcl, rnn]:
            loss = []
            for reg in [False, True]:
                opt = Optimizers.Sgd(1e-3)
                if reg:
                    opt.add_regularizer(Constraints.L1_Regularizer(8e-2))
                net = NeuralNetwork.NeuralNetwork(opt, Initializers.Constant(0.5),
                                                  Initializers.Constant(0.1))

                net.data_layer = Helpers.IrisData(100, random=False)
                net.loss_layer = Loss.CrossEntropyLoss()
                net.append_layer(layer)
                net.append_layer(SoftMax.SoftMax())
                net.train(1)
                current_loss = np.sum(net.loss)
                loss.append(current_loss)

                print(f"Layer: {layer.__class__.__name__}, Regularizer: {reg}")
                print(f"LOSS: {loss}")
                print(f"Regularization Loss (if reg): {current_loss - np.sum(net.loss_layer.loss)}")

            self.assertFalse(np.isclose(loss[0], loss[1], atol=tolerance),
                             "Regularization Loss is not calculated and added to the overall loss "
                             "for " + layer.__class__.__name__)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    tester = test()
    tester.test_regularization_loss()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
