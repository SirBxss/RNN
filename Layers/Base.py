class BaseLayer:
    def __init__(self):
        self.trainable = False
        self.weights = None
        self.testing_phase = False
#         self.phase = 'train'
#
#
# class Phase:
#     train = 'train'
#     test = 'test'
