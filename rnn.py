import numpy as np
import theano

class RNN:
    def __init__(self, hidden_size, input_size, output_size):
        self.W_h = theano.shared(np.random.uniform(size=(hidden_size, hidden_size), low=-.01, high=.01))
        self.W_x = theano.shared(np.random.uniform(size=(hidden_size, input_size), low=-.01, high=.01))
        self.W_y = theano.shared(np.random.uniform(size=(output_size, hidden_size), low=-.01, high=.01))
        self.h = 0
        self.b = 0

    def step(self, x):
        self.h = se

    def predict(self, x_vec):
        # return symbolic output of theano pass
        pass

    def train(self, x_vec, y_vec)


def main()
    x = [[1,2,3,4,5],[6,7,8,9,10],[11,12,13,14,15]]
