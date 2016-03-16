import numpy as np
import theano
from theano import tensor as t, printing
from theano.tests.breakpoint import PdbBreakpoint
from load_data import x_vec, y_vec, vocab

theano.config.optimizer = 'None'

hidden_size = input_size = output_size = vocab['size'];

#class RNN:
#    def __init__(, hidde   n_size, input_size, output_size):
#W_h = theano.shared(np.zeros((hidden_size, hidden_size)))
#W_x = theano.shared(np.ones((hidden_size, input_size)))
#W_y = theano.shared(np.ones((output_size, hidden_size)))
W_h = theano.shared(np.random.uniform(size=(hidden_size, hidden_size), low=-.01, high=.01))
W_x = theano.shared(np.random.uniform(size=(hidden_size, input_size), low=-.01, high=.01))
W_y = theano.shared(np.random.uniform(size=(output_size, hidden_size), low=-.01, high=.01))
b_h=theano.shared(np.random.uniform(size=(hidden_size, 1), low=-.01, high=.01))
b_y=theano.shared(np.random.uniform(size=(output_size, 1), low=-.01, high=.01))
# Define Inputs
x = t.matrix()
y = t.matrix()

h0 = t.vector()

lr = t.scalar()

def step(x_t, h_t_1, W_h, W_x, W_y,b_h,b_y):
    # Add breakpoint

    h = t.tanh(theano.dot(W_h, h_t_1) + theano.dot(W_x, x_t)+b_h)
    y = theano.dot(W_y, h)+b_y
    return h, y

    #def predict(, x_vec):
        # return symbolic output of theano pass
[h, out], _ = theano.scan(step, sequences=x, outputs_info=[h0, None], non_sequences=[W_h, W_x, W_y,b_h,b_y])

error = ((out - y)**2).sum()

gW_h, gW_x, gW_y, gb_h, gb_y = t.grad(error, [W_h, W_x, W_y, b_h, b_y])

output = theano.function([h0, x, y], [out, error], on_unused_input='warn')

train = theano.function([h0, x, y, lr], [error], updates={W_h: W_h - lr * gW_h, W_x: W_x - lr * gW_x,
    W_y: W_y - lr * gW_y, b_h: b_h-lr*gb_h, b_y: b_y-lr*gb_y})

#x_in = np.array([[1,1,1],[2,2,2]])
#y_in = np.cumsum(x_in, axis=1)
#print y_in
#import pdb; pdb.set_trace()
h0_in = np.zeros(vocab['size'])
#print output(h0_in, x_in.transpose(), y_in.transpose())

#net = RNN(4,4,4)
#pred = net.predict(x)
#main()

# too much error, implement batching
#

for i in range(5000):
    idx = np.random.randint(len(x_vec)-3)
    lr = 0.01
    #print lr
    print(train(h0_in, x_vec[idx:idx+3].transpose(), y_vec[idx:idx+3].transpose(), lr))

# Use trained model
