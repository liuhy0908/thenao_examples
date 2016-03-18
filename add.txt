import numpy as np
import theano
from theano import tensor as t, printing
from theano.tests.breakpoint import PdbBreakpoint
from load_data import x_vec, y_vec, vocab

theano.config.optimizer = 'None'
epsilon = 0.01
gamma = 0.1

batch_size = 32
hidden_size = 4
input_size = output_size = vocab['size'];

#class RNN:
#    def __init__(, hidde   n_size, input_size, output_size):
#W_h = theano.shared(np.zeros((hidden_size, hidden_size)))
#W_x = theano.shared(np.ones((hidden_size, input_size)))
#W_y = theano.shared(np.ones((output_size, hidden_size)))

W_h = theano.shared(np.random.uniform(size=(hidden_size, hidden_size), low=-.001, high=.001))
W_x = theano.shared(np.random.uniform(size=(hidden_size, input_size), low=-.001, high=.001))
W_y = theano.shared(np.random.uniform(size=(output_size, hidden_size), low=-.001, high=.001))

# Add biases
b_h=theano.shared(np.random.uniform(size=(hidden_size, batch_size), low=-.001, high=.001))
b_y=theano.shared(np.random.uniform(size=(output_size, batch_size), low=-.001, high=.001))

# Adagrad parameters
params = [W_h, W_x, W_y, b_h, b_y]
param_shapes = [(hidden_size, hidden_size), (hidden_size, input_size), (output_size, hidden_size), (hidden_size, batch_size), (output_size, batch_size)]
grad_hists = [theano.shared(np.zeros(shape=param_shape)) for param_shape, param in zip(param_shapes, params)]

# Define Inputs
x = t.tensor3()
y = t.tensor3()

h0 = t.matrix()

lr = t.scalar()

# Step function to step through each timestep
def step(x_t, h_t_1, W_h, W_x, W_y):
    # Add breakpoint

    h = t.tanh(theano.dot(W_h, h_t_1) + theano.dot(W_x, x_t) + b_h)
    y = (theano.dot(W_y, h) + b_y)
    e_y = t.exp(y - y.max(axis=0, keepdims=True))
    smax_y = e_y / e_y.sum(axis=0, keepdims=True)
    return h, y

    #def predict(, x_vec):
        # return symbolic output of theano pass
[h, out], _ = theano.scan(step, sequences=x, outputs_info=[h0, None], non_sequences=[W_h, W_x, W_y])

error = ((out - y)**2).sum()
#error = t.nnet.categorical_crossentropy(out, y).sum()

# Implement adagrad and define symbolic updates which is a list of tuples
param_grads = t.grad(error, params)

new_grad_hists = [g_hist + g ** 2 for g_hist, g in zip(grad_hists, param_grads)]

param_updates = [
    (param, param - (gamma * epsilon / (t.sqrt(g_hist) + epsilon)) * param_grad)
    for param, param_grad, g_hist in zip(params, param_grads, grad_hists)
]

grad_hist_update = zip(grad_hists, new_grad_hists)
updates = grad_hist_update + param_updates

# Calculate output and train functions
output = theano.function([h0, x, y], [out, error], on_unused_input='warn')

train = theano.function([h0, x, y], [error, out], updates=updates)


#x_in = np.array([[1,1,1],[2,2,2]])
#y_in = np.cumsum(x_in, axis=1)
#print y_in
#import pdb; pdb.set_trace()
h0_in = np.zeros(shape=(hidden_size, batch_size))
#print output(h0_in, x_in.transpose(), y_in.transpose())

#net = RNN(4,4,4)
#pred = net.predict(x)
#main()

# too much error, implement batching
#
#import pdb; pdb.set_trace()

for i in range(100000):
    idx = np.random.randint(x_vec.shape[2]-batch_size)
    lr = 0.01
    #print lr
    x_in = x_vec[:,:,idx:idx+batch_size]
    y_in = y_vec[:,:,idx:idx+batch_size]
    error, out = train(h0_in, x_in, y_in)
    print (error, i)
    #import pdb; pdb.set_trace()

# Use trained model
