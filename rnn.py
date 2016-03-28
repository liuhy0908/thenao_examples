import numpy as np
import theano
from theano import tensor as t, printing
#from theano.tests.breakpoint import PdbBreakpoint
from load_data import x_vec, y_vec, vocab, matrix_to_text
from theano.gradient import grad_clip

theano.config.optimizer = 'None'
gamma = 0.01

hidden_size1 = 100
hidden_size2 = 120
input_size = output_size = vocab['size']

# def generateSamples(predictFn, x):
#     h0_in = np.zeros(shape=(hidden_size, 1))
#     rez = predictFn(h0, x)
#     import pdb; pdb.set_trace()
#     return rez

#class RNN:
#    def __init__(, hidde   n_size, input_size, output_size):
#W_h1 = theano.shared(np.zeros((hidden_size, hidden_size)))
#W_x = theano.shared(np.ones((hidden_size, input_size)))
#W_y = theano.shared(np.ones((output_size, hidden_size)))

W_h1 = theano.shared(np.random.randn(hidden_size1, hidden_size1)*0.01)
W_h2 = theano.shared(np.random.randn(hidden_size2, hidden_size2)*0.01)
W_h12 = theano.shared(np.random.randn(hidden_size2, hidden_size1)*0.01)

W_x = theano.shared(np.random.randn(hidden_size1, input_size)*0.01)
W_y = theano.shared(np.random.randn(output_size, hidden_size2)*0.01)

# Add biases
b_h1 = theano.shared(np.zeros(shape=(hidden_size1,)))
b_h2 = theano.shared(np.zeros(shape=(hidden_size2,)))
b_y = theano.shared(np.zeros(shape=(output_size,)))

# Adagrad parameters
params = [W_x, W_h1, W_h2, W_h12, W_y, b_h1, b_h2, b_y]
param_shapes = [param.shape for param in params]

# Define Inputs
x = t.matrix()
y = t.matrix()
h0_1 = t.vector()
h0_2 = t.vector()

lr = t.scalar()

# Step function to step through each timestep
def step(x_t, h1_t_1, h2_t_1, W_x, W_h1, W_h2, W_h12, W_y):
    # Add breakpoint

    h1 = t.tanh(theano.dot(W_h1, h1_t_1) + theano.dot(W_x, x_t) + b_h1)
    h2 = t.tanh(theano.dot(W_h12, h1)+theano.dot(W_h2,h2_t_1)+b_h2)
    y = (theano.dot(W_y, h2) + b_y)
    e_y = t.exp(y - y.max())
    smax_y = e_y / e_y.sum()
    return h1, h2, smax_y

    #def predict(, x_vec):
        # return symbolic output of theano pass
[h1, h2, out], _ = theano.scan(step, sequences=x, outputs_info=[h0_1,h0_2,None], 
    non_sequences=[W_x, W_h1, W_h2, W_h12, W_y])


#error = ((out - y)**2).sum()
error = t.nnet.categorical_crossentropy(out, y).sum()

# Implement adagrad and define symbolic updates which is a list of tuples
grads = t.grad(error, params)
#param_grads = grads
param_grads = [grad_clip(grad, -5, 5) for grad in grads]

# new_grad_hists = [g_hist + g ** 2 for g_hist, g in zip(grad_hists, param_grads)]

# param_updates = [
#     (param, param - gamma * param_grad/t.sqrt(g_hist + 1e-8))
#     for param, param_grad, g_hist in zip(params, param_grads, grad_hists)
# ]

#Iplemening gradient clipping here
param_updates = [
    (param, param - 0.01 * param_grad)
    for param, param_grad in zip(params, param_grads)
]
updates = param_updates

#grad_hist_update = zip(grad_hists, new_grad_hists)
#updates = grad_hist_update + param_updates

# Calculate output and train functions
output = theano.function([h0_1, h0_2, x], [out,h1,h2], on_unused_input='warn')

train = theano.function([h0_1, h0_2, x, y], [error, out, h1[-1],h2[-1]], updates=updates)


#x_in = np.array([[1,1,1],[2,2,2]])
#y_in = np.cumsum(x_in, axis=1)
#print y_in
#import pdb; pdb.set_trace()
h0_1in = np.zeros(shape=(hidden_size1,))
h0_2in = np.zeros(shape=(hidden_size2,))
#print output(h0_in, x_in.transpose(), y_in.transpose())

#net = RNN(4,4,4)
#pred = net.predict(x)
#main()

def sample(h_initial1,h_initial2,x_seed,k):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  gen_seq=[x_seed]
  for i in range(k):
    gen_prob,hidden1,hidden2 = output(h_initial1,h_initial2,x_seed)
    h_initial1 = np.reshape(hidden1,(hidden_size1))
    h_initial2 = np.reshape(hidden2,(hidden_size2))
    char_ix = np.random.choice(range(vocab['size']), p=gen_prob.ravel())
    x_seed = np.zeros((1,vocab['size']))
    x_seed[0,char_ix]=1
    gen_seq.append(x_seed)
  return gen_seq

# too much error, implement batching
#
#import pdb; pdb.set_trace()
idx = 0

for i in range(100000):
    #idx = np.random.randint(x_vec.shape[2])
    lr = 0.01
    #print lr
    x_in = x_vec[:,:,idx]
    y_in = y_vec[:,:,idx]
    error, out, h0_1in, h0_2in = train(h0_1in, h0_2in, x_in, y_in)
    #import pdb; pdb.set_trace()
    if i%100==0:
        # Also print out the input sentence here
        print (error, i, idx)

        hprev1 = np.zeros((hidden_size1,))
        hprev2 = np.zeros((hidden_size2,))
        k = np.random.randint(x_vec.shape[2])
        sample_ix = sample(hprev1,hprev2,np.reshape(x_vec[0,:,k],(1,vocab['size'])), 500)
        txt = ''.join(vocab['decoder'][np.argmax(ix)] for ix in sample_ix)
        print '----\n %s \n----' % (txt, )

    #import pdb; pdb.set_trace()
    #in1 = x_in[:,:,0:1]
    #generateSamples(output, in1)
    #import pdb; pdb.set_trace()
    # sweep from left ti right. otherwise this will be pretty random
    idx = idx + 1
    if idx == x_vec.shape[2]:
        idx = 0
        h0_1in = np.zeros(shape=(hidden_size1,))
        h0_2in = np.zeros(shape=(hidden_size2,))
# Use trained model
