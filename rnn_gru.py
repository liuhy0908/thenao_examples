import numpy as np
import theano as th
from theano import tensor as t, printing
#from th.tests.breakpoint import PdbBreakpoint
from load_data import x_vec, y_vec, vocab, matrix_to_text
from theano.gradient import grad_clip

#th.config.optimizer = 'None'
gamma = 0.01

hidden_size = 512
input_size = output_size = vocab['size'];

# Reset gate parameters, this should output a hidden_size vector
U_r = th.shared(np.random.randn(hidden_size, hidden_size)*0.01)
W_r = th.shared(np.random.randn(hidden_size, input_size)*0.01)

# Update gate parameters, should output a hidden size vector z used for interpolation
U_z = th.shared(np.random.randn(hidden_size, hidden_size)*0.01)
W_z = th.shared(np.random.randn(hidden_size, input_size)*0.01)

# Candidate activation parameters
U_d = th.shared(np.random.randn(hidden_size, hidden_size)*0.01)
W_d = th.shared(np.random.randn(hidden_size, input_size)*0.01)

# Output weights
Y = th.shared(np.random.randn(output_size, hidden_size)*0.01)

# Comment out biases for now
#b_h = th.shared(np.zeros(shape=(hidden_size,)))
b_y = th.shared(np.zeros(shape=(output_size,)))

# Adagrad parameters
params = [W_r, U_r, W_z, U_z, W_d, U_d, Y, b_y]
param_shapes = [param.shape for param in params]

# Define Inputs
x = t.matrix()
y = t.matrix()
h0 = t.vector()

lr = t.scalar()

def softmax(y):
    e_y = t.exp(y - y.max())
    smax_y = e_y / e_y.sum()
    return smax_y

# Step function to step through each timestep
def gru_step(x_t, h_t_1):
    # Calculate reset gate vector
    r = t.nnet.sigmoid(th.dot(W_r, x_t) + th.dot(U_r, h_t_1))
    # Calculate z gate vector
    z = t.nnet.sigmoid(th.dot(W_z, x_t) + th.dot(U_z, h_t_1))
    # Calculate candidate activation vector to add to the previous value
    h_d = t.tanh(th.dot(W_d, x_t) + th.dot(U_d, r * h_t_1))
    # Calculate final hidden state value
    h = (1.0 - z) * h_t_1 + z * h_d
    y = (th.dot(Y, h) + b_y)
    return h, softmax(y)

    #def predict(, x_vec):
        # return symbolic output of th pass
[h, out], _ = th.scan(gru_step, sequences=x, outputs_info=[h0, None])

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
output = th.function([h0, x], [out,h], on_unused_input='warn')

train = th.function([h0, x, y], [error, out, h[-1]], updates=updates)


#x_in = np.array([[1,1,1],[2,2,2]])
#y_in = np.cumsum(x_in, axis=1)
#print y_in
#import pdb; pdb.set_trace()
h0_in = np.zeros(shape=(hidden_size,))
#print output(h0_in, x_in.transpose(), y_in.transpose())

#net = RNN(4,4,4)
#pred = net.predict(x)
#main()

def sample(h_initial,x_seed,k):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  gen_seq=[x_seed]
  for i in range(k):
    gen_prob,hidden= output(h_initial,x_seed)
    h_initial = np.reshape(hidden,(hidden_size))
    char_ix = np.random.choice(range(vocab['size']), p=gen_prob.ravel())
    x_seed = np.zeros((1,vocab['size']))
    x_seed[0,char_ix]=1
    gen_seq.append(x_seed)
  return gen_seq

def print_samples():
    # Run test code
    hprev = np.zeros((hidden_size,))
    k = np.random.randint(x_vec.shape[2])
    sample_ix = sample(hprev,np.reshape(x_vec[0,:,k],(1,vocab['size'])), 500)
    txt = ''.join(vocab['decoder'][np.argmax(ix)] for ix in sample_ix)
    print '----\n %s \n----' % (txt, )

# too much error, implement batching
#
#import pdb; pdb.set_trace()
min_error = 1000
idx = 0

for i in range(100000):
    #idx = np.random.randint(x_vec.shape[2])
    lr = 0.01
    #print lr
    x_in = x_vec[:,:,idx]
    y_in = y_vec[:,:,idx]
    error, out, h0_in = train(h0_in, x_in, y_in)

    if min_error > error:
        min_error = error
        print (error, i, idx, h0_in.sum())
        print_samples()

    #import pdb; pdb.set_trace()
    if i%100==0:
        # Also print out the input sentence here
        print (error, i, idx, h0_in.sum())


    #import pdb; pdb.set_trace()
    #in1 = x_in[:,:,0:1]
    #generateSamples(output, in1)
    #import pdb; pdb.set_trace()
    # sweep from left ti right. otherwise this will be pretty random
    idx = idx + 1
    if idx == x_vec.shape[2]:
        idx = 0
        h0_in = np.zeros(shape=(hidden_size,))
# Use trained model
