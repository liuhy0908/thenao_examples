import theano
import theano.tensor as t
import numpy as np

theano.config.optimizer = 'None'

# symbolic graph
x = t.tensor3('x')
y = t.matrix('y')

def step(x_t, h_t_1):
    h = h_t_1 + x_t
    y = h
    return h, y

[h, out], _ = theano.scan(step, sequences=x, outputs_info=[t.zeros_like(y), None])

answer = theano.function([x, y], out)

# inputs
batches = 2
vlen = 3
timesteps = 8
x_in = np.ones(shape=(timesteps, vlen, batches))
y_in = x_in.sum(axis=0)

#print x_in.shape, y_in.shape

print(answer(x_in, y_in))
