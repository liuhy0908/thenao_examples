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

out, _ = theano.scan(step, sequences=x, outputs_info=[t.zeros_like(y), None])

answer = theano.function([x, y], out)

# inputs
x_in =        [np.array([[1,3,5,11],
               [2,4,6,12]]),
              np.array([[3,5,7,9],
               [4,6,8,10]])]

y_in = [np.array([20,24]),np.array([24,28])]
#print x_in.shape, y_in.shape

print(answer(x_in, y_in))
