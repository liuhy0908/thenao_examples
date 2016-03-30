import theano
import numpy as np
from theano import tensor as tt
from blocks import initialization
from blocks.bricks import Identity
from blocks.bricks import Linear
from blocks.bricks.recurrent import BaseRecurrent, SimpleRecurrent

class MyRnn(BaseRecurrent): # Extend the base recurrent class to create one of your own
  def __init__(self, dim, **kwargs):
    super(MyRnn, self).__init__(**kwargs)
    self.dim = dim
    self.layer1 = SimpleRecurrent(dim=self.dim, activation=Identity(), name='recurrent layer 1', weights_init=initialization.Identity())
    self.layer2 = SimpleRecurrent(dim=self.dim, activation=Identity(), name='recurrent layer 2', weights_init=initialization.Identity())
    self.children = [self.layer1, self.layer2]

  def apply(self, inputs, first_states=None, second_states=None):
    first_h = self.layer1.apply(inputs=inputs, states=first_states, iterate=False)
    second_h = self.layer2.apply(inputs=first_h, states=second_states, iterate=False)
    return first_h, second_h

  def get_dim(self):
    pass

x = tt.matrix()
import pdb; pdb.set_trace()
h0 = tt.vector()

rnn = MyRnn(dim=3)
rnn.initialize()
fh, sh = rnn.apply(inputs=x)

f = theano.function([x], [first_h, second_h])

print f(np.ones((10, 3)))

