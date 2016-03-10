import theano
import theano.tensor as T
import numpy as np

def memnet(D,V):
    # Inputs
    x = theano.tensor.tensor4('x') # 4d: BATCH_SIZE * N * V * L
    q = theano.tensor.tensor3('q') # 3d: BATCH_SIZE * V * L

    # embedding matrix D * V
    A = theano.shared(
                np.asarray(
                    np.random.uniform(low=-1, high=1, size=(D,V)),
                    dtype=theano.config.floatX
                )
            )
    # These can also be separate embedding matrices initialized just like A
    B = C = W = A

    # get embedded matrices from one hot vectors
    m_4d = T.tensordot(x,A,axes=[[2],[1]]).swapaxes(2,3) # batch*N*D*L
    c_4d = T.tensordot(x,C,axes=[[2],[1]]).swapaxes(2,3) # batch*N*D*L
    u_3d = T.tensordot(q,B,axes=[[1],[1]]).swapaxes(1,2)

    # positional/temporal embedding, not implemented for now

    # get embedding by summing them up along horizontal axis
    m = m_4d.sum(axis=3)
    c = c_4d.sum(axis=3)
    u = u_3d.sum(axis=2)

    # calculate scores and take softmax
    scores = T.batched_dot(m, u)
    smax = T.nnet.softmax(scores)

    # calculate output
    o = T.batched_dot(smax, c)
    result = o + u

    # decode result into probability distribution
    pred_scores = T.dot(result, W)
    pred_smax = T.nnet.softmax(pred_scores)

    return theano.function([x,q], pred_smax, on_unused_input='warn')

def main():
    BATCH_SIZE = 10
    D = 3
    N = 6
    V = 8
    L = 20
    xx = np.ones((BATCH_SIZE,N,V,L))
    qq = np.ones((BATCH_SIZE,V,L))
    net = memnet(D,V)
    print net(xx, qq)

if __name__=="__main__":
    main()
