import numpy as np

batch_size = 25

def vector_to_char(vect, vocab):
    char = vocab['decoder'][np.where(vect==1)[0][0]]
    return char

def matrix_to_text(matr, vocab):
    string = ''
    for vect in matr:
        string += vector_to_char(vect, vocab)
    return string

def build_vocab(text):
    chars = list(set(text))
    char2indx = {ch:i for i,ch in enumerate(chars)}
    indx2char = {i:ch for i,ch in enumerate(chars)}
    size = len(char2indx)
    return {'encoder': char2indx, 'decoder': indx2char, 'size': size}

def onehot(char, vocab):
    vector = np.zeros(vocab['size'])
    vector[vocab['encoder'][char]] = 1
    return vector

def encode(text, vocab):
    # matrix with concatenation of column vectors
    arr = np.array(map(lambda c: onehot(c, vocab), text))
    encoding_matrix = np.zeros(shape=(batch_size, vocab['size']))
    for i, c in enumerate(text):
        encoding_matrix[i,:] = onehot(c, vocab)
        vector_to_char(encoding_matrix[i,:], vocab)
    return encoding_matrix

def encode_dataset(x, y, vocab):
    x_vec = np.zeros(shape=(len(x[0]), vocab['size'], len(x)))
    y_vec = np.zeros(shape=(len(x[0]), vocab['size'], len(x)))

    for i in range(len(x)):
        x_vec[:,:,i] = encode(x[i], vocab)
        y_vec[:,:,i] = encode(y[i], vocab)
    return x_vec, y_vec

with open('data/data.txt') as fl:
    text = fl.read()
    data = [(text[i-1:i+batch_size-1], text[i:i+batch_size]) for i in range(1,len(text), batch_size)]
    #data = [(line[:-1],line[1:]) for line in data]
    x, y = zip(*data)
    vocab = build_vocab(text)
    x_vec, y_vec = encode_dataset(x, y, vocab)
