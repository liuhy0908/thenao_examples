import numpy as np

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
    encoding_matrix = np.zeros(shape=(25, vocab['size']))
    for i, c in enumerate(text):
        encoding_matrix[i,:] = onehot(c, vocab)
    return encoding_matrix

def encode_dataset(x, y, vocab):
    x_vec = np.zeros(shape=(len(x[0]), vocab['size'], len(x)))
    y_vec = np.zeros(shape=(len(x[0]), vocab['size'], len(x)))

    for i in range(len(x)):
        x_vec[:,:,i] = encode(x[i], vocab)
        y_vec[:,:,i] = encode(y[i], vocab)

    return x_vec, y_vec

with open('data.txt') as fl:
    text = fl.read().replace('\n',' ').replace('  ', ' ').replace(';',',').lower()
    data = [text[i:i+26] for i in range(0,len(text), 26)]
    data = [(line[:-1],line[1:]) for line in data]
    x, y = zip(*data)

    vocab = build_vocab(text)
    x_vec, y_vec = encode_dataset(x, y, vocab)
