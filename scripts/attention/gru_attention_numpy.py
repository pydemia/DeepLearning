import numpy as np

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def softmax(X):
    return np.exp(X) / np.exp(X).sum(axis=0)


def MyGRUAttention():

    h = np.array([[.5],
                  [.3],
                  [.4]]).reshape(3, 1)

    units = 1
    seq_len, ndim = h.shape

    s0 = np.array([[0.]]).reshape(units, ndim)

    y0 = np.array([[.2],
                   [.1],
                   [.3]]).reshape(seq_len, ndim)

    w0 = np.array([[.1],
                   [.7],
                   [.4]]).reshape(seq_len, ndim)
    w1 = np.array([[.8]]).reshape(units, units)
    w2 = np.array([[.5]]).reshape(ndim, units)

    # %%

    v = np.array([[.1, .9, .5]]).reshape(ndim, units*3)
    v0 = v[:,        : units*1]
    v1 = v[:, units*1: units*2]
    v2 = v[:, units*2]

    u = np.array([[.4, .6, .2]]).reshape(units, units*3)
    u0 = u[:,        : units*1]
    u1 = u[:, units*1: units*2]
    u2 = u[:, units*2]

    uu = np.array([[.6, .4, .1]]).reshape(ndim, units*3)
    uu0 = uu[:,        : units*1]
    uu1 = uu[:, units*1: units*2]
    uu2 = uu[:, units*2]

    # %%
    outputs = []
    for seq in range(len(h)):

        # Attention Part

        # x.shape : (a, b),
        # K.repeat(x, 3).shape : (a, 3, b)
        rs = np.repeat(s0, seq_len).reshape(seq_len, ndim)
        ws = np.dot(rs, w1)
        wh = np.dot(h, w2)

        e = np.tanh(ws + wh)
        a = softmax(e).reshape(ndim, seq_len)

        c = np.dot(a, h)
        print('s0: %s\t c: %s' % (s0, c))

        # GRU Part

        h1 = h[seq].reshape(-1, ndim)
        z = sigmoid(h1 + np.dot(s0, u0) + np.dot(c, uu0))
        r = sigmoid(h1 + np.dot(s0, u1) + np.dot(c, uu1))
        ss = sigmoid(h1 + np.dot(r*s0, u2) + np.dot(c, uu2))

        s = z * s0 + (1-z) * ss

        s0 = s

        s, [s]

        last_output = s
        outputs += [s]
        states = [s]

    outputs = np.stack(outputs, axis=1)

    return last_output, outputs, states
