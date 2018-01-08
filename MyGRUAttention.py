import numpy as np

def sigmoid(x, derivative=False):
    return x*(1-x) if derivative else 1/(1+np.exp(-x))

def softmax(X):
    return np.exp(X) / np.exp(X).sum(axis=0)


h = np.array([[.5],
              [.3],
              [.4]]).reshape(3, 1)
print('h:', h.shape)
h

units = 1
seq_len, ndim = h.shape

s0 = np.array([[.1]]).reshape(units, ndim)
print('s0:', s0.shape)
s0

y0 = np.array([[.2],
               [.1],
               [.3]]).reshape(seq_len, ndim)
print('y0:', y0.shape)
y0

w0 = np.array([[.1],
               [.2],
               [.4]]).reshape(seq_len, input_dim)
print('w0:', w0.shape)
w0

w1 = np.array([[.2]]).reshape(units, units)
print('w1:', w1.shape)
w1

# x.shape : (a, b),
# K.repeat(x, 3).shape : (a, 3, b)
rs = np.repeat(s0, seq_len).reshape(seq_len, ndim)
print('repeated_s0:', rs.shape)
rs

ws = np.dot(rs, w1)
print('ws:', ws.shape)
ws

w2 = np.array([[.3]]).reshape(input_dim, units)
print('w2:', w2.shape)
w2

wh = np.dot(h, w2)
print('wh:', wh.shape)
wh

e = np.tanh(ws + wh).round(2)
print('e:', e.shape)
e

a = softmax(e).reshape(ndim, seq_len)
print('a:', a.shape)
a

c = np.dot(a, h)
print('c:', c)
c

# %%

v = np.array([[.1, .2, .5]]).reshape(ndim, units*3)
print('v(kernel):', v.shape)
v0 = v[:,        : units*1]
v1 = v[:, units*1: units*2]
v2 = v[:, units*2]

print('v0:', v0.shape)
v0

u = np.array([[.4, .2, .2]]).reshape(units, units*3)
print('u(recurrent kernel):', u.shape)
u0 = u[:,        : units*1]
u1 = u[:, units*1: units*2]
u2 = u[:, units*2]

print('u0:', u0.shape)
u0

uu = np.array([[.6, .4, .1]]).reshape(ndim, units*3)
print('uu(recurrent kernel):', uu.shape)
uu0 = uu[:,        : units*1]
uu1 = uu[:, units*1: units*2]
uu2 = uu[:, units*2]

print('uu0:', uu0.shape)
uu0


# %%

h1 = h[0].reshape(-1, ndim)
print('h1:', h1.shape)
h1

z = sigmoid(h1 + np.dot(s0, u0) + np.dot(c, uu0))#.reshape(seq_len, ndim)
print('z:', z.shape)
z

r = sigmoid(h1 + np.dot(s0, u1) + np.dot(c, uu1))#.reshape(seq_len, ndim)
print('r:', r.shape)
r

ss = sigmoid(h1 + np.dot(s0, u2) + np.dot(c, uu2))#.reshape(seq_len, ndim)
print('ss:', ss.shape)
ss

s = z * s0 + (1-z) * ss
print('s:', s.shape)
s

s, [s]
