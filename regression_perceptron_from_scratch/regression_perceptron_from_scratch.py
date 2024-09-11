import numpy as np
import sklearn.metrics as met
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.model_selection as ms

np.random.seed(0)
plt.style.use('ggplot')

def SLP (W:np.ndarray, B:np.ndarray, X:np.ndarray):
    Z = np.dot(X, W) + B
    #O = 1/(1 + np.exp(-Z))
    O = Z
    return O

nD = 800
nX0 = 2
nY = 1
r = 0.05

X0 = np.random.uniform(0, 1, (nD, nX0))
Y = np.zeros((nD, nY))

for i in range(nD):
    e = np.random.uniform(-r, +r)
    Y[i, 0] = 1.2*X0[i, 0]**1.3 - 0.7*X0[i, 1]**0.7 - 1.6*X0[i, 0]*X0[i, 1] - 0.1 + e

for i in range(nX0):
    plt.scatter(X0[:, i], Y[:, 0], s=20)
    plt.xlabel(f'X{i+1}')
    plt.ylabel('Y')
    plt.show()

PF = pp.PolynomialFeatures(degree=3, include_bias=False)
X = PF.fit_transform(X0)

nX = X.shape[1]

trX, teX, trY, teY = ms.train_test_split(X, Y, train_size=0.7, random_state=0, shuffle=True)

W = np.random.uniform(-1, +1, (nX, nY))
B = np.random.uniform(-1, +1, (nY))

nEpoch = 300
lr = 1e-2

for I in range(nEpoch):
    for x, y in zip(trX, trY):
        for i in range(nX):
            for j in range(nY):
                o = SLP(W, B, x)
                W[i, j] += lr * x[i] * (y[j] - o[j]) * 1
        for j in range(nY):
            o = SLP(W, B, x)
            B[j] += lr * (y[j] - o[j]) * 1
    print(f'Epoch {I+1} Ended.')
    
    trO = SLP(W, B, trX)
    teO = SLP(W, B, teX)
    trR2 = met.r2_score(trY, trO)
    teR2 = met.r2_score(teY, teO)
    a = min(np.min(trY), np.min(trO))
    b = max(np.max(trY), np.max(trO))
    plt.cla()
    plt.scatter(trY[:, 0], trO[:, 0], s=20, c='b', label='Train')
    plt.scatter(teY[:, 0], teO[:, 0], s=20, c='g', label='Test')
    plt.plot([a, b], [a, b], lw=1.2, c='r', label='y=x')
    plt.text(0.2, 0.85, f'Epoch: {I+1}', fontdict={'size': 15})
    plt.text(0.2, 0.7, f'Train R2: {round(trR2, 4)}', fontdict={'size': 15})
    plt.text(0.2, 0.55, f'Test  R2: {round(teR2, 4)}', fontdict={'size': 15})
    plt.xlabel('Target Values')
    plt.ylabel('Predicted Values')
    plt.legend()
    plt.pause(0.1)