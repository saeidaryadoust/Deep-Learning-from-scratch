import numpy as np
import sklearn.metrics as met
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.model_selection as ms

np.random.seed(0)
plt.style.use('ggplot')

def SLP (W:np.ndarray, B:np.ndarray, X:np.ndarray):
    Z = np.dot(X, W) + B
    O = 1/(1 + np.exp(-Z))
    return O

s = 0.05
Ms = np.array([[0.3, 0.35], [0.5, 0.7], [0.35, 0.6], [0.65, 0.35]])
nC = Ms.shape[0]
nX = Ms.shape[1]

nD = 800
nY = nC

X = np.zeros((nD, nX))
Y = np.zeros((nD, 1))

for i in range(nD):
    c = np.random.randint(0, nC)
    Y[i, 0] = c
    X[i, :] = Ms[c] + s*np.random.randn(nX)

OHE = pp.OneHotEncoder()
OHY = OHE.fit_transform(Y).toarray()

plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=20)
plt.xlabel('X1')
plt.ylabel('X2')
plt.show()

W = np.random.uniform(-1, +1, (nX, nY))
B = np.random.uniform(-1, +1, (nY))

nEpoch = 300
lr = 1e-2

for I in range(nEpoch):
    for x, y in zip(X, OHY):
        for i in range(nX):
            for j in range(nY):
                o = SLP(W, B, x)
                W[i, j] += lr * x[i] * (y[j] - o[j]) * (o[j] * (1 - o[j]))
        for j in range(nY):
            o = SLP(W, B, x)
            B[j] += lr * (y[j] - o[j]) * (o[j] * (1 - o[j]))
    print(f'Epoch {I+1} Ended.')
    
    Od = SLP(W, B, X)
    Od2 = np.zeros((Od.shape[0], 1))
    for i in range(Od.shape[0]):
        Od2[i, 0] = np.argmax(Od[i])
    Ac = met.accuracy_score(Y, Od2)
    

    X1s = np.linspace(0, 1, num=81)
    X2s = np.linspace(0, 1, num=81)
    mX1s, mX2s = np.meshgrid(X1s, X2s)
    Points = np.c_[mX1s.ravel(), mX2s.ravel()]
    O = SLP(W, B, Points)
    O2 = np.zeros(O.shape[0])
    for i in range(O.shape[0]):
        O2[i] = np.argmax(O[i])
    O2 = O2.reshape(mX1s.shape)
    plt.cla()
    plt.contourf(mX1s, mX2s, O2, cmap='Spectral')
    plt.scatter(X[:, 0], X[:, 1], c=Y[:, 0], s=20)
    plt.text(0.1, 0.9, f'Epoch: {I+1}', fontdict={'size': 15})
    plt.text(0.1, 0.75, f'Accuracy: {round(100*Ac, 4)} %', fontdict={'size': 15})
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.pause(0.1)

    if Ac>=0.999:
        break
