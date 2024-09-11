import numpy as np
import sklearn.metrics as met
import matplotlib.pyplot as plt
import sklearn.preprocessing as pp
import sklearn.model_selection as ms

np.random.seed(0)
plt.style.use('ggplot')

def relu(Z:np.ndarray):
    return (Z + np.abs(Z)) / 2

def d_relu(z:float):
    return {True:1, False:0}[z>0]

def leakyrelu(Z:np.ndarray):
    return (1.01*Z + np.abs(0.99*Z)) / 2

def d_leakyrelu(z:float):
    return {True:1, False:0.01}[z>0]

def sigmoid(Z:np.ndarray):
    return 1/(1 + np.exp(-Z))

def d_sigmoid(z:float):
    return z*(1 - z)

class Model1:
    def __init__(self, Name:str, nH:int):
        self.Name = Name
        self.nH = nH
    def _model(self,
                W1:np.ndarray,
                B1:np.ndarray,
                W2:np.ndarray,
                B2:np.ndarray,
                X:np.ndarray):
        Z1 = np.dot(X, W1) + B1
        O1 = leakyrelu(Z1)
        Z2 = np.dot(O1, W2) + B2
        O2 = sigmoid(Z2)
        return O1, O2
    def predict(self, X:np.ndarray):
        return self._model(self.W1, self.B1, self.W2, self.B2, X)
    def create_dataset(self, nD:int, Ms:np.ndarray, s:float):
        self.Ms = Ms
        self.s = s
        self.nC = Ms.shape[0]
        self.nX = Ms.shape[1]
        self.nY = self.nC
        self.X = np.zeros((nD, self.nX))
        self.Y = np.zeros((nD, 1))
        for i in range(nD):
            c = np.random.randint(0, self.nC)
            self.Y[i, 0] = c
            self.X[i, :] = self.Ms[c] + self.s*np.random.randn(self.nX)
        self.OHE = pp.OneHotEncoder()
        self.OHY = self.OHE.fit_transform(self.Y).toarray()
    def plot_dataset(self):
        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y[:, 0], s=20)
        plt.scatter(self.Ms[:, 0], self.Ms[:, 1], c='r', s=120, marker='*', label='Center')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()
    def _initialize_parameters(self, r:float=1):
        self.W1 = np.random.uniform(-r, +r, (self.nX, self.nH))
        self.B1 = np.random.uniform(-r, +r, (self.nH))
        self.W2 = np.random.uniform(-r, +r, (self.nH, self.nY))
        self.B2 = np.random.uniform(-r, +r, (self.nY))
    def fit(self, nEpoch:int, lr:float=1e-2, MaxAcc:float=None):
        self._initialize_parameters(1)
        for I in range(nEpoch):
            for x, y in zip(self.X, self.OHY):
                # Updating W(I >> H)
                for i in range(self.nX):
                    for j in range(self.nH):
                        o1, o2 = self._model(self.W1, self.B1, self.W2, self.B2, x)
                        delta_j = 0
                        for k in range(self.nY):
                            delta_j += self.W2[j, k] * d_sigmoid(o2[k]) * (y[k] - o2[k])
                        delta_j *= d_leakyrelu(o1[j])
                        self.W1[i, j] += lr * delta_j * x[i]
                # Updating B(H)
                for j in range(self.nH):
                    o1, o2 = self._model(self.W1, self.B1, self.W2, self.B2, x)
                    delta_j = 0
                    for k in range(self.nY):
                        delta_j += self.W2[j, k] * d_sigmoid(o2[k]) * (y[k] - o2[k])
                    delta_j *= d_leakyrelu(o1[j])
                    self.B1[j] += lr * delta_j
                # Updating W(H >> O)
                for j in range(self.nH):
                    for k in range(self.nY):
                        o1, o2 = self._model(self.W1, self.B1, self.W2, self.B2, x)
                        delta_k = d_sigmoid(o2[k]) * (y[k] - o2[k])
                        self.W2[j, k] += lr * delta_k * o1[j]
                # Updating B(O)
                for k in range(self.nY):
                    _, o2 = self._model(self.W1, self.B1, self.W2, self.B2, x)
                    delta_k = d_sigmoid(o2[k]) * (y[k] - o2[k])
                    self.B2[k] += lr * delta_k
            _, Od = self._model(self.W1, self.B1, self.W2, self.B2, self.X)
            Od2 = np.zeros((Od.shape[0], 1))
            for i in range(Od.shape[0]):
                Od2[i, 0] = np.argmax(Od[i])
            Ac = met.accuracy_score(self.Y, Od2)
            X1s = np.linspace(0, 1, num=21)
            X2s = np.linspace(0, 1, num=21)
            mX1s, mX2s = np.meshgrid(X1s, X2s)
            Points = np.c_[mX1s.ravel(), mX2s.ravel()]
            _, go = self._model(self.W1, self.B1, self.W2, self.B2, Points)
            go2 = np.zeros(go.shape[0])
            for i in range(go.shape[0]):
                go2[i] = np.argmax(go[i])
            go2 = go2.reshape(mX1s.shape)
            plt.cla()
            plt.contourf(mX1s, mX2s, go2, cmap='Spectral')
            plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y[:, 0], s=20)
            plt.scatter(self.Ms[:, 0], self.Ms[:, 1], c='r', s=120, marker='*', label='Center')
            plt.text(0.1, 0.9, f'Epoch: {I+1}', fontdict={'size': 15})
            plt.text(0.1, 0.75, f'Accuracy: {round(100*Ac, 4)} %', fontdict={'size': 15})
            plt.xlabel('X1')
            plt.ylabel('X2')
            plt.legend()
            plt.pause(0.05)
            print(f'Epoch: {I+1} -- Accuracy: {round(100*Ac, 4)} %')
            if MaxAcc is not None:
                if Ac>=MaxAcc:
                    print('Model Learning Stopped.\nReched Max Accuracy.')
                    break
        if I==nEpoch-1:
            print('Model Learning Stopped.\nReched Max Epoch.')
    def classification_report(self):
        _, Od = self.predict(self.X)
        Od2 = np.zeros((Od.shape[0], 1))
        for i in range(Od.shape[0]):
            Od2[i, 0] = np.argmax(Od[i])
        CR = met.classification_report(self.Y, Od2)
        print(f'Classification Report For {self.Name}:\n{CR}')

Object = Model1('First DLP With OOP', 12)

Ms = np.array([[0.22, 0.35],
                [0.8, 0.6],
                [0.55, 0.45],
                [0.3, 0.7],
                [0.8, 0.25]])

Object.create_dataset(750, Ms, 0.05)

Object.plot_dataset()

Object.fit(100, MaxAcc=0.97)

Object.classification_report()