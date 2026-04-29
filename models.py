import numpy as np

# Permet de tranformer les labels en vecteur ( car il prédit une matrice p avec la probabilité pour chaque valeur)
# Nécéssaire pour la cross entropie
def one_hot(y, num_classes=10):
    Y = np.zeros((y.shape[0], num_classes))
    Y[np.arange(y.shape[0]), y] = 1
    return Y

# Transforme les scores en probabilité
# Axis: sur quelle dimension on travaille
# Keepdims : Permet de garder la forme
def softmax(Z):
    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    exp_Z = np.exp(Z_stable)
    return exp_Z / np.sum(exp_Z, axis=1, keepdims=True)

# Mesurer l'erreur: a quel point le modèle est confiant sur la bonne réponse
def cross_entropy(Y, P):
    epsilon = 1e-15
    P = np.clip(P, epsilon, 1 - epsilon)
    return -np.mean(np.sum(Y * np.log(P), axis=1))

# Mesure le taux de bonne réponse
def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


class ModeleLineaire:
    """
    Partie 1.2.1 : modèle linéaire multi-classe

    Formule :
        o = X A + b

    Puis :
        P = softmax(o)
    """

    def __init__(self, input_dim=784, output_dim=10):
        self.A = 0.01 * np.random.randn(input_dim, output_dim)
        self.b = np.zeros((1, output_dim))

    def forward(self, X):
        self.X = X
        self.logits = X @ self.A + self.b
        self.P = softmax(self.logits)
        return self.P

    def backward(self, Y):
        n = self.X.shape[0]

        # Gradient de la cross-entropy combinée au softmax
        dZ = (self.P - Y) / n

        # Gradients des paramètres
        self.dA = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0, keepdims=True)

    def update(self, learning_rate):
        self.A -= learning_rate * self.dA
        self.b -= learning_rate * self.db

    def predict(self, X):
        P = softmax(X @ self.A + self.b)
        return np.argmax(P, axis=1)


class ModeleUneCoucheCachee:
    """
    Architecture :
        X -> couche cachée -> ReLU -> couche sortie -> softmax

    Formules :
        Z1 = X W1 + b1
        H = ReLU(Z1)
        Z2 = H W2 + b2
        P = softmax(Z2)
    """

    def __init__(self, input_dim=784, hidden_dim=64, output_dim=10):
        self.W1 = np.sqrt(2 / input_dim) * np.random.randn(input_dim, hidden_dim)
        self.b1 = np.zeros((1, hidden_dim))

        self.W2 = np.sqrt(2 / hidden_dim) * np.random.randn(hidden_dim, output_dim)
        self.b2 = np.zeros((1, output_dim))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def forward(self, X):
        self.X = X

        self.Z1 = X @ self.W1 + self.b1
        self.H = self.relu(self.Z1)

        self.Z2 = self.H @ self.W2 + self.b2
        self.P = softmax(self.Z2)

        return self.P

    def backward(self, Y):
        n = self.X.shape[0]

        # Gradient sortie softmax + cross-entropy
        dZ2 = (self.P - Y) / n

        # Gradients couche de sortie
        self.dW2 = self.H.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        # Rétropropagation vers la couche cachée
        dH = dZ2 @ self.W2.T
        dZ1 = dH * self.relu_derivative(self.Z1)

        # Gradients couche cachée
        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

    def predict(self, X):
        Z1 = X @ self.W1 + self.b1
        H = self.relu(Z1)

        Z2 = H @ self.W2 + self.b2
        P = softmax(Z2)

        return np.argmax(P, axis=1)

class ModeleDeuxCouchesCachees:
    def __init__(self, input_dim=784, hidden1=64, hidden2=32, output_dim=10):

        self.W1 = np.sqrt(2 / input_dim) * np.random.randn(input_dim, hidden1)
        self.b1 = np.zeros((1, hidden1))

        self.W2 = np.sqrt(2 / hidden1) * np.random.randn(hidden1, hidden2)
        self.b2 = np.zeros((1, hidden2))

        self.W3 = np.sqrt(2 / hidden2) * np.random.randn(hidden2, output_dim)
        self.b3 = np.zeros((1, output_dim))

    def relu(self, Z):
        return np.maximum(0, Z)

    def relu_derivative(self, Z):
        return (Z > 0).astype(float)

    def forward(self, X):
        self.X = X

        self.Z1 = X @ self.W1 + self.b1
        self.H1 = self.relu(self.Z1)

        self.Z2 = self.H1 @ self.W2 + self.b2
        self.H2 = self.relu(self.Z2)

        self.Z3 = self.H2 @ self.W3 + self.b3
        self.P = softmax(self.Z3)

        return self.P

    def backward(self, Y):
        n = self.X.shape[0]

        # sortie
        dZ3 = (self.P - Y) / n
        self.dW3 = self.H2.T @ dZ3
        self.db3 = np.sum(dZ3, axis=0, keepdims=True)

        # couche 2
        dH2 = dZ3 @ self.W3.T
        dZ2 = dH2 * self.relu_derivative(self.Z2)
        self.dW2 = self.H1.T @ dZ2
        self.db2 = np.sum(dZ2, axis=0, keepdims=True)

        # couche 1
        dH1 = dZ2 @ self.W2.T
        dZ1 = dH1 * self.relu_derivative(self.Z1)
        self.dW1 = self.X.T @ dZ1
        self.db1 = np.sum(dZ1, axis=0, keepdims=True)

    def update(self, learning_rate):
        self.W1 -= learning_rate * self.dW1
        self.b1 -= learning_rate * self.db1

        self.W2 -= learning_rate * self.dW2
        self.b2 -= learning_rate * self.db2

        self.W3 -= learning_rate * self.dW3
        self.b3 -= learning_rate * self.db3

    def predict(self, X):
        Z1 = X @ self.W1 + self.b1
        H1 = self.relu(Z1)

        Z2 = H1 @ self.W2 + self.b2
        H2 = self.relu(Z2)

        Z3 = H2 @ self.W3 + self.b3
        P = softmax(Z3)

        return np.argmax(P, axis=1)