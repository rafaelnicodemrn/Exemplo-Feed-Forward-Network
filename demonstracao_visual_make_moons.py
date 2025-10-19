# --- Demonstração Visual das Fronteiras de Decisão ---
#
# Este script utiliza as classes 'Perceptron' e 'MLP_FFN' implementadas
# "do zero" (from scratch) para testá-las em um conjunto de dados
# não-linear (make_moons).
#
# As bibliotecas scikit-learn e matplotlib são usadas APENAS para
# gerar os dados e plotar os gráficos, respectivamente.
# O treinamento e a predição são feitos pelas nossas classes.
#
# -----------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# -----------------------------------------------------------------
# CLASSE 1: PERCEPTRON (Copiada do repositório 'Exemplo-Perceptron')
# -----------------------------------------------------------------
class Perceptron:
    """
    Implementação do classificador Perceptron de Rosenblatt.
    ... (cole o docstring completo) ...
    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for i in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
            if errors == 0 and i > 0: # Adicionei i > 0 para garantir pelo menos 2 épocas
                break
        return self

    def net_input(self, X):
        X_proc = np.atleast_1d(X)
        return np.dot(X_proc, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)

# -----------------------------------------------------------------
# CLASSE 2: FFN (MLP) (Código principal deste repositório)
# -----------------------------------------------------------------
class MLP_FFN:
    """
    Implementação de uma Rede Neural Feed Forward (MLP) com uma camada oculta.
    ... (cole o docstring completo) ...
    """
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1, epochs=10000, alpha=0.9, random_state=1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha
        rgen = np.random.RandomState(random_state)
        self.weights_h = rgen.normal(loc=0.0, scale=0.1, size=(n_input, n_hidden))
        self.bias_h = np.zeros(n_hidden)
        self.weights_o = rgen.normal(loc=0.0, scale=0.1, size=(n_hidden, n_output))
        self.bias_o = np.zeros(n_output)
        self.prev_update_wh = np.zeros_like(self.weights_h)
        self.prev_update_bh = np.zeros_like(self.bias_h)
        self.prev_update_wo = np.zeros_like(self.weights_o)
        self.prev_update_bo = np.zeros_like(self.bias_o)

    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_derivative(self, s):
        return s * (1.0 - s)

    def _forward(self, X):
        net_hidden = np.dot(X, self.weights_h) + self.bias_h
        act_hidden = self._sigmoid(net_hidden)
        net_output = np.dot(act_hidden, self.weights_o) + self.bias_o
        act_output = self._sigmoid(net_output)
        return act_hidden, act_output

    def fit(self, X, y):
        self.cost_ = []
        n_samples = X.shape[0]
        for epoch in range(self.epochs):
            act_hidden, act_output = self._forward(X)
            cost = np.sum((y - act_output)**2) / (2.0 * n_samples)
            self.cost_.append(cost)
            if cost < 0.001 and epoch > 0: # Condição de parada
                break
            error_output = y - act_output
            delta_output = error_output * self._sigmoid_derivative(act_output)
            error_hidden = np.dot(delta_output, self.weights_o.T)
            delta_hidden = error_hidden * self._sigmoid_derivative(act_hidden)
            grad_weights_o = np.dot(act_hidden.T, delta_output) / n_samples
            grad_bias_o = np.sum(delta_output, axis=0) / n_samples
            grad_weights_h = np.dot(X.T, delta_hidden) / n_samples
            grad_bias_h = np.sum(delta_hidden, axis=0) / n_samples
            update_wo = (self.learning_rate * grad_weights_o) + (self.alpha * self.prev_update_wo)
            update_bo = (self.learning_rate * grad_bias_o) + (self.alpha * self.prev_update_bo)
            update_wh = (self.learning_rate * grad_weights_h) + (self.alpha * self.prev_update_wh)
            update_bh = (self.learning_rate * grad_bias_h) + (self.alpha * self.prev_update_bh)
            self.weights_o += update_wo
            self.bias_o += update_bo
            self.weights_h += update_wh
            self.bias_h += update_bh
            self.prev_update_wo = update_wo
            self.prev_update_bo = update_bo
            self.prev_update_wh = update_wh
            self.prev_update_bh = update_bh
        return self

    def predict(self, X):
        X_proc = np.atleast_2d(X)
        _, act_output = self._forward(X_proc)
        return np.where(act_output > 0.5, 1, 0)

# -----------------------------------------------------------------
# FUNÇÃO DE PLOTAGEM
# -----------------------------------------------------------------
def plot_decision_boundary(model, X, y, title):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    if Z.ndim > 1:
        Z = Z.ravel()
    Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=20, edgecolor='k', cmap='RdYlBu')
    plt.title(title)
    plt.xlabel('Característica 1')
    plt.ylabel('Característica 2')

# -----------------------------------------------------------------
# CÓDIGO DE EXECUÇÃO (EXPERIMENTOS)
# -----------------------------------------------------------------

# --- 1. GERAR DADOS ---
print("Gerando dados 'make_moons'...")
X, y = make_moons(n_samples=200, noise=0.1, random_state=42)
y_perceptron = np.where(y == 0, -1, 1) 
y_ffn = y.reshape(-1, 1)

# Plotar dados brutos
plt.figure(figsize=(8, 6))
plt.scatter(X[y == 0, 0], X[y == 0, 1], label='Classe 0 (Lua 1)')
plt.scatter(X[y == 1, 0], X[y == 1, 1], label='Classe 1 (Lua 2)')
plt.title('Conjunto de Dados "Make Moons"')
plt.legend()
plt.grid(True)

# --- 2. TESTE DO PERCEPTRON ---
print("Treinando Perceptron...")
ppn_moons = Perceptron(eta=0.01, n_iter=100)
ppn_moons.fit(X, y_perceptron)
plot_decision_boundary(ppn_moons, X, y, 
                       title="Perceptron (Falha na Separação Linear)")

# --- 3. TESTE DA FFN (MLP) ---
print("Treinando FFN/MLP...")
ffn_moons = MLP_FFN(n_input=2, n_hidden=4, n_output=1, 
                    epochs=5000, 
                    learning_rate=0.1, 
                    alpha=0.9)
ffn_moons.fit(X, y_ffn)
plot_decision_boundary(ffn_moons, X, y, 
                       title="FFN/MLP (Sucesso na Separação Não-Linear)")

print("Demonstração concluída. Exibindo gráficos...")
plt.show()
