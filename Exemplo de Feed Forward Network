import numpy as np
# Remova esta linha se não for plotar o gráfico
import matplotlib.pyplot as plt 

class MLP_FFN:
    """
    Implementação de uma Rede Neural Feed Forward (MLP) com uma camada oculta.
    Utiliza a função Sigmoide como ativação e gradiente descendente em lote com momentum.
    """
    def __init__(self, n_input, n_hidden, n_output, learning_rate=0.1, epochs=10000, alpha=0.9, random_state=1):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.alpha = alpha # Coeficiente de Momentum

        # Inicialização dos pesos e biases (Formato: input_dim x output_dim)
        rgen = np.random.RandomState(random_state)
        self.weights_h = rgen.normal(loc=0.0, scale=0.1, size=(n_input, n_hidden))
        self.bias_h = np.zeros(n_hidden)
        self.weights_o = rgen.normal(loc=0.0, scale=0.1, size=(n_hidden, n_output))
        self.bias_o = np.zeros(n_output)

        # Para armazenar as atualizações anteriores (necessário para o momentum)
        self.prev_update_wh = np.zeros_like(self.weights_h)
        self.prev_update_bh = np.zeros_like(self.bias_h)
        self.prev_update_wo = np.zeros_like(self.weights_o)
        self.prev_update_bo = np.zeros_like(self.bias_o)


    def _sigmoid(self, z):
        # np.clip para evitar overflow/underflow no exp()
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    def _sigmoid_derivative(self, s):
        # s é a saída da sigmoide
        return s * (1.0 - s)

    def _forward(self, X):
        # Passagem da entrada para a camada oculta
        # Z_h = X * W_h + b_h  (Dimensões: [samples, input] * [input, hidden] + [hidden] = [samples, hidden])
        net_hidden = np.dot(X, self.weights_h) + self.bias_h
        act_hidden = self._sigmoid(net_hidden)

        # Passagem da camada oculta para a camada de saída
        # Z_o = A_h * W_o + b_o  (Dimensões: [samples, hidden] * [hidden, output] + [output] = [samples, output])
        net_output = np.dot(act_hidden, self.weights_o) + self.bias_o
        act_output = self._sigmoid(net_output)

        return act_hidden, act_output

    def fit(self, X, y):
        """
        Treina a rede usando backpropagation em lote com momentum.
        """
        self.cost_ = []
        n_samples = X.shape[0]

        for epoch in range(self.epochs):

            # 1. Passagem Direta (Forward Pass) - Usando todo o lote
            act_hidden, act_output = self._forward(X)

            # 2. Cálculo do Erro (Função de Custo - Erro Quadrático Médio por amostra)
            cost = np.sum((y - act_output)**2) / (2.0 * n_samples)
            self.cost_.append(cost)

            # --- Impressão opcional do custo para monitoramento ---
            if (epoch + 1) % 500 == 0:
                 print(f"Época {epoch+1}/{self.epochs}, Custo: {cost:.6f}")
            
            # --- Verificação de Convergência ---
            # Se o custo for muito baixo, podemos parar
            if cost < 0.001:
                print(f"Convergência alcançada na época {epoch+1} com custo {cost:.6f}.")
                break


            # 3. Retropropagação (Backward Pass) - Calculando gradientes para o lote

            # Erro e delta da camada de saída
            error_output = y - act_output
            delta_output = error_output * self._sigmoid_derivative(act_output)

            # Erro e delta da camada oculta (retropropagado)
            error_hidden = np.dot(delta_output, self.weights_o.T) 
            delta_hidden = error_hidden * self._sigmoid_derivative(act_hidden)

            # Cálculo dos gradientes médios no lote
            grad_weights_o = np.dot(act_hidden.T, delta_output) / n_samples
            grad_bias_o = np.sum(delta_output, axis=0) / n_samples

            grad_weights_h = np.dot(X.T, delta_hidden) / n_samples
            grad_bias_h = np.sum(delta_hidden, axis=0) / n_samples

            # 4. Atualização dos Pesos (Gradiente Descendente com Momentum)

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

        print(f"Treinamento concluído após {self.epochs} épocas.")
        return self

    def predict(self, X):
        X_proc = np.atleast_2d(X)
        _, act_output = self._forward(X_proc)
        # Retorna 1 se a saída for > 0.5 (limiar), 0 caso contrário
        return np.where(act_output > 0.5, 1, 0)

# --- Exemplo de Uso (Problema XOR, não linearmente separável) ---
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([[0], [1], [1], [0]]) # Saídas esperadas para XOR

# --- HIPERPARÂMETROS OTIMIZADOS ---
# Topologia: 2 entradas, 4 neurônios ocultos (mais robusto), 1 saída
# Aumentei o learning_rate e diminuí as épocas (ajuste)
ffn = MLP_FFN(n_input=2, n_hidden=4, n_output=1, 
              epochs=5000, 
              learning_rate=0.5, 
              alpha=0.9) # Momentum alto ajuda a "pular" mínimos locais
ffn.fit(X_xor, y_xor)

print("\n--- Teste da FFN (Porta Lógica XOR) ---")
print(f"Custo final: {ffn.cost_[-1]:.6f}")
print("Predições para o XOR:")
predictions = ffn.predict(X_xor)
print(predictions)

# Verificar acurácia
correct_predictions = np.sum(predictions == y_xor)
accuracy = correct_predictions / y_xor.shape[0]
print(f"Acurácia no XOR: {accuracy * 100:.2f}%") # Deve ser 100%

# Plotar a curva de custo (descomente para usar)
# plt.plot(range(1, len(ffn.cost_) + 1), ffn.cost_)
# plt.xlabel('Épocas')
# plt.ylabel('Custo (MSE)')
# plt.title('Curva de Aprendizado da FFN no XOR')
# plt.grid(True)
# plt.show()
