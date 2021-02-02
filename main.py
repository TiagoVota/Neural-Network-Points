import random
import numpy as np
import time


"""
Esse programa faz um treino para analizar se um ponto está abaixo de uma parábola (pontos da
forma [x, x ** 2]) ou acima dela, no intervalo de domínio (0, 1).
"""

t0 = time.time()  # Início da cronometagem do processo

max_train = 10000  # Quantidades de treinos realizados
gamma = 1  # Taxa de aprendizado

# Criar base de dados
n = 10000  # Número de elementos
lista_dados = []
for i in range(n):
	# Adicionar pontos randomicos à nossa base de dados
    x, y = random.random(), random.random()
    elem = [x, y]
    lista_dados.append(elem)

# Criando um gabarito para nossos pontos
lista_resp = []
for i in range(n):
    if lista_dados[i][0] ** 2 < lista_dados[i][1]:
        lista_resp.append(1)
    else:
        lista_resp.append(0)


# Criar função da Rede Neural
def NN(peso, bias, arr):
    output = peso[0] * arr[0] + peso[1] * arr[1] + bias
    output = 1 / (1 + np.exp(-output))
    return output


# Criar função Erro
def gradNN(peso, bias, arr):
    x = peso[0] * arr[0] + peso[1] * arr[1] + bias
    escalar = np.exp(-x) * (1 + np.exp(-x)) ** (-2)
    output = [escalar * arr[0], escalar * arr[1], escalar]
    return output


# Inicializando vetor peso e bias
peso = [random.random(), random.random()]
bias = random.random()
# Método do gradiente para treinar a neural network
gradiente = [0, 0, 0]
for k in range(max_train):
    # Escolhe um ponto i aleatório da base, para não deixar o treino viciado
    i = random.randint(0, n - 1)
    # Calcula o gradiente
    temp = 2 * (NN(peso, bias, lista_dados[i]) - lista_resp[i])
    gradiente[0] = temp * gradNN(peso, bias, lista_dados[i])[0]
    gradiente[1] = temp * gradNN(peso, bias, lista_dados[i])[1]
    gradiente[2] = temp * gradNN(peso, bias, lista_dados[i])[2]
    # Atualiza o peso e bias
    peso[0] -= gamma * gradiente[0]
    peso[1] -= gamma * gradiente[1]
    bias -= gamma * gradiente[2]

# Verificando quantidade de acertos e erros em uma nova base de testes
n_test = 1000000  # Quantidade de testes realizados
acertos, erros = 0, 0
for i in range(n_test):
    ponto = [random.random(), random.random()]
    gab = 0
    if ponto[0] ** 2 < ponto[1]:
        gab = 1
    if NN(peso, bias, ponto) < 0.5:
        valor = 0
    else:
        valor = 1
    if gab == valor:
        acertos += 1
    else:
        erros += 1

# Exibir resultado
print('Quantidade de erros: {};\nQuantidade de acertos: {};\nPorcentagem de acertos: {:.2f}%.'
      .format(erros, acertos, acertos / n_test * 100))
print('Tempo de excecução: {:.2f} seg.'.format(time.time() - t0))
