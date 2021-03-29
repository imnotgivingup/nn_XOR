# XOR Function
# X Y X xor Y
# 0 0   0
# 0 1   1
# 1 0   1
# 1 1   0
import numpy as np
import random

X = np.array(
    [[[0.0], [0.0]],
     [[0.0], [1.0]],
     [[1.0], [0.0]],
     [[1.0], [1.0]]]
)
Y = np.array([[0.0], [1.0], [1.0], [0.0]])


# set up weights and biases
iterations = 1000
eta = 9
num_layers = 3
sizes = [2, 10, 1]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]


def forward(X, Y):
    af = []
    for x, y in zip(X, Y):
        a = x
        for b, w in zip(biases, weights):
            a = sigmoid(np.dot(w, a)+b)
        af.append(a)
    return af


def cost_derivative(output_activations, y):
    return (output_activations-y)


def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))


def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))


def backprop(X, Y):
    nabla_bf, nabla_wf = [], []
    for x, y in zip(X, Y):
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]

        # feedforward
        activation = x
        activations = [x]  # list to store all the activations, layer by layer
        zs = []
        for b, w in zip(biases, weights):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)

        # backward pass
        delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for l in range(2, num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        nabla_bf.append(nabla_b)
        nabla_wf.append(nabla_w)

    return (nabla_bf, nabla_wf)


def update(X, Y):
    global weights, biases

    delta_nabla_bf, delta_nabla_wf = backprop(X, Y)

    for delta_nabla_b, delta_nabla_w in zip(delta_nabla_bf, delta_nabla_wf):
        nabla_b = [np.zeros(b.shape) for b in biases]
        nabla_w = [np.zeros(w.shape) for w in weights]

        nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
        nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        weights = [w-(eta*nw) for w, nw in zip(weights, nabla_w)]
        biases = [b-(eta*nb) for b, nb in zip(biases, nabla_b)]


for i in range(iterations):
    # for j in range(len(X)):
    update(X, Y)

pred = forward(X, Y)
print("Predicted result", pred)
print("Expected result [[0],[1],[1],[0]]")
error = 0
for i in range(4):
    error += abs(pred[i][0][0]-Y[i][0])
print((1-error/4)*100, "% accurate")
