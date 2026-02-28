# written by: ConGon
# Date: 2026-02-27

import numpy as np
from utils import visualization

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(1)

input_neurons = 2
hidden_neurons = 4
output_neurons = 1

W1 = np.random.uniform(size=(input_neurons, hidden_neurons))
b1 = np.zeros((1, hidden_neurons))
W2 = np.random.uniform(size=(hidden_neurons, output_neurons))
b2 = np.zeros((1, output_neurons))

learning_rate = 0.5
epochs = 10000

loss_history = []
prediction_history = []

output_folder = "output_graphs"

for epoch in range(epochs):

    # Forward Propagation
    hidden_input = np.dot(X, W1) + b1
    hidden_output = sigmoid(hidden_input)
    final_input = np.dot(hidden_output, W2) + b2
    predicted_output = sigmoid(final_input)

    # Store prediction snapshot
    prediction_history.append(predicted_output.copy())

    # Loss (Mean Squared Error)
    error = y - predicted_output
    loss = np.mean(np.square(error))
    loss_history.append(loss)

    # Backpropagation
    d_output = error * sigmoid_derivative(predicted_output)
    error_hidden = d_output.dot(W2.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_output)

    # Gradient Descent
    W2 += hidden_output.T.dot(d_output) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate

print("Final Predicted Output:")
print(np.round(predicted_output, 3))

# Visualization calls
visualization.plot_loss(loss_history, output_folder)
visualization.plot_predictions(y, predicted_output, output_folder)
visualization.plot_epoch_predictions(y, prediction_history, output_folder)

print(f"Graphs saved to folder: {output_folder}")