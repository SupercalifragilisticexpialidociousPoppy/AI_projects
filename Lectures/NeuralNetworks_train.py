import numpy as np
import matplotlib.pyplot as plt

import NeuralNetworks_numpy as nn


learning_rate = 0.1
steps = 300

loss_history = []


for step in range(steps):

    # Forward pass
    a = nn.forward_pass(nn.W1, nn.x, nn.b1)
    h = nn.sigmoid(a)

    z = nn.forward_pass(nn.W2, h, nn.b2)
    y_hat = nn.sigmoid(z)

    L = nn.calculate_loss(y_hat, nn.y)

    loss_history.append(L.item())


    # Backward pass
    dL_dyhat = y_hat - nn.y

    dL_dz = (
        dL_dyhat *
        nn.signmoid_derivative(z)
    )

    dL_dW2 = dL_dz @ h.T
    dL_db2 = dL_dz

    dL_dh = nn.W2.T @ dL_dz

    dL_da = (
        dL_dh *
        nn.signmoid_derivative(a)
    )

    dL_dW1 = dL_da @ nn.x.T
    dL_db1 = dL_da


    # Update
    nn.W1 -= learning_rate * dL_dW1
    nn.b1 -= learning_rate * dL_db1

    nn.W2 -= learning_rate * dL_dW2
    nn.b2 -= learning_rate * dL_db2


plt.plot(loss_history)
plt.xlabel("Training step")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()
