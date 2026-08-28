import numpy as np

# ===============================================================
# FUNCTIONS
# ===============================================================
def sigmoid (x : np.ndarray) -> np.ndarray:
    return 1/(1 + np.exp(-x))

def signmoid_derivative (x : np.ndarray) -> np.ndarray:
    sig = sigmoid(x)
    return (1 - sig)*(sig)

def forward_pass (W: np.ndarray, x:np.ndarray, b:np.ndarray) -> np.ndarray:
    return W@x + b

def calculate_loss (x: ndarray, y : ndarray) -> ndarray:
    return 0.5*(x - y)**2

# ===============================================================
# PARAMETERS
# ===============================================================

W1 = np.array([
    [0.4, -0.2],
    [0.1, 0.6]
])

b1 = np.array([
    [0.1],
    [-0.1]
])

W2 = np.array([
    [0.3, -0.5]
])

b2 = np.array([
    [0.2]
])

y = 1
x = np.array([
    [0.5],
    [-1.0]
])


if __name__ == "__main__":

# ===============================================================
# FORWARD PASSES
# ===============================================================

    a = forward_pass(W1, x, b1)
    h = sigmoid(a)

    z = forward_pass(W2, h, b2)
    y_hat = sigmoid(z)

    L = calculate_loss(y_hat, y)

    print("a:")
    print(a)

    print("h:")
    print(h)

    print("z:")
    print(z)

    print("y_hat:")
    print(y_hat)

    print("L:")
    print(L)

    print("Backpropogation:")

    part_L_yhat = y_hat - y
    part_L_z = part_L_yhat * signmoid_derivative(z)
    part_L_W2 = part_L_z @ h.T
    part_L_h = W2.T @ part_L_z
    part_L_b2 = part_L_z
    part_L_a = part_L_h * signmoid_derivative(a)
    part_L_W1 = part_L_a @ x.T
    part_L_x = W1.T @ part_L_a
    part_L_b1 = part_L_a

    print("L/yhat")
    print(part_L_yhat)
    print("L/z")
    print(part_L_z)
    print("L/W2")
    print(part_L_W2)
    print("L/h")
    print(part_L_h)
    print("L/b2")
    print(part_L_b2)
    print("L/W1")
    print(part_L_W1)
    print("L/a")
    print(part_L_a)
    print("L/x")
    print(part_L_x)
    print("L/b")
    print(part_L_b1)

# ==========================================
# FINITE DIFFERENCE CHECK
# ==========================================

    print ("Running finite difference check...")

    def calculate_loss(x: np.ndarray, y: np.ndarray) -> float:
        return np.sum(0.5 * (x - y)**2)

    def calculate_loss_for_params(W1, b1, W2, b2, x, y):
        a = forward_pass(W1, x, b1)
        h = sigmoid(a)

        z = forward_pass(W2, h, b2)
        y_hat = sigmoid(z)

        return calculate_loss(y_hat, y)


    def numerical_gradient(param, analytical_gradient, loss_function, epsilon=1e-5):
        numerical_gradient = np.zeros_like(param, dtype=float)

        iterator = np.nditer(param, flags=["multi_index"], op_flags=["readwrite"])

        while not iterator.finished:
            index = iterator.multi_index

            original_value = param[index]

            # f(x + epsilon)
            param[index] = original_value + epsilon
            loss_plus = loss_function()

            # f(x - epsilon)
            param[index] = original_value - epsilon
            loss_minus = loss_function()

            # Restore original value
            param[index] = original_value

            # Central difference
            numerical_gradient[index] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

            iterator.iternext()

        difference = np.linalg.norm(
            numerical_gradient - analytical_gradient
        )

        print("Analytical gradient:")
        print(analytical_gradient)

        print("Numerical gradient:")
        print(numerical_gradient)

        print("Difference:")
        print(difference)

        print()

        return numerical_gradient

    print("========== W1 ==========")

    numerical_gradient(
        W1,
        part_L_W1,
        lambda: calculate_loss_for_params(W1, b1, W2, b2, x, y)
    )


    print("========== b1 ==========")

    numerical_gradient(
        b1,
        part_L_b1,
        lambda: calculate_loss_for_params(W1, b1, W2, b2, x, y)
    )


    print("========== W2 ==========")

    numerical_gradient(
        W2,
        part_L_W2,
        lambda: calculate_loss_for_params(W1, b1, W2, b2, x, y)
    )


    print("========== b2 ==========")

    numerical_gradient(
        b2,
        part_L_b2,
        lambda: calculate_loss_for_params(W1, b1, W2, b2, x, y)
    )


    print("========== x ==========")

    numerical_gradient(
        x,
        part_L_x,
        lambda: calculate_loss_for_params(W1, b1, W2, b2, x, y)
    )

    print("===== dL/da norm =====")
    print (np.linalg.norm(part_L_a))
