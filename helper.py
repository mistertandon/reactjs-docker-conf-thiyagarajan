import numpy as np

def compute_linear_eqn(X, weights, bias):
    """
    FORWARD PROPAGATION
    --------------------

    Here we are doing linear computations of all examples (On example contains available all input features) in one training
    dataset simulataneously.


    Where X : array([[0, 0, 0],
                     [0, 1, 1],
                     [1, 0, 1],
                     [1, 1, 1]])

    And

    weights: array([0.1, 0.6])
    """

    Z = np.add(bias, np.dot(weights, X))

    return Z


def compute_sigmoid(z):
    """
    We are here trying to squash output to range (0, 1)
    Note: Parenthesis implies exclusive boundary values.

    Here we input linear computation to sigmoid function and gets output range (0, 1)

    z: Linear equation.
    For this neural network it would be
    [
        weight[0] * X[0][0] + weight[1] * X[1][0] + b
        weight[0] * X[0][1] + weight[1] * X[1][1] + b
        weight[0] * X[0][2] + weight[1] * X[1][2] + b
        weight[0] * X[0][3] + weight[1] * X[1][3] + b
    ]

    sigmoid output
    """

    sig = np.divide(1, np.add(1, np.exp(-z)))

    return sig


def compute_cost(y, y_hat, m, const=2):
    """
    As we're considering all example simulataneously therefore we're using
    cost function instead loss function.

    We have to sum up all the loos due to each example and compute average.
    """

    total_cost = np.sum(np.divide(np.subtract(y, y_hat) ** 2, const))

    avg_cost = np.divide(total_cost, m)

    return avg_cost


def compute__del_cost__by__del_y_hat(y, y_hat, m):
    """
    here we'are computing gradient of cost function w.r.t y_hat
    as we know cost function is (y - y_hat)^2 / ( 2 * m )
    Therefore it's gardien would be

    (-1 / m)(y - y_hat)

    avg_grad: vector of m length. Each element contains loss corresponding to each example.
    """

    grad = -np.subtract(y, y_hat)

    avg_grad = np.divide(grad, m)

    return avg_grad


def compute__del_y_hat__by__del_z(y_hat):
    """
        del_y_hat__by__del_z [$(y_hat)/$(z)]
            As we know
            y_hat = sigmoid(z), therefore,
            $(sigmoid(z)) / $(z) : sigmoid(z)[1 - sigmoid(z)] OR [y_hat * (1 - y_hat)]
    """

    one_sub_sigma = np.subtract(1, y_hat)

    grad = np.multiply(y_hat, one_sub_sigma)

    return grad


def final_grad_at__z(local__del_cost__by__del_y_hat, local__del_y_hat__by__del_z):
    """
    Here we're computing final gradient at Z node i.e. multiplication of
    local gradient at Z node and incming gradient from cost.
    ( $cost/$y_hat * $y_hat/$z )

    local__del_cost__by__del_y_hat : Gradient of cost function i.e. ($cost/$y_hat)
    grad_y_hat__by__del_z: Local gradient at node Z ($y_hat/$z)

    grad_at_node__z: final gradient at node z
    """
    grad_at_node__z = np.multiply(local__del_cost__by__del_y_hat, local__del_y_hat__by__del_z)

    return grad_at_node__z


def compute__del_z__by__del_w(X):
    """
    How to comput local gradient of z w.r.t w ($z/$w)
    And
    local gradient of z w.r.t b ($z/$b)

    1. w is a vector i.e. np.array([0.1, 0.6])

    2. We do have 4 examples in training dataset

    array([[0, 0],
           [0, 1],
           [1, 0],
           [1, 1]])

    local__del_z__by__del_w matrix would contains

     __       __               __                    __               __                    __
    |           |             |                        |             |                        |
    |   $z1/$w  |             |   $z1/$w1    $z1/$w2   |             |   x[0][0]    x[0][1]   |
    |           |             |                        |             |                        |
    |   $z2/$w  |             |   $z2/$w1    $z2/$w2   |             |   x[1][0]    x[1][1]   |
    |           |    ===>     |                        |    ===>     |                        |
    |   $z3/$w  |             |   $z3/$w1    $z3/$w2   |             |   x[2][0]    x[2][1]   |
    |           |             |                        |             |                        |
    |   $z4/$w  |             |   $z4/$w1    $z4/$w2   |             |   x[3][0]    x[3][1]   |
    |__       __|             |__                    __|             |__                    __|

    In above matrix:

        z1 = weights[0] * X[0][0] + weights[1] * X[0][1] + b
        z2 = weights[0] * X[1][0] + weights[1] * X[1][1] + b
        z3 = weights[0] * X[2][0] + weights[1] * X[2][1] + b
        z4 = weights[0] * X[3][0] + weights[1] * X[3][1] + b

        AND

        w1, w2 = weights
    """

    return X


def compute__del_z__by__del_b():
    """
    Note: gradient of z w.r.t bias ($z/$b)

    compute__del_z__by__del_b OR del_z__by__del_b

    As we know that at Z node linear equation would be (In general)
    z = np.mdot(weight, X) + b

    Therefore it's differentiation w.r.t bias would be 1

    """

    return np.ones(1)