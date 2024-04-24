import math
from svdi.svd.result import SVDResult
import numpy as np


def power_svd(A, iterations):
    """Compute SVD using Power Method.
    Refercence Link: http://www.cs.yale.edu/homes/el327/datamining2013aFiles/07_singular_value_decomposition.pdf
    """
    mu, sigma = 0, 1
    x = np.random.normal(mu, sigma, size=A.shape[1])
    B = A.T.dot(A)

    for _ in range(iterations):
        new_x = B.dot(x)
        x = new_x
    v = x / np.linalg.norm(x)
    sigma = np.linalg.norm(A.dot(v))
    u = A.dot(v) / sigma
    return np.reshape(u, (A.shape[0], 1)), sigma, np.reshape(v, (A.shape[1], 1))


def get_svd(A, k):
    # Define the number of iterations
    delta = 0.1
    epsilon = 0.97
    lamda = 2
    iterations = int(
        math.log(4 * math.log(2 * A.shape[1] / delta) / (epsilon * delta)) / (2 * lamda)
    )

    rank = np.linalg.matrix_rank(A)
    U = np.zeros((A.shape[0], 1))
    S = []
    V = np.zeros((A.shape[1], 1))

    # SVD using Power Method
    for _ in range(rank):
        u, sigma, v = power_svd(A, iterations)
        U = np.hstack((U, u))
        S.append(sigma)
        V = np.hstack((V, v))
        A = A - u.dot(v.T) * sigma

    # Discard the initial zero column used for initialization
    U = U[:, 1:]
    V = V[:, 1:]

    # Return the SVDResult
    return SVDResult(U=U[:, :k], S=np.array(S)[:k], Vh=V.T[:k, :])
