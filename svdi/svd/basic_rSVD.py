import numpy as np
from svdi.svd.result import SVDResult
import scipy.linalg as la


def get_svd(A, k, p=1000):
    """
    Compute SVD using rSVD algorithm
    Reference: N Halko, P. G Martinsson, and J. A Tropp. Finding structure with randomness: Probabilistic algorithms for constructing approximate matrix decompositions. Siam Review, 53(2):217-288, 2011.
    """
    s = 5
    m, n = A.shape
    B = np.random.normal(size=(n, k + s))
    Q, _ = la.qr(A @ B, mode="economic")

    for _ in range(p):
        Q, _ = la.qr((A.T @ Q), mode="economic")
        Q, _ = la.qr((A @ Q), mode="economic")

    B = Q.T @ A
    U, S, V = np.linalg.svd(B, full_matrices=False)
    U = Q @ U
    U = U[:, :k]
    S = S[:k]
    V = V[:k, :]

    return SVDResult(U, S, V)
