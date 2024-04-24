import numpy as np
from svdi.svd.result import SVDResult
import scipy.linalg as la


def get_svd(A, k, p=1000):
    """
    Compute SVD using pcafast algorithm
    Reference: H. Li, G. C. Linderman, A. Szlam, K. P. Stanton, Y. Kluger, and M. Tygert. Algorithm 971: An implementation of a randomized algorithm for principal component analysis. Acm Transactions on Mathematical Software, 43(3):1-14, 2017.
    """
    s = 5
    m, n = A.shape
    B = np.random.normal(size=(n, k + s))
    if p == 0:
        Q, _ = la.qr(A @ B, mode="economic")
    else:
        Q, _ = la.lu(A @ B, permute_l=True)

    for i in range(p):
        Q, _ = la.lu(A.T @ Q, permute_l=True)
        if i == p - 1:
            Q, _ = la.qr(A @ Q, mode="economic")
        else:
            Q, _ = la.lu(A @ Q, permute_l=True)

    B = Q.T @ A

    U, S, V = np.linalg.svd(B, full_matrices=False)

    U = Q @ U
    U = U[:, :k]
    S = S[:k]
    V = V[:k, :]
    return SVDResult(U, S, V)
