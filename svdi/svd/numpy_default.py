import numpy as np

from svdi.svd.result import SVDResult


def get_svd(A, k) -> SVDResult:
    U, S, Vh = np.linalg.svd(A)
    return SVDResult(U=U[:, :k], S=S[:k], Vh=Vh[:k, :])
