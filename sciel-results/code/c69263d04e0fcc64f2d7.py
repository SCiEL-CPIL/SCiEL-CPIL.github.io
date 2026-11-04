<<<<<<< SEARCH
def _verify(U, V, W, T, trials=3):
    """Correct evaluator check: reconstruct C = A@B from the tensor factors."""
    n, m, p = _DIMS
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.RandomState(0)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        Cexp = A @ B
        Ccomp = np.zeros((n, p))
        for r in range(rank):
            ur = np.dot(A.ravel(), U[:, r])
            vr = np.dot(B.ravel(), V[:, r])
            Ccomp += (ur * vr) * W[:, r].reshape(p, n)
        if not np.allclose(Ccomp, Cexp, atol=1e-8):
            return False
    return True
=======
def _verify(U, V, W, T, trials=3):
    """Correct evaluator check: reconstruct C = A@B from the tensor factors."""
    n, m, p = _DIMS
    rank = U.shape[1]
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rng = np.random.RandomState(0)
    for _ in range(trials):
        A = rng.randn(n, m)
        B = rng.randn(m, p)
        Cexp = A @ B
        Ccomp = np.zeros((n, p))
        for r in range(rank):
            ur = np.dot(A.ravel(), U[:, r])
            vr = np.dot(B.ravel(), V[:, r])
            # W[:, r] is flattened with index z = j*n + i
            # Reshape to (p, n) gives M[j, i] = W[j*n + i]
            # C[i, j] needs M[j, i], so we transpose
            Ccomp += (ur * vr) * W[:, r].reshape(p, n).T
        if not np.allclose(Ccomp, Cexp, atol=1e-8):
            return False
    return True
>>>>>>> REPLACE
