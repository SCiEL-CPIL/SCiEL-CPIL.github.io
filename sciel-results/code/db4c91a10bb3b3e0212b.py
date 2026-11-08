def verify(U, V, W, R, n, m, p):
    import random
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    C_computed = np.zeros((n, p))
    for r in range(R):
        u = np.array(U[:, r]).reshape(n, m)
        v = np.array(V[:, r]).reshape(m, p)
        w = np.array(W[:, r]).reshape(p, n)
        # (u @ A) is not quite right...
        # Actually: sum_k U[i*m+k, r] * A[i,k] for each i
        # This is (U_r reshaped as n×m) element-wise multiplied with A, then summed over k for each i
        # = row sums of (U_r * A)
        uA = np.sum(u * A, axis=1)  # shape (n,)
        vB = np.sum(v * B, axis=0)  # shape (p,)
        # C_computed[i,j] += uA[i] * vB[j] * W[j*n+i, r]
        C_computed += np.outer(uA, vB) * w
    return np.allclose(C_expected, C_computed, atol=1e-10)
