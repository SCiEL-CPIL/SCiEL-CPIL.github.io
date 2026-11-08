def verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    rank = U.shape[1]
    T = np.zeros((n, m, p))
    for r in range(rank):
        Ur = U[:, r].reshape(n, m)
        Vr = V[:, r].reshape(m, p)
        Wr = W[:, r].reshape(p, n)
        T += np.einsum('ij,jk,ki->ijk', Ur, Vr, Wr)
    if not np.allclose(T, 1.0, atol=tol):
        return False
    return True
