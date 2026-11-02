def verify(U, V, W):
    T = np.einsum('ijr,jkr,kir->ijk', U, V, W)
    return np.allclose(T, 1.0, atol=1e-6)
