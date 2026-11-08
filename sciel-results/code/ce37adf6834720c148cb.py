def _verify(U, V, W):
    """Check that sum_r U[:,r]⊗V[:,r]⊗W[:,r] equals multiplication tensor.
    U: (12, r) idx i*m+j ; V: (20, r) idx j*p+k ; W: (15, r) idx k*n+i."""
    r = U.shape[1]
    # Build tensor T[i,j,k]
    # T = einsum('ir,jr,kr->ijk', U.reshape(n,m,r), V.reshape(m,p,r), W.reshape(p,n,r))
    # But indices are flattened.
    Umat = np.asarray(U, dtype=float).reshape(n, m, r)
    Vmat = np.asarray(V, dtype=float).reshape(m, p, r)
    Wmat = np.asarray(W, dtype=float).reshape(p, n, r)
    T = np.einsum('ijr,jkr,kir->ijk', Umat, Vmat, Wmat)
    M = np.ones((n, m, p))
    return np.allclose(T, M, atol=1e-9)
