def _validate_fast(U, V, W, T):
    n, m, p = T.shape
    # U: (n*m, R), V: (m*p, R), W: (p*n, R)
    # Reshape to (n, m, R), (m, p, R), (p, n, R)
    U_mat = U.reshape(n, m, -1)
    V_mat = V.reshape(m, p, -1)
    W_mat = W.reshape(p, n, -1)
    
    # Sum over R: U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Result shape: (n, m, p)
    # einsum: 'ijr,jkr,kir->ijk'
    recon = np.einsum('ijr,jkr,kir->ijk', U_mat, V_mat, W_mat, optimize=True)
    return np.allclose(recon, T, atol=1e-9)
