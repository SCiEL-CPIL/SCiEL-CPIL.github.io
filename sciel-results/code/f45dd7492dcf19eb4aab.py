def _verify(U, V, W, n, m, p):
    r = U.shape[1]
    # T[i, j, jp, k] = sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c]
    # We want T[i, j, jp, k] == 1 if j==jp else 0
    
    # Reshape U, V, W to allow broadcasting
    # U: (n, m, r)
    # V: (m, p, r)
    # W: (p, n, r)
    
    U_reshaped = U.reshape(n, m, r)
    V_reshaped = V.reshape(m, p, r)
    W_reshaped = W.reshape(p, n, r)
    
    # We need to sum over r.
    # Term for (i, j, jp, k) is U[i, j, :] dot V[jp, k, :] * W[k, i, :]? No.
    # The tensor entry is sum_c U[i*m+j, c] * V[jp*p+k, c] * W[k*n+i, c].
    # This is not a simple dot product of vectors from U, V, W because the indices are mixed.
    
    # We can construct the full tensor T of shape (n, m, m, p)
    # T[i, j, jp, k] = sum_c U[i, j, c] * V[jp, k, c] * W[k, i, c]
    
    # This can be computed as:
    # Let M1[i, j, c] = U[i, j, c]
    # Let M2[jp, k, c] = V[jp, k, c]
    # Let M3[k, i, c] = W[k, i, c]
    
    # We want sum_c M1[i, j, c] * M2[jp, k, c] * M3[k, i, c]
    
    # We can do this by iterating over c? No, that's slow.
    # We can use np.einsum
    # 'ijc,jkc,kic->ijk' ? No, indices are i, j, jp, k.
    # U: i, j, c
    # V: jp, k, c
    # W: k, i, c
    # Result: i, j, jp, k
    
    # np.einsum('ijc,jkc,kic->ijpk', U_reshaped, V_reshaped, W_reshaped)
    # Note: W is reshaped to (p, n, r) so indices are k, i, c.
    # So 'kic' is correct.
    
    try:
        T = np.einsum('ijc,jkc,kic->ijpk', U_reshaped, V_reshaped, W_reshaped)
    except:
        return False
        
    # Expected tensor
    # E[i, j, jp, k] = 1 if j == jp else 0
    # This is independent of i and k.
    # E[i, j, jp, k] = delta_{j, jp}
    
    # We can check if T[i, j, jp, k] == delta_{j, jp}
    # For all i, k.
    
    # Check T[:, :, :, :] against delta
    # delta is (m, m)
    delta = np.eye(m)
    
    # T shape (n, m, m, p)
    # We want T[i, j, jp, k] == delta[j, jp]
    
    # Reshape T to (n*p, m, m) ?
    # T[i, j, jp, k] -> T_reshaped[i, k, j, jp]
    T_reshaped = T.transpose(0, 3, 1, 2).reshape(n * p, m, m)
    
    # Compare each (n*p) slice to delta
    # All slices should be delta.
    
    if not np.allclose(T_reshaped, delta[None, :, :]):
        return False
        
    return True
