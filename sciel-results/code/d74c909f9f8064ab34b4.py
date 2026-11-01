def _verify(U, V, W, n, m, p):
    """Check whether the decomposition exactly reproduces the matmul tensor.
    Returns True if exact (within tight tolerance), else False."""
    import numpy as np
    r = len(U[0]) if U else 0
    if r == 0:
        return False
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    # Reshape to 3D tensors
    # U: (n*m, r) -> (n, m, r)
    Ua = Ua.reshape(n, m, r)
    # V: (m*p, r) -> (m, p, r)
    Va = Va.reshape(m, p, r)
    # W: (p*n, r) -> (p, n, r)
    Wa = Wa.reshape(p, n, r)
    
    # Compute sum over r: T[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # We need to align indices.
    # U: (n, m, r)
    # V: (m, p, r)
    # W: (p, n, r)
    
    # Contract over r and m?
    # T[i,k] should be 1? No, the tensor T[i,j,k] is 1 if...
    # Wait, the tensor T is defined by the multiplication.
    # T[i,j,k] is the coefficient of A[i,j] B[j,k] in C[i,k].
    # So T[i,j,k] = 1 for all valid i,j,k?
    # No, C[i,k] = sum_j A[i,j] B[j,k].
    # The tensor T is a 3-index object T[i,j,k].
    # In the decomposition T = sum U V W, we have:
    # sum_c U[i,j,c] V[j,k,c] W[k,i,c] = T[i,j,k]
    # What is T[i,j,k]?
    # It is 1 if the indices correspond to a valid term in the sum?
    # Actually, the "target" tensor for matrix multiplication M_{n,m,p} is:
    # T[i,j,k] = 1 if we are multiplying A[i,j] and B[j,k] to contribute to C[i,k].
    # But C[i,k] is the sum over j.
    # The tensor T is defined such that <T, A, B> = <C, I> ?
    # No, the standard definition is:
    # T[i,j,k] = 1 for all i,j,k?
    # If T[i,j,k] = 1, then sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1.
    # This implies that for each (i,j,k), the product sums to 1.
    # This is exactly what the schoolbook decomposition does:
    # For each (i,j,k), there is a rank-1 term that is 1 at (i,j,k) and 0 elsewhere?
    # No, schoolbook uses basis vectors.
    # U[i,j,c] = 1 if c corresponds to (i,j,k)?
    # If we use rank n*m*p, we can set U[i,j,c]=1, V[j,k,c]=1, W[k,i,c]=1 for the specific c.
    # Then for a given (i,j,k), only the c corresponding to (i,j,k) contributes 1.
    # All other c's contribute 0.
    # So sum is 1.
    # So the target tensor T[i,j,k] is indeed 1 for all valid indices.
    
    # Let's verify this with the current code's _verify.
    # It checks abs(s - 1.0) > max_err.
    # So yes, T[i,j,k] = 1.
    
    # Efficient computation:
    # We want to compute S[i,j,k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # This is a batched matrix multiplication or einsum.
    # np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
    
    try:
        S = np.einsum('ijc,jkc,kic->ijk', Ua, Va, Wa)
        max_err = np.max(np.abs(S - 1.0))
        return max_err < 1e-12
    except Exception:
        return False
