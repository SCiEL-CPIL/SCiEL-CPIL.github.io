def _verify_decomposition(U, V, W, m, n, p):
    """Verify that U, V, W correctly decompose m x n * n x p."""
    import numpy as np
    rank = len(U)
    # Random test
    np.random.seed(42)
    A = np.random.rand(m, n)
    B = np.random.rand(n, p)
    C = A @ B
    
    C_calc = np.zeros((m, p))
    for r in range(rank):
        # U row: m*n
        u = np.array(U[r]).reshape(m, n)
        # V row: n*p
        v = np.array(V[r]).reshape(n, p)
        # W row: p*m -> W[k*m+i]
        w = np.array(W[r]).reshape(p, m)
        
        # Term is u * v * w?
        # C[i,k] = sum_r U[i,j] V[j,k] W[k,i]?
        # No, the tensor is sum_r U[r] V[r] W[r]
        # U[r] is a matrix m x n
        # V[r] is a matrix n x p
        # W[r] is a matrix p x m?
        # The convention is W[k*m+i, r].
        # So W[r] reshaped to (p, m) has W[k, i] = W[k*m+i].
        # The term is sum_j U[i,j] V[j,k] * W[k,i]?
        # No, the decomposition is C[i,k] = sum_r U[i,j] V[j,k] W[k,i] is wrong.
        # It is C[i,k] = sum_r U[r]_{i,j} V[r]_{j,k} W[r]_{k,i}?
        # No, the tensor is T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        # So C[i,k] = sum_j sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}.
        # This is not a simple matrix product.
        
        # Let's check the Strassen example.
        # C11 = M1 + M4 - M5 - M7
        # M1 = (a+d)(e+h)
        # U[0] = [1, 0, 0, 1] (a+d)
        # V[0] = [1, 0, 0, 1] (e+h)
        # W[0] = [1, 0, 0, 1, -1, 0, -1] (C11 coeff 1)
        # So C11 gets +1 * M1.
        
        # So C[i,k] = sum_r U[i,j] V[j,k] W[k,i] is NOT correct.
        # The term is U[i,j] * V[j,k] * W[k,i]?
        # No, U and V are contracted over j.
        # The result is a matrix M_r = U_r @ V_r.
        # Then C = sum_r M_r * W_r?
        # No, W_r is a matrix.
        # The decomposition is C = sum_r (U_r @ V_r) * W_r?
        # No, W_r is not a scalar.
        
        # The tensor decomposition is:
        # T_{i,j,k} = sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # This means C[i,k] = sum_j sum_r U_{i,j,r} V_{j,k,r} W_{k,i,r}
        # This is equivalent to:
        # C = sum_r (U_r @ V_r) * W_r (element-wise? No)
        # Actually, if we define M_r = U_r @ V_r, then M_r is m x p.
        # Then C[i,k] = sum_r M_r[i,k] * W_r[k,i]?
        # No, W_r[k,i] is the coefficient for C[i,k] from M_r.
        # So C = sum_r M_r * W_r (where * is element-wise multiplication? No, W_r is a matrix of coefficients).
        # Yes, C[i,k] = sum_r M_r[i,k] * W_r[k,i].
        # So C = sum_r (U_r @ V_r) * W_r (element-wise).
        
        M_r = u @ v
        # W_r is p x m. We need W_r[k, i] for C[i, k].
        # So we need to multiply M_r[i, k] by W_r[k, i].
        # This is element-wise multiplication of M_r and W_r.T?
        # W_r.T is m x p. (W_r.T)[i, k] = W_r[k, i].
        # So C += M_r * W_r.T
        
        C_calc += M_r * w.T

    if not np.allclose(C, C_calc):
        return False
    return True
