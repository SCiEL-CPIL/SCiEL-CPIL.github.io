import numpy as np

def _verify(result, num_tests=100):
    """Verify the decomposition by random integer tests using NumPy."""
    n, m, p = 2, 2, 6
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]

    if U.shape != (n * m, rank) or V.shape != (m * p, rank) or W.shape != (p * n, rank):
        return False

    rng = np.random.default_rng(42)
    for _ in range(num_tests):
        A = rng.integers(-5, 6, size=(n, m))
        B = rng.integers(-5, 6, size=(m, p))
        Cref = A @ B
        
        # Compute C using decomposition
        # C[i,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # This is a bit complex to vectorize directly in one line without loops over i,k,j
        # But we can use einsum or matrix multiplications.
        # U is (nm, r), V is (mp, r), W is (pn, r)
        # We want C (n, p)
        
        # Let's reshape U, V, W to 3D
        # U[i, j, r], V[j, k, r], W[k, i, r]
        U3 = U.reshape(n, m, rank)
        V3 = V.reshape(m, p, rank)
        W3 = W.reshape(p, n, rank)
        
        # C[i, k] = sum_j sum_r U3[i, j, r] V3[j, k, r] W3[k, i, r]
        # This is not a standard dot product due to index permutations.
        # We can iterate over r? No, that's slow.
        # We can use np.einsum
        # 'ijr,jkr,kr->ik' ? 
        # U3 indices: i, j, r
        # V3 indices: j, k, r
        # W3 indices: k, i, r
        # Result: i, k
        C = np.einsum('ijr,jkr,kr->ik', U3, V3, W3)
        
        if not np.allclose(C, Cref):
            return False
    return True
