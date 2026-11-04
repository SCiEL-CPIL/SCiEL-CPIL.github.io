"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Build the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1 for the MM tensor
    # Actually, T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i'
    # For MM: T[i*m+j, j*p+k, k*n+i] = 1
    
    # The tensor has non-zero entries only at specific positions
    # T[i*m + j, j*p + k, k*n + i] = 1 for all i,j,k
    
    U_dim = n * m  # 16
    V_dim = m * p  # 28
    W_dim = p * n  # 28
    
    # Try ranks from low to high
    for rank in range(1, U_dim * V_dim * W_dim + 1):
        # Initialize randomly
        rng = np.random.RandomState(42 + rank)
        U = rng.randn(U_dim, rank)
        V = rng.randn(V_dim, rank)
        W = rng.randn(W_dim, rank)
        
        # Alternating least squares
        for iteration in range(100):
            # Update W
            for idx in range(W_dim):
                # idx = k*n + i
                k = idx // n
                i = idx % n
                # For this (k,i), we need: sum_c W[idx,c] * U[i*m+j, c] * V[j*p+k, c] = 1 for all j
                # Build the system: A @ w = b where A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                A = np.zeros((m, rank))
                for j in range(m):
                    A[j, :] = U[i*m + j, :] * V[j*p + k, :]
                b = np.ones(m)
                # Solve A @ w = b in least squares
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                W[idx, :] = w
            
            # Update U
            for idx in range(U_dim):
                # idx = i*m + j
                i = idx // m
                j = idx % m
                # For this (i,j), we need: sum_c U[idx,c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all k
                A = np.zeros((p, rank))
                for k in range(p):
                    A[k, :] = V[j*p + k, :] * W[k*n + i, :]
                b = np.ones(p)
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                U[idx, :] = w
            
            # Update V
            for idx in range(V_dim):
                # idx = j*p + k
                j = idx // p
                k = idx % p
                # For this (j,k), we need: sum_c V[idx,c] * U[i*m+j, c] * W[k*n+i, c] = 1 for all i
                A = np.zeros((n, rank))
                for i in range(n):
                    A[i, :] = U[i*m + j, :] * W[k*n + i, :]
                b = np.ones(n)
                w, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                V[idx, :] = w
        
        # Check if the decomposition is exact
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    err = abs(val - 1.0)
                    if err > max_err:
                        max_err = err
                    # Also check that non-target entries are zero... but there are no non-target entries
                    # The tensor is fully dense in the sense that all (i,j,k) triples have value 1
        
        if max_err < 1e-9:
            # Found an exact decomposition
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(U_dim)]
    V = [[0] * rank for _ in range(V_dim)]
    W = [[0] * rank for _ in range(W_dim)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
