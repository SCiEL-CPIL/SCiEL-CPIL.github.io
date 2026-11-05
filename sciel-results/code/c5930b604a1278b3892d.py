"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W):
    """Verify that U, V, W form a valid rank-r decomposition of the multiplication tensor.
    
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    The tensor is T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    n, m, p = DIMENSIONS
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    r = U_arr.shape[1]
    
    # Reshape to 3D tensors
    U_3d = U_arr.reshape(n, m, r)
    V_3d = V_arr.reshape(m, p, r)
    W_3d = W_arr.reshape(p, n, r)
    
    # Compute the tensor: T[i,j,k] = sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
    # Use einsum for efficiency
    T = np.einsum('ijr,jkr,kir->ijk', U_3d, V_3d, W_3d)
    
    # Expected tensor: T[i,j,k] = 1 if all indices are valid (which they always are for matrix mult)
    # Actually, the multiplication tensor for A(i x j) * B(j x k) = C(i x k)
    # T[i,j,k] = 1 for all i, j, k (since each output C[i,k] = sum_j A[i,j]*B[j,k])
    T_expected = np.ones((n, m, p), dtype=np.float64)
    
    return np.allclose(T, T_expected, atol=1e-10)

def construct():
    n, m, p = DIMENSIONS
    rank = n * m * p
    
    # Schoolbook decomposition
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
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
