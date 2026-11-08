"""Tensor decomposition for matrix multiplication 3x3 * 3x8."""
import time
import numpy as np

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_c U[:,c] V^T[:,c] W^T[:,c] = T where T[i*m+j, j*p+k, k*n+i] = 1."""
    rank = U.shape[1]
    # Build the tensor T as a 3D array
    # T is indexed by (a, b, c) where:
    #   a = i*m + j  (index into A)
    #   b = j'*p + k (index into B) -- wait, the tensor for matmul has indices (i,j,k)
    #   The decomposition says: T_{i,j,k} = sum_c U_{i*m+j, c} * V_{j*p+k, c} * W_{k*n+i, c}
    #   So we need to check that for each (i,j,k), the sum equals 1.
    #   Equivalently, build the tensor and compare.
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    
    # Compute the tensor from the decomposition
    # Reshape U, V, W to use einsum
    U_3d = U.reshape(n * m, rank)
    V_3d = V.reshape(m * p, rank)
    W_3d = W.reshape(p * n, rank)
    
    # T_comp[a, b, c] = sum_r U[a, r] * V[b, r] * W[c, r]
    T_comp = np.einsum('ar,br,cr->abc', U_3d, V_3d, W_3d)
    
    return np.allclose(T_comp, T, atol=tol)

def _build_strassen_3x3(n, m, p):
    """
    Try to construct a decomposition of rank less than n*m*p using a Strassen-like approach.
    For 3x3 * 3x8, we can't directly use standard Strassen (which is for 2x2 or 4x4).
    However, we can attempt a simple decomposition.
    
    Actually, for the purpose of this task, let's try to find a lower rank decomposition
    using a simple heuristic or a known lower bound approach.
    
    Since we don't have a guaranteed lower rank decomposition for 3x3 * 3x8 that's easily
    constructible, we will fall back to schoolbook unless we find a better one.
    """
    # For now, this is a placeholder that returns None to indicate no better decomposition found.
    return None

def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS
    
    # Try to find a better decomposition first
    better = _build_strassen_3x3(n, m, p)
    if better is not None:
        U, V, W = better
        if _verify(U, V, W, n, m, p):
            # Convert to lists of lists with integer/half-integer entries
            U_list = [[int(round(x * 2)) / 2 for x in row] for row in U.tolist()]
            V_list = [[int(round(x * 2)) / 2 for x in row] for row in V.tolist()]
            W_list = [[int(round(x * 2)) / 2 for x in row] for row in W.tolist()]
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fall back to schoolbook
    U, V, W = _build_schoolbook(n, m, p)
    U_list = [[int(round(x * 2)) / 2 for x in row] for row in U.tolist()]
    V_list = [[int(round(x * 2)) / 2 for x in row] for row in V.tolist()]
    W_list = [[int(round(x * 2)) / 2 for x in row] for row in W.tolist()]
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
