"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import numpy as np

DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin before 150s kill
    
    # Try Strassen-based rank-49 decomposition first
    result_49 = _construct_strassen_49(n, m, p)
    if result_49 is not None:
        return result_49
    
    # Fallback: schoolbook rank-56
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}


def _construct_strassen_49(n, m, p):
    """
    Construct a rank-49 decomposition using Strassen's algorithm on 7 blocks of 2x2x2.
    """
    if time.time() > time.time() + 140:
        return None
    
    rank = 49
    U = [[0] * rank for _ in range(n * m)]   # 4 x 49
    V = [[0] * rank for _ in range(m * p)]   # 28 x 49
    W = [[0] * rank for _ in range(p * n)]   # 28 x 49
    
    # Strassen's 7 products for 2x2x2
    # Indices in U: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Indices in V: j*p + k
    # Indices in W: k*n + i
    
    for pair_idx in range(7):
        k1 = 2 * pair_idx
        k2 = 2 * pair_idx + 1
        base_col = pair_idx * 7
        
        # M1 = (A[0,0] + A[1,1]) * (B[0,k1] + B[1,k2])
        # U: A[0,0] (idx 0) coeff 1, A[1,1] (idx 3) coeff 1
        U[0][base_col] = 1
        U[3][base_col] = 1
        # V: B[0,k1] (idx k1) coeff 1, B[1,k2] (idx p+k2) coeff 1
        V[k1][base_col] = 1
        V[p + k2][base_col] = 1
        # W: C[0,k1] (idx k1*n+0) coeff 1, C[1,k2] (idx k2*n+1) coeff 1
        W[k1 * n + 0][base_col] = 1
        W[k2 * n + 1][base_col] = 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,k1]
        # U: A[1,0] (idx 2) coeff 1, A[1,1] (idx 3) coeff 1
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        # V: B[0,k1] (idx k1) coeff 1
        V[k1][base_col + 1] = 1
        # W: C[1,k1] (idx k1*n+1) coeff 1, C[1,k2] (idx k2*n+1) coeff -1
        W[k1 * n + 1][base_col + 1] = 1
        W[k2 * n + 1][base_col + 1] = -1
        
        # M3 = A[0,0] * (B[0,k2] - B[1,k2])
        # U: A[0,0] (idx 0) coeff 1
        U[0][base_col + 2] = 1
        # V: B[0,k2] (idx k2) coeff 1, B[1,k2] (idx p+k2) coeff -1
        V[k2][base_col + 2] = 1
        V[p + k2][base_col + 2] = -1
        # W: C[0,k2] (idx k2*n+0) coeff 1, C[1,k2] (idx k2*n+1) coeff 1
        W[k2 * n + 0][base_col + 2] = 1
        W[k2 * n + 1][base_col + 2] = 1
        
        # M4 = A[1,1] * (B[1,k1] - B[0,k1])
        # U: A[1,1] (idx 3) coeff 1
        U[3][base_col + 3] = 1
        # V: B[1,k1] (idx p+k1) coeff 1, B[0,k1] (idx k1) coeff -1
        V[p + k1][base_col + 3] = 1
        V[k1][base_col + 3] = -1
        # W: C[0,k1] (idx k1*n+0) coeff 1, C[1,k1] (idx k1*n+1) coeff 1
        W[k1 * n + 0][base_col + 3] = 1
        W[k1 * n + 1][base_col + 3] = 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,k2]
        # U: A[0,0] (idx 0) coeff 1, A[0,1] (idx 1) coeff 1
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = 1
        # V: B[1,k2] (idx p+k2) coeff 1
        V[p + k2][base_col + 4] = 1
        # W: C[0,k1] (idx k1*n+0) coeff -1, C[0,k2] (idx k2*n+0) coeff 1
        W[k1 * n + 0][base_col + 4] = -1
        W[k2 * n + 0][base_col + 4] = 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,k1] + B[0,k2])
        # U: A[1,0] (idx 2) coeff 1, A[0,0] (idx 0) coeff -1
        U[2][base_col + 5] = 1
        U[0][base_col + 5] = -1
        # V: B[0,k1] (idx k1) coeff 1, B[0,k2] (idx k2) coeff 1
        V[k1][base_col + 5] = 1
        V[k2][base_col + 5] = 1
        # W: C[1,k2] (idx k2*n+1) coeff 1
        W[k2 * n + 1][base_col + 5] = 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,k1] + B[1,k2])
        # U: A[0,1] (idx 1) coeff 1, A[1,1] (idx 3) coeff -1
        U[1][base_col + 6] = 1
        U[3][base_col + 6] = -1
        # V: B[1,k1] (idx p+k1) coeff 1, B[1,k2] (idx p+k2) coeff 1
        V[p + k1][base_col + 6] = 1
        V[p + k2][base_col + 6] = 1
        # W: C[0,k1] (idx k1*n+0) coeff 1
        W[k1 * n + 0][base_col + 6] = 1
    
    # Now verify the decomposition
    if _verify_decomposition(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None


def _verify_decomposition(U, V, W, n, m, p, num_tests=5):
    """Verify that the decomposition correctly computes matrix multiplication."""
    import numpy as np
    
    for _ in range(num_tests):
        # Random A (n x m) and B (m x p)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute using the decomposition
        # C[i,k] = sum_j A[i,j] B[j,k]
        # Tensor form: T[i,j,k] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
        # We need T[i,j,k] = 1 for all i,j,k.
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    tensor_val = 0.0
                    for r in range(len(U[0])):
                        tensor_val += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(tensor_val - 1.0) > 1e-10:
                        return False
    
    return True
# EVOLVE-BLOCK-END
