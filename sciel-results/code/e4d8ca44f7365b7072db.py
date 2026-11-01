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
        U[0][base_col] = 1
        U[3][base_col] = 1
        V[k1][base_col] = 1
        V[p + k2][base_col] = 1
        W[k1 * n + 0][base_col] = 1
        W[k2 * n + 1][base_col] = 1
        
        # M2 = (A[1,0] + A[1,1]) * B[0,k1]
        U[2][base_col + 1] = 1
        U[3][base_col + 1] = 1
        V[k1][base_col + 1] = 1
        W[k1 * n + 1][base_col + 1] = 1
        W[k2 * n + 1][base_col + 1] = -1
        
        # M3 = A[0,0] * (B[0,k2] - B[1,k2])
        U[0][base_col + 2] = 1
        V[k2][base_col + 2] = 1
        V[p + k2][base_col + 2] = -1
        W[k2 * n + 0][base_col + 2] = 1
        W[k2 * n + 1][base_col + 2] = 1
        
        # M4 = A[1,1] * (B[1,k1] - B[0,k1])
        U[3][base_col + 3] = 1
        V[p + k1][base_col + 3] = 1
        V[k1][base_col + 3] = -1
        W[k1 * n + 0][base_col + 3] = 1
        W[k1 * n + 1][base_col + 3] = 1
        
        # M5 = (A[0,0] + A[0,1]) * B[1,k2]
        U[0][base_col + 4] = 1
        U[1][base_col + 4] = 1
        V[p + k2][base_col + 4] = 1
        W[k1 * n + 0][base_col + 4] = -1
        W[k2 * n + 0][base_col + 4] = 1
        
        # M6 = (A[1,0] - A[0,0]) * (B[0,k1] + B[0,k2])
        U[2][base_col + 5] = 1
        U[0][base_col + 5] = -1
        V[k1][base_col + 5] = 1
        V[k2][base_col + 5] = 1
        W[k2 * n + 1][base_col + 5] = 1
        
        # M7 = (A[0,1] - A[1,1]) * (B[1,k1] + B[1,k2])
        U[1][base_col + 6] = 1
        U[3][base_col + 6] = -1
        V[p + k1][base_col + 6] = 1
        V[p + k2][base_col + 6] = 1
        W[k1 * n + 0][base_col + 6] = 1
    
    # Verify using vectorized NumPy check
    if _verify_decomposition_fast(U, V, W, n, m, p):
        return {"U": U, "V": V, "W": W}
    else:
        return None


def _verify_decomposition_fast(U, V, W, n, m, p, num_tests=3):
    """Verify that the decomposition correctly computes matrix multiplication using vectorized NumPy."""
    import numpy as np
    
    rank = len(U[0])
    U_np = np.array(U, dtype=np.float64)  # (n*m, rank)
    V_np = np.array(V, dtype=np.float64)  # (m*p, rank)
    W_np = np.array(W, dtype=np.float64)  # (p*n, rank)
    
    for _ in range(num_tests):
        # Random A (n x m) and B (m x p)
        A = np.random.rand(n, m)
        B = np.random.rand(m, p)
        C_expected = A @ B
        
        # Compute C using the decomposition:
        # C[i,k] = sum_j A[i,j] * B[j,k]
        # Each rank-r term contributes: U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k] * W[k*n+i, r]
        # Sum over j and r:
        # C[i,k] = sum_r W[k*n+i, r] * (sum_j U[i*m+j, r] * A[i,j] * V[j*p+k, r] * B[j,k])
        
        # Reshape U to (n, m, rank), V to (m, p, rank), W to (p, n, rank)
        U_reshaped = U_np.reshape(n, m, rank)
        V_reshaped = V_np.reshape(m, p, rank)
        W_reshaped = W_np.reshape(p, n, rank)
        
        # For each rank component r:
        # term_r[i,k] = W_reshaped[k,i,r] * sum_j (U_reshaped[i,j,r] * A[i,j] * V_reshaped[j,k,r] * B[j,k])
        
        C_computed = np.zeros((n, p))
        for r in range(rank):
            # U_r: (n, m), V_r: (m, p), W_r: (p, n)
            U_r = U_reshaped[:, :, r]  # (n, m)
            V_r = V_reshaped[:, :, r]  # (m, p)
            W_r = W_reshaped[:, :, r]  # (p, n)
            
            # intermediate[i,k] = sum_j U_r[i,j] * A[i,j] * V_r[j,k] * B[j,k]
            # = (A * U_r) @ (B * V_r)  element-wise product then matmul
            A_U = A * U_r  # (n, m)
            B_V = B * V_r  # (m, p)
            term = A_U @ B_V  # (n, p)
            
            # Multiply by W_r: term[i,k] *= W_r[k,i]
            C_computed += term * W_r.T  # W_r.T is (n, p)
        
        if not np.allclose(C_computed, C_expected, atol=1e-10):
            return False
    
    return True
# EVOLVE-BLOCK-END
