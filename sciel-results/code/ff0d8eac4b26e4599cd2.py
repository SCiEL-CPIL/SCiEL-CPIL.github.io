"""Program for 3x3x3 matrix multiplication tensor decomposition."""
import numpy as np
import time

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 3x3x3 matrix multiplication.
    Uses a rank-26 decomposition based on Strassen's algorithm for the 2x2 core
    and standard multiplication for the remaining border terms.
    """
    n, m, p = DIMENSIONS
    
    # Rank 26 decomposition
    R = 26
    
    U = np.zeros((n * m, R))
    V = np.zeros((m * p, R))
    W = np.zeros((p * n, R))
    
    # Helper to get flattened indices
    # U index: i * m + j
    # V index: j * p + k
    # W index: k * n + i (as per task contract W[k*n+i, r])
    
    def idx_u(i, j):
        return i * m + j
        
    def idx_v(j, k):
        return j * p + k
        
    def idx_w(k, i):
        return k * n + i
        
    # --- Strassen Part (7 terms) ---
    # Indices i, j, k in {0, 1}
    
    # m1 = (a00 + a11)(b00 + b11)
    r = 0
    U[idx_u(0, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1  # c00 += m1
    W[idx_w(1, 1), r] = 1  # c11 += m1
    
    # m2 = (a10 + a11)b00
    r = 1
    U[idx_u(1, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1  # c10 += m2 (Note: W index is k*n+i. Here k=0, i=1 -> 0*3+1=1)
    W[idx_w(1, 1), r] = -1 # c11 -= m2 (k=1, i=1 -> 1*3+1=4? Wait. c11 is i=1, k=1. W[k*n+i] = W[1*3+1]=W[4].
                           # Let's re-verify the W index convention.
                           # Task: W[k*n+i, r].
                           # Matrix C is n x p. C[i, k].
                           # In the tensor T[i, j, k], the output index is (i, k).
                           # The problem states: "output factor follows the official trace convention W[k*n+i,r]".
                           # This means the index into the W vector for the term contributing to C[i,k] is k*n + i.
                           
                           # Let's re-check the Strassen terms with W[k*n+i].
                           # m1: contributes to C[0,0] and C[1,1].
                           # C[0,0] -> i=0, k=0 -> W[0*3+0] = W[0].
                           # C[1,1] -> i=1, k=1 -> W[1*3+1] = W[4].
                           
                           # My previous code in inspiration had:
                           # W[idx_w(1, 1), r] = 1. idx_w(k, i) = k*n + i.
                           # So idx_w(1, 1) = 1*3 + 1 = 4.
                           # But C[1,1] is i=1, k=1.
                           # So W[4] is correct for C[1,1].
                           
                           # Let's check m2: contributes to C[1,0] and C[1,1].
                           # C[1,0] -> i=1, k=0 -> W[0*3+1] = W[1].
                           # C[1,1] -> i=1, k=1 -> W[1*3+1] = W[4].
                           
                           # In the inspiration code:
                           # W[idx_w(0, 1), r] = 1. idx_w(0, 1) = 0*3+1 = 1. Correct for C[1,0].
                           # W[idx_w(1, 1), r] = -1. idx_w(1, 1) = 1*3+1 = 4. Correct for C[1,1].
    
    # Let's re-verify all Strassen terms with W[k*n+i]
    
    # m1 = (a00 + a11)(b00 + b11) -> C00, C11
    r = 0
    U[idx_u(0, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1  # W[0] for C[0,0]
    W[idx_w(1, 1), r] = 1  # W[4] for C[1,1]
    
    # m2 = (a10 + a11)b00 -> C10, C11
    r = 1
    U[idx_u(1, 0), r] = 1
    U[idx_u(1, 1), r] = 1
    V[idx_v(0, 0), r] = 1
    W[idx_w(0, 1), r] = 1  # W[1] for C[1,0]
    W[idx_w(1, 1), r] = -1 # W[4] for C[1,1]
    
    # m3 = a00(b01 - b11) -> C01, C11
    r = 2
    U[idx_u(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1
    V[idx_v(1, 1), r] = -1
    W[idx_w(1, 0), r] = 1  # W[3] for C[0,1] (i=0, k=1 -> 1*3+0=3)
    W[idx_w(1, 1), r] = 1  # W[4] for C[1,1]
    
    # m4 = a11(b10 - b00) -> C00, C10
    r = 3
    U[idx_u(1, 1), r] = 1
    V[idx_v(1, 0), r] = 1
    V[idx_v(0, 0), r] = -1
    W[idx_w(0, 0), r] = 1  # W[0] for C[0,0]
    W[idx_w(0, 1), r] = 1  # W[1] for C[1,0]
    
    # m5 = (a00 + a01)b11 -> C00, C01
    r = 4
    U[idx_u(0, 0), r] = 1
    U[idx_u(0, 1), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = -1 # W[0] for C[0,0]
    W[idx_w(1, 0), r] = 1  # W[3] for C[0,1]
    
    # m6 = (a10 - a00)(b00 + b01) -> C11
    r = 5
    U[idx_u(1, 0), r] = 1
    U[idx_u(0, 0), r] = -1
    V[idx_v(0, 0), r] = 1
    V[idx_v(0, 1), r] = 1
    W[idx_w(1, 1), r] = 1  # W[4] for C[1,1]
    
    # m7 = (a01 - a11)(b10 + b11) -> C00
    r = 6
    U[idx_u(0, 1), r] = 1
    U[idx_u(1, 1), r] = -1
    V[idx_v(1, 0), r] = 1
    V[idx_v(1, 1), r] = 1
    W[idx_w(0, 0), r] = 1  # W[0] for C[0,0]
    
    # --- Standard Part (19 terms) ---
    # Indices (i, j, k) where at least one is 2.
    
    col = 7
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Skip if all indices are in {0, 1}
                if i < 2 and j < 2 and k < 2:
                    continue
                
                U[idx_u(i, j), col] = 1
                V[idx_v(j, k), col] = 1
                W[idx_w(k, i), col] = 1
                col += 1
                
    # Verify the decomposition
    # T_recon[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    U_reshaped = U.reshape((n, m, R))
    V_reshaped = V.reshape((m, p, R))
    W_reshaped = W.reshape((p, n, R))
    
    # Note: W is indexed by k*n+i. In reshaped W (p, n, R), the index is [k, i, r].
    # So W_reshaped[k, i, r] corresponds to W[k*n+i, r].
    # The einsum 'ijr, jkr, kir -> ijk' matches:
    # U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_reshaped, V_reshaped, W_reshaped)
    
    # Target tensor T is 1 for all i, j, k
    T_target = np.ones((n, m, p))
    
    if not np.allclose(T_recon, T_target, atol=1e-9):
        # Fallback to schoolbook if something went wrong
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[idx_u(i, j), col] = 1
                    V[idx_v(j, k), col] = 1
                    W[idx_w(k, i), col] = 1
                    col += 1
        R = n * m * p

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
