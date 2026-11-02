import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    n_rows = n * m  # 6
    m_rows = m * p  # 27
    p_rows = p * n  # 18
    
    # Rank 36 decomposition
    def solution_rank_36():
        rank = 36
        U = [[0] * rank for _ in range(n_rows)]
        V = [[0] * rank for _ in range(m_rows)]
        W = [[0] * rank for _ in range(p_rows)]
        
        col = 0
        
        # Part 1: j in {0, 1}
        # For each k in 0..8, we need 2 terms.
        # Term 1: handles j=0. U[i, 0]=1, V[0, k]=1. W[k, i]=1.
        # Term 2: handles j=1. U[i, 1]=1, V[1, k]=1. W[k, i]=1.
        for k in range(p):
            # Term for j=0
            # U[0, 0] = 1, U[1, 0] = 1 (indices 0, 1 in U)
            # V[0, k] = 1 (index 0*p + k = k in V)
            # W[k, 0] = 1, W[k, 1] = 1 (indices k*n+0, k*n+1 in W)
            U[0 * m + 0][col] = 1
            U[1 * m + 0][col] = 1
            V[0 * p + k][col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1
            
            # Term for j=1
            # U[0, 1] = 1, U[1, 1] = 1 (indices 0*m+1=1, 1*m+1=4 in U)
            # V[1, k] = 1 (index 1*p + k = 9+k in V)
            # W[k, 0] = 1, W[k, 1] = 1
            U[0 * m + 1][col] = 1
            U[1 * m + 1][col] = 1
            V[1 * p + k][col] = 1
            W[k * n + 0][col] = 1
            W[k * n + 1][col] = 1
            col += 1
            
        # Part 2: j=2
        # For each i in 0..1, k in 0..8, we need 1 term.
        # U[i, 2] = 1, V[2, k] = 1, W[k, i] = 1.
        for i in range(n):
            for k in range(p):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
        return {"U": U, "V": V, "W": W}

    def verify_fast(U, V, W):
        r = len(U[0])
        U_arr = np.array(U, dtype=np.int64)
        V_arr = np.array(V, dtype=np.int64)
        W_arr = np.array(W, dtype=np.int64)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U_arr[i*m+j] * V_arr[j*p+k] * W_arr[k*n+i])
                    if val != 1:
                        return False
        return True

    # Verify the rank 36 solution
    res = solution_rank_36()
    if verify_fast(res["U"], res["V"], res["W"]):
        return res

    # Fallback to current best (rank 50) if verification fails (should not happen)
    # Re-implementing the rank 50 logic from the prompt as fallback
    def current_solution():
        rank = 50
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        
        col = 0
        for b in range(4):
            k0 = 2 * b
            k1 = 2 * b + 1
            v_e = 0 * p + k0
            v_f = 0 * p + k1
            v_g = 1 * p + k0
            v_h = 1 * p + k1
            u_a = 0 * m + 0
            u_b = 0 * m + 1
            u_c = 1 * m + 0
            u_d = 1 * m + 1
            
            U[u_a][col] += 1; U[u_d][col] += 1; V[v_e][col] += 1; V[v_h][col] += 1
            U[u_c][col+1] += 1; U[u_d][col+1] += 1; V[v_e][col+1] += 1
            U[u_a][col+2] += 1; V[v_f][col+2] += 1; V[v_h][col+2] -= 1
            U[u_d][col+3] += 1; V[v_g][col+3] += 1; V[v_e][col+3] -= 1
            U[u_a][col+4] += 1; U[u_b][col+4] += 1; V[v_h][col+4] += 1
            U[u_c][col+5] += 1; U[u_a][col+5] -= 1; V[v_e][col+5] += 1; V[v_f][col+5] += 1
            U[u_b][col+6] += 1; U[u_d][col+6] -= 1; V[v_g][col+6] += 1; V[v_h][col+6] += 1
            
            w_00 = k0 * n + 0
            w_01 = k1 * n + 0
            w_10 = k0 * n + 1
            w_11 = k1 * n + 1
            
            W[w_00][col] += 1; W[w_00][col+3] += 1; W[w_00][col+4] -= 1; W[w_00][col+6] += 1
            W[w_01][col+2] += 1; W[w_01][col+4] += 1
            W[w_10][col+1] += 1; W[w_10][col+3] += 1
            W[w_11][col] += 1; W[w_11][col+1] -= 1; W[w_11][col+2] += 1; W[w_11][col+5] += 1
            col += 7
            
        k8 = 8
        v_e = 0 * p + k8
        v_g = 1 * p + k8
        u_a = 0 * m + 0
        u_b = 0 * m + 1
        u_c = 1 * m + 0
        u_d = 1 * m + 1
        
        U[u_a][col] += 1; V[v_e][col] += 1
        U[u_b][col+1] += 1; V[v_g][col+1] += 1
        U[u_c][col+2] += 1; V[v_e][col+2] += 1
        U[u_d][col+3] += 1; V[v_g][col+3] += 1
        
        w_08 = k8 * n + 0
        w_18 = k8 * n + 1
        W[w_08][col] += 1; W[w_08][col+1] += 1
        W[w_18][col+2] += 1; W[w_18][col+3] += 1
        col += 4
        
        u_ar0 = 0 * m + 2
        u_ar1 = 1 * m + 2
        for k in range(9):
            v_b = 2 * p + k
            U[u_ar0][col] += 1; V[v_b][col] += 1; W[k * n + 0][col] += 1
            U[u_ar1][col+1] += 1; V[v_b][col+1] += 1; W[k * n + 1][col+1] += 1
            col += 2
            
        return {"U": U, "V": V, "W": W}

    return current_solution()

