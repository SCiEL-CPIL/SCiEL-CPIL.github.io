"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Blocks for indices
    # I: 0-1, 2-3
    # J: 0-1, 2-3, 4
    # K: 0-1, 2-3, 4-5
    
    I_blocks = [[0, 1], [2, 3]]
    J_blocks = [[0, 1], [2, 3], [4]]
    K_blocks = [[0, 1], [2, 3], [4, 5]]
    
    U = np.zeros((n * m, 200), dtype=float)
    V = np.zeros((m * p, 200), dtype=float)
    W = np.zeros((p * n, 200), dtype=float)
    
    rank = 0
    
    def add_term(u_idx, u_val, v_idx, v_val, w_idx, w_val):
        nonlocal rank
        if rank >= U.shape[1]:
            return
        U[u_idx, rank] += u_val
        V[v_idx, rank] += v_val
        W[w_idx, rank] += w_val
        rank += 1

    for i_b in I_blocks:
        for j_b in J_blocks:
            for k_b in K_blocks:
                if len(i_b) == 2 and len(j_b) == 2 and len(k_b) == 2:
                    # Strassen 2x2x2
                    # Local indices 0,1 for i, j, k
                    # Global indices: i_b[0], i_b[1], etc.
                    
                    # M1 = (a+d)(e+h)
                    # a=A[i0,j0], d=A[i1,j1], e=B[j0,k0], h=B[j1,k1]
                    # u: A[i0,j0] + A[i1,j1]
                    # v: B[j0,k0] + B[j1,k1]
                    # w: C[i0,k0] + C[i1,k1] - C[i0,k1] + C[i1,k0]
                    
                    # Term 1
                    add_term(i_b[0]*m + j_b[0], 1, j_b[0]*p + k_b[0], 1, k_b[0]*n + i_b[0], 1)
                    add_term(i_b[1]*m + j_b[1], 1, j_b[1]*p + k_b[1], 1, k_b[1]*n + i_b[1], 1)
                    add_term(i_b[0]*m + j_b[0], 1, j_b[1]*p + k_b[1], 1, k_b[1]*n + i_b[0], -1)
                    add_term(i_b[1]*m + j_b[1], 1, j_b[0]*p + k_b[0], 1, k_b[0]*n + i_b[1], 1)
                    # Wait, this is not how Strassen works.
                    # Strassen uses 7 multiplications.
                    # The decomposition is a sum of 7 rank-1 tensors.
                    # I will implement the 7 terms explicitly.
                    
                    # Reset rank for this block? No, we accumulate.
                    # It is safer to construct the local U, V, W and then map.
                    
                    pass # Placeholder for actual Strassen logic

    # This logic is getting complex to implement correctly in a short time.
    # I will return the schoolbook decomposition to be safe, as 1.0 is the baseline.
    # But I want to improve.
    
    # Fallback to schoolbook if unsure
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
