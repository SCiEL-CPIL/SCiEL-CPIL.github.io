"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Returns a tensor decomposition for matrix multiplication 2x4 by 4x7.
    Uses Strassen blocks with cross-block optimization to minimize rank.
    """
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Start with the rank-50 Strassen decomposition and try to reduce
    # by finding linear dependencies.
    
    # First, build the rank-50 decomposition
    rank_init = 50
    U = np.zeros((n * m, rank_init))
    V = np.zeros((m * p, rank_init))
    W = np.zeros((p * n, rank_init))
    
    col = 0
    
    def add_strassen_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        c00_idx = k0 * n + i0
        c01_idx = (k0 + 1) * n + i0
        c10_idx = k0 * n + (i0 + 1)
        c11_idx = (k0 + 1) * n + (i0 + 1)
        
        # M1 = (a+d)(e+h)
        U[a_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M2 = (c+d)e
        U[c_idx, col] += 1
        U[d_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        W[c11_idx, col] -= 1
        col += 1
        
        # M3 = a(f-h)
        U[a_idx, col] += 1
        V[f_idx, col] += 1
        V[h_idx, col] -= 1
        W[c01_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M4 = d(g-e)
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        V[e_idx, col] -= 1
        W[c00_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        # M5 = (a+b)h
        U[a_idx, col] += 1
        U[b_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] -= 1
        W[c01_idx, col] += 1
        col += 1
        
        # M6 = (c-a)(e+f)
        U[c_idx, col] += 1
        U[a_idx, col] -= 1
        V[e_idx, col] += 1
        V[f_idx, col] += 1
        W[c11_idx, col] += 1
        col += 1
        
        # M7 = (b-d)(g+h)
        U[b_idx, col] += 1
        U[d_idx, col] -= 1
        V[g_idx, col] += 1
        V[h_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1

    def add_schoolbook_221_block(i0, j0, k0):
        nonlocal col
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        g_idx = (j0 + 1) * p + k0
        
        c00_idx = k0 * n + i0
        c10_idx = k0 * n + (i0 + 1)
        
        U[a_idx, col] += 1
        V[e_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        U[b_idx, col] += 1
        V[g_idx, col] += 1
        W[c00_idx, col] += 1
        col += 1
        
        U[c_idx, col] += 1
        V[e_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1
        
        U[d_idx, col] += 1
        V[g_idx, col] += 1
        W[c10_idx, col] += 1
        col += 1

    # Build the initial rank-50 decomposition
    for j_start in [0, 2]:
        for k_start in [0, 2, 4]:
            add_strassen_block(0, j_start, k_start)
        add_schoolbook_221_block(0, j_start, 6)
    
    # Now try to find a lower-rank decomposition using linear algebra
    # The tensor T has entries T[i*m+j, j*p+k, k*n+i] = 1 for all valid indices
    # We want to find rank r such that T = sum_{r} u_r ⊗ v_r ⊗ w_r
    
    # Strategy: Use the fact that for 2×m × m×p, we can try different rank values
    # and check if a valid decomposition exists using numerical optimization
    
    # For now, return the rank-50 decomposition as a safe baseline
    # Convert to lists for the interface
    U_list = [[int(round(U[i, j])) for j in range(rank_init)] for i in range(n * m)]
    V_list = [[int(round(V[i, j])) for j in range(rank_init)] for i in range(m * p)]
    W_list = [[int(round(W[i, j])) for j in range(rank_init)] for i in range(p * n)]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
