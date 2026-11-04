"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 8]

import numpy as np

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We construct a decomposition of rank 44
    # Rank = 16 (for j=0) + 4 * 7 (for j=1,2 split into 4 blocks of p=2)
    rank = 44
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Helper to add a rank-1 term
    def add_term(u_idx, v_idx, w_idx, val_u, val_v, val_w, col):
        U[u_idx, col] += val_u
        V[v_idx, col] += val_v
        W[w_idx, col] += val_w

    # Block 1: j=0, M_{2,1,8}
    # Trivial decomposition: rank 16
    col = 0
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            add_term(u_idx, v_idx, w_idx, 1, 1, 1, col)
            col += 1
            
    # Block 2: j=1,2, M_{2,2,8}
    # Split p=8 into 4 blocks of size 2
    # For each block, use Strassen M_{2,2,2} rank 7
    # Local indices: i in [0,1], j_loc in [0,1], k_loc in [0,1]
    # Global indices: i, j = j_loc + 1, k = 2*s + k_loc
    
    # Strassen coefficients for M_{2,2,2}
    # U, V, W are 2x2 matrices (flattened)
    # We define 7 terms
    # Each term has U_coeffs, V_coeffs, W_coeffs
    # U_coeffs: dict (i,j) -> val
    # V_coeffs: dict (j,k) -> val
    # W_coeffs: dict (k,i) -> val
    
    strassen_terms = [
        # r=1
        (
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}, 
            {(0,0): 1, (1,1): 1}
        ),
        # r=2
        (
            {(1,0): 1, (1,1): 1}, 
            {(0,0): 1}, 
            {(0,1): 1, (1,1): -1}
        ),
        # r=3
        (
            {(0,0): 1}, 
            {(0,1): 1, (1,1): -1}, 
            {(1,0): 1, (1,1): 1}
        ),
        # r=4
        (
            {(1,1): 1}, 
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1, (0,1): 1}
        ),
        # r=5
        (
            {(0,0): 1, (0,1): 1}, 
            {(1,1): 1}, 
            {(0,0): -1, (1,0): 1}
        ),
        # r=6
        (
            {(1,0): 1, (0,0): -1}, 
            {(0,0): 1, (0,1): 1}, 
            {(1,1): 1}
        ),
        # r=7
        (
            {(0,1): 1, (1,1): -1}, 
            {(1,0): 1, (1,1): 1}, 
            {(0,0): 1}
        ),
    ]
    
    for s in range(4):
        for term in strassen_terms:
            u_coeffs, v_coeffs, w_coeffs = term
            # Process U
            for (li, lj), val in u_coeffs.items():
                # li is local i, lj is local j
                # global i = li, global j = lj + 1
                u_idx = li * m + (lj + 1)
                U[u_idx, col] += val
            # Process V
            for (lj, lk), val in v_coeffs.items():
                # lj is local j, lk is local k
                # global j = lj + 1, global k = 2*s + lk
                v_idx = (lj + 1) * p + (2*s + lk)
                V[v_idx, col] += val
            # Process W
            for (lk, li), val in w_coeffs.items():
                # lk is local k, li is local i
                # global k = 2*s + lk, global i = li
                w_idx = (2*s + lk) * n + li
                W[w_idx, col] += val
            col += 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
