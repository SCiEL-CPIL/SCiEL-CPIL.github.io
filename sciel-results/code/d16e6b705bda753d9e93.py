import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def _schoolbook(n, m, p):
    """
    Returns U, V, W for n x m * m x p -> n x p.
    U: (n*m, r), V: (m*p, r), W: (p*n, r)
    """
    r = n * m * p
    U = np.zeros((n * m, r))
    V = np.zeros((m * p, r))
    W = np.zeros((p * n, r))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _strassen_2x2x2():
    """
    Returns U, V, W for 2 x 2 * 2 x 2 -> 2 x 2.
    U: (4, 7), V: (4, 7), W: (4, 7)
    """
    n = m = p = 2
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1
    
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1
    
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[3, 3] = 1
    
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[1, 4] = 1
    
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[2, 5] = 1
    
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[3, 6] = 1
    
    return U, V, W

def construct():
    n, m, p = DIMENSIONS # 3, 5, 7
    
    # Partition sizes
    a_rows = [2, 1]
    a_cols = [2, 2, 1] # b_rows
    b_cols = [2, 2, 2, 1]
    
    # Check sums
    assert sum(a_rows) == n
    assert sum(a_cols) == m
    assert sum(b_cols) == p
    
    total_rank = 0
    # Pre-calculate total rank to allocate memory
    # We'll just append to lists and stack at the end
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Mapping of local indices to global indices
    # Global U index: i_global * m + j_global
    # Global V index: j_global * p + k_global
    # Global W index: k_global * n + i_global
    
    a_row_offsets = [0]
    for size in a_rows[:-1]:
        a_row_offsets.append(a_row_offsets[-1] + size)
        
    a_col_offsets = [0]
    for size in a_cols[:-1]:
        a_col_offsets.append(a_col_offsets[-1] + size)
        
    b_col_offsets = [0]
    for size in b_cols[:-1]:
        b_col_offsets.append(b_col_offsets[-1] + size)

    for i_idx, a_r in enumerate(a_rows):
        for k_idx, b_c in enumerate(b_cols):
            # C_block is a_r x b_c
            for j_idx, a_c in enumerate(a_cols):
                # A_block is a_r x a_c
                # B_block is a_c x b_c
                
                if a_r == 2 and a_c == 2 and b_c == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(a_r, a_c, b_c)
                
                r_loc = U_loc.shape[1]
                
                # Create global columns
                U_glob = np.zeros((n * m, r_loc))
                V_glob = np.zeros((m * p, r_loc))
                W_glob = np.zeros((p * n, r_loc))
                
                # Fill U
                for r in range(r_loc):
                    for i in range(a_r):
                        for j in range(a_c):
                            val = U_loc[i * a_c + j, r]
                            if val != 0:
                                gi = a_row_offsets[i_idx] + i
                                gj = a_col_offsets[j_idx] + j
                                U_glob[gi * m + gj, r] = val
                                
                # Fill V
                for r in range(r_loc):
                    for j in range(a_c):
                        for k in range(b_c):
                            val = V_loc[j * b_c + k, r]
                            if val != 0:
                                gj = a_col_offsets[j_idx] + j
                                gk = b_col_offsets[k_idx] + k
                                V_glob[gj * p + gk, r] = val
                                
                # Fill W
                for r in range(r_loc):
                    for i in range(a_r):
                        for k in range(b_c):
                            val = W_loc[k * a_r + i, r]
                            if val != 0:
                                gi = a_row_offsets[i_idx] + i
                                gk = b_col_offsets[k_idx] + k
                                W_glob[gk * n + gi, r] = val
                                
                U_cols.append(U_glob)
                V_cols.append(V_glob)
                W_cols.append(W_glob)
                
    U_final = np.hstack(U_cols)
    V_final = np.hstack(V_cols)
    W_final = np.hstack(W_cols)
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}

# EVOLVE-BLOCK-END
