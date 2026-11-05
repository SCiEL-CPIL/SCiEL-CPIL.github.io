"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Block decomposition:
    # i: 0,1 (2x), 2 (1x)
    # j: 0,1 (2x), 2,3 (2x)
    # k: 0,1 (2x), 2,3 (2x), 4,5 (2x)
    
    # We will accumulate U, V, W
    # Rank estimate: 6 * 7 (Strassen) + 6 * 4 (Schoolbook 1x2x2) = 42 + 24 = 66
    
    U = np.zeros((n * m, 66), dtype=float)
    V = np.zeros((m * p, 66), dtype=float)
    W = np.zeros((p * n, 66), dtype=float)
    
    col = 0
    
    def add_strassen(i0, j0, k0, start_col):
        # Adds 7 columns for a 2x2x2 block at i0, j0, k0
        # Strassen terms:
        # M1 = (a+d)(e+h)
        # M2 = (c+d)e
        # M3 = a(f-h)
        # M4 = d(g-e)
        # M5 = (a+b)h
        # M6 = (c-a)(e+f)
        # M7 = (b-d)(g+h)
        
        # Indices
        a_idx = i0 * m + j0
        b_idx = i0 * m + (j0 + 1)
        c_idx = (i0 + 1) * m + j0
        d_idx = (i0 + 1) * m + (j0 + 1)
        
        e_idx = j0 * p + k0
        f_idx = j0 * p + (k0 + 1)
        g_idx = (j0 + 1) * p + k0
        h_idx = (j0 + 1) * p + (k0 + 1)
        
        # Output indices (k*n + i)
        # c00: k0*n + i0
        # c01: (k0+1)*n + i0
        # c10: k0*n + (i0+1)
        # c11: (k0+1)*n + (i0+1)
        
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        w10 = k0 * n + (i0 + 1)
        w11 = (k0 + 1) * n + (i0 + 1)
        
        # M1
        U[a_idx, start_col] = 1
        U[d_idx, start_col] = 1
        V[e_idx, start_col] = 1
        V[h_idx, start_col] = 1
        W[w00, start_col] = 1
        W[w01, start_col] = 1
        W[w10, start_col] = 1
        W[w11, start_col] = -1
        col = start_col + 1
        
        # M2: (c+d)e
        U[c_idx, col] = 1
        U[d_idx, col] = 1
        V[e_idx, col] = 1
        W[w10, col] = 1
        col += 1
        
        # M3: a(f-h)
        U[a_idx, col] = 1
        V[f_idx, col] = 1
        V[h_idx, col] = -1
        W[w01, col] = 1
        col += 1
        
        # M4: d(g-e)
        U[d_idx, col] = 1
        V[g_idx, col] = 1
        V[e_idx, col] = -1
        W[w00, col] = 1
        W[w10, col] = 1
        col += 1
        
        # M5: (a+b)h
        U[a_idx, col] = 1
        U[b_idx, col] = 1
        V[h_idx, col] = 1
        W[w01, col] = 1
        W[w11, col] = -1
        col += 1
        
        # M6: (c-a)(e+f)
        U[c_idx, col] = 1
        U[a_idx, col] = -1
        V[e_idx, col] = 1
        V[f_idx, col] = 1
        W[w11, col] = 1
        col += 1
        
        # M7: (b-d)(g+h)
        U[b_idx, col] = 1
        U[d_idx, col] = -1
        V[g_idx, col] = 1
        V[h_idx, col] = 1
        W[w00, col] = 1
        W[w11, col] = 1
        col += 1
        
        return col

    def add_schoolbook_1x2x2(i0, j0, k0, start_col):
        # Adds 4 columns for a 1x2x2 block at i0, j0, k0
        # A is 1x2, B is 2x2, C is 1x2
        # C00 = A00*B00 + A01*B10
        # C01 = A00*B01 + A01*B11
        
        a0_idx = i0 * m + j0
        a1_idx = i0 * m + (j0 + 1)
        
        b00_idx = j0 * p + k0
        b01_idx = j0 * p + (k0 + 1)
        b10_idx = (j0 + 1) * p + k0
        b11_idx = (j0 + 1) * p + (k0 + 1)
        
        w00 = k0 * n + i0
        w01 = (k0 + 1) * n + i0
        
        # Term 1: A00*B00 -> C00
        U[a0_idx, start_col] = 1
        V[b00_idx, start_col] = 1
        W[w00, start_col] = 1
        col = start_col + 1
        
        # Term 2: A01*B10 -> C00
        U[a1_idx, col] = 1
        V[b10_idx, col] = 1
        W[w00, col] = 1
        col += 1
        
        # Term 3: A00*B01 -> C01
        U[a0_idx, col] = 1
        V[b01_idx, col] = 1
        W[w01, col] = 1
        col += 1
        
        # Term 4: A01*B11 -> C01
        U[a1_idx, col] = 1
        V[b11_idx, col] = 1
        W[w01, col] = 1
        col += 1
        
        return col

    # Blocks:
    # i=0,1 (Strassen part)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            col = add_strassen(0, j0, k0, col)
            
    # i=2 (Schoolbook part)
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            col = add_schoolbook_1x2x2(2, j0, k0, col)
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
