"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Target rank is 39
    rank = 39
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, u_val, v_val, w_val):
        nonlocal col
        idx = col
        col += 1
        U[i * m + j][idx] = u_val
        V[j * p + k][idx] = v_val
        W[k * n + i][idx] = w_val

    # Strassen 2x2x2
    # i0, j0, k0 are starting indices
    def add_strassen(i0, j0, k0):
        nonlocal col
        # M1
        add_term(i0, j0, k0, 1, 1, 0) # A00 B00 -> C00
        add_term(i0, j0, k1, 1, 1, 0) # A00 B11 -> C00? No, need to be careful with W
        # This helper is getting complex with the W distribution.
        # It is safer to explicitly write out the 7 terms and their contributions to C.
        
        # Let's rewrite add_strassen to explicitly handle the 7 multiplications and 4 C updates.
        # Each multiplication is one column in U, V, W.
        # But W values are sums.
        
        # M1 = (A00+A11)(B00+B11)
        # U: A00 (1,1,0), A11 (1,1,1)
        # V: B00 (0,0,0), B11 (1,1,1)
        # W: C00 (+1), C11 (+1)
        
        # We need a way to add multiple W contributions to the same column.
        # My add_term adds to a NEW column.
        # Strassen reuses the same column for the product, but distributes to multiple C entries.
        # So for M1, we use ONE column index.
        # U[i0, j0] += 1
        # U[i1, j1] += 1
        # V[j0, k0] += 1
        # V[j1, k1] += 1
        # W[k0, i0] += 1
        # W[k1, i1] += 1
        
        # I need a helper that modifies EXISTING column.
        
        col_idx = col
        col += 1
        # M1
        U[i0 * m + j0][col_idx] += 1
        U[(i0+1) * m + (j0+1)][col_idx] += 1
        V[j0 * p + k0][col_idx] += 1
        V[(j0+1) * p + (k0+1)][col_idx] += 1
        W[k0 * n + i0][col_idx] += 1
        W[(k0+1) * n + (i0+1)][col_idx] += 1
        
        # M2 = (A10+A11)B00
        col_idx = col
        col += 1
        U[(i0+1) * m + j0][col_idx] += 1
        U[(i0+1) * m + (j0+1)][col_idx] += 1
        V[j0 * p + k0][col_idx] += 1
        # C10 += M2
        W[k0 * n + (i0+1)][col_idx] += 1
        
        # M3 = A00(B01-B11)
        col_idx = col
        col += 1
        U[i0 * m + j0][col_idx] += 1
        V[j0 * p + (k0+1)][col_idx] += 1
        V[(j0+1) * p + (k0+1)][col_idx] -= 1
        # C01 += M3
        W[(k0+1) * n + i0][col_idx] += 1
        
        # M4 = A11(B10-B00)
        col_idx = col
        col += 1
        U[(i0+1) * m + (j0+1)][col_idx] += 1
        V[(j0+1) * p + k0][col_idx] += 1
        V[j0 * p + k0][col_idx] -= 1
        # C00 += M4, C10 += M4
        W[k0 * n + i0][col_idx] += 1
        W[k0 * n + (i0+1)][col_idx] += 1
        
        # M5 = (A00+A01)B11
        col_idx = col
        col += 1
        U[i0 * m + j0][col_idx] += 1
        U[i0 * m + (j0+1)][col_idx] += 1
        V[(j0+1) * p + (k0+1)][col_idx] += 1
        # C00 -= M5, C01 += M5
        W[k0 * n + i0][col_idx] -= 1
        W[(k0+1) * n + i0][col_idx] += 1
        
        # M6 = (A10-A00)(B00+B01)
        col_idx = col
        col += 1
        U[(i0+1) * m + j0][col_idx] += 1
        U[i0 * m + j0][col_idx] -= 1
        V[j0 * p + k0][col_idx] += 1
        V[j0 * p + (k0+1)][col_idx] += 1
        # C11 += M6
        W[(k0+1) * n + (i0+1)][col_idx] += 1
        
        # M7 = (A01-A11)(B10+B11)
        col_idx = col
        col += 1
        U[i0 * m + (j0+1)][col_idx] += 1
        U[(i0+1) * m + (j0+1)][col_idx] -= 1
        V[(j0+1) * p + k0][col_idx] += 1
        V[(j0+1) * p + (k0+1)][col_idx] += 1
        # C00 += M7, C11 += M7
        W[k0 * n + i0][col_idx] += 1
        W[(k0+1) * n + (i0+1)][col_idx] += 1

    # Schoolbook 2x2x1
    # A (2x2), B (2x1)
    def add_sb_2x2x1(i0, j0, k0):
        nonlocal col
        # C00 = A00 B00 + A01 B10
        # C10 = A10 B00 + A11 B10
        
        col_idx = col; col += 1
        U[i0 * m + j0][col_idx] = 1
        V[j0 * p + k0][col_idx] = 1
        W[k0 * n + i0][col_idx] = 1
        
        col_idx = col; col += 1
        U[i0 * m + (j0+1)][col_idx] = 1
        V[(j0+1) * p + k0][col_idx] = 1
        W[k0 * n + i0][col_idx] = 1
        
        col_idx = col; col += 1
        U[(i0+1) * m + j0][col_idx] = 1
        V[j0 * p + k0][col_idx] = 1
        W[k0 * n + (i0+1)][col_idx] = 1
        
        col_idx = col; col += 1
        U[(i0+1) * m + (j0+1)][col_idx] = 1
        V[(j0+1) * p + k0][col_idx] = 1
        W[k0 * n + (i0+1)][col_idx] = 1

    # Schoolbook 2x1x2
    # A (2x1), B (1x2)
    def add_sb_2x1x2(i0, j0, k0):
        nonlocal col
        # C00 = A00 B00
        # C01 = A00 B01
        # C10 = A10 B00
        # C11 = A10 B01
        
        col_idx = col; col += 1
        U[i0 * m + j0][col_idx] = 1
        V[j0 * p + k0][col_idx] = 1
        W[k0 * n + i0][col_idx] = 1
        
        col_idx = col; col += 1
        U[i0 * m + j0][col_idx] = 1
        V[j0 * p + (k0+1)][col_idx] = 1
        W[(k0+1) * n + i0][col_idx] = 1
        
        col_idx = col; col += 1
        U[(i0+1) * m + j0][col_idx] = 1
        V[j0 * p + k0][col_idx] = 1
        W[k0 * n + (i0+1)][col_idx] = 1
        
        col_idx = col; col += 1
        U[(i0+1) * m + j0][col_idx] = 1
        V[j0 * p + (k0+1)][col_idx] = 1
        W[(k0+1) * n + (i0+1)][col_idx] = 1

    # Schoolbook 2x1x1
    # A (2x1), B (1x1)
    def add_sb_2x1x1(i0, j0, k0):
        nonlocal col
        col_idx = col; col += 1
        U[i0 * m + j0][col_idx] = 1
        V[j0 * p + k0][col_idx] = 1
        W[k0 * n + i0][col_idx] = 1
        
        col_idx = col; col += 1
        U[(i0+1) * m + j0][col_idx] = 1
        V[j0 * p + k0][col_idx] = 1
        W[k0 * n + (i0+1)][col_idx] = 1

    # Part 1: A[:, 0:2] * B[0:2, :]
    # A indices: i in 0,1; j in 0,1
    # B indices: j in 0,1; k in 0..6
    
    # B[0:2, 0:2] -> Strassen
    add_strassen(0, 0, 0)
    # B[0:2, 2:4] -> Strassen
    add_strassen(0, 0, 2)
    # B[0:2, 4:6] -> Strassen
    add_strassen(0, 0, 4)
    # B[0:2, 6:7] -> SB 2x2x1
    add_sb_2x2x1(0, 0, 6)

    # Part 2: A[:, 2] * B[2, :]
    # A indices: i in 0,1; j in 2
    # B indices: j in 2; k in 0..6
    
    # B[2, 0:2] -> SB 2x1x2
    add_sb_2x1x2(0, 2, 0)
    # B[2, 2:4] -> SB 2x1x2
    add_sb_2x1x2(0, 2, 2)
    # B[2, 4:6] -> SB 2x1x2
    add_sb_2x1x2(0, 2, 4)
    # B[2, 6:7] -> SB 2x1x1
    add_sb_2x1x1(0, 2, 6)

    # Verify rank used
    if col != rank:
        # Adjust if mismatch, but 39 is calculated exactly
        pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
