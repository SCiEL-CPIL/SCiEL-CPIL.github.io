import time
import numpy as np

# EVOLVE-BLOCK-START

def _apply_strassen_22(A_base, B_base, C_base, k, n, p, m, U, V, W, col_offset):
    """
    Apply a standard Strassen decomposition for a 2x2 * 2x2 block.
    Returns the new column index.
    A_base: start index for A elements (always 0)
    B_base: start index for B elements (col offset)
    C_base: start index for C elements (col offset)
    """
    col = col_offset
    
    # Indices for A: 0,1 (row 0), 2,3 (row 1)
    a_idx = 0
    b_idx = 1
    c_idx = 2
    d_idx = 3
    
    # Indices for B: relative to B_base
    # B is 2 x 2 block.
    # In V, index is i*p + j.
    # The block covers columns C_base and C_base+1? No, B_base is column index.
    # B_col0 = B_base, B_col1 = B_base + 1
    e_idx = B_base       # B[0, col0] -> index 0*p + col0
    f_idx = B_base + 1   # B[0, col1] -> index 0*p + col1
    g_idx = p + B_base   # B[1, col0] -> index 1*p + col0
    h_idx = p + B_base + 1 # B[1, col1] -> index 1*p + col1
    
    # Indices for C: relative to C_base
    # C is 2 x 2 block.
    # W index convention: k*n + i where C is p x n (13 x 2).
    # C[row, col] -> index col*n + row.
    # Col0: C_base, Col1: C_base + 1
    c00_idx = C_base * n + 0       # C[0, col0]
    c10_idx = C_base * n + 1       # C[1, col0]
    c01_idx = (C_base + 1) * n + 0 # C[0, col1]
    c11_idx = (C_base + 1) * n + 1 # C[1, col1]

    # Strassen Multiplications:
    # M1 = (a+d)(e+h)
    U[a_idx, col] += 1.0; U[d_idx, col] += 1.0
    V[e_idx, col] += 1.0; V[h_idx, col] += 1.0
    W[c00_idx, col] += 1.0; W[c11_idx, col] += 1.0
    col += 1

    # M2 = (c+d)e
    U[c_idx, col] += 1.0; U[d_idx, col] += 1.0
    V[e_idx, col] += 1.0
    W[c10_idx, col] += 1.0; W[c11_idx, col] -= 1.0
    col += 1

    # M3 = a(f-h)
    U[a_idx, col] += 1.0
    V[f_idx, col] += 1.0; V[h_idx, col] -= 1.0
    W[c01_idx, col] += 1.0; W[c11_idx, col] += 1.0
    col += 1

    # M4 = d(g-e)
    U[d_idx, col] += 1.0
    V[g_idx, col] += 1.0; V[e_idx, col] -= 1.0
    W[c00_idx, col] += 1.0; W[c10_idx, col] += 1.0
    col += 1

    # M5 = (a+b)h
    U[a_idx, col] += 1.0; U[b_idx, col] += 1.0
    V[h_idx, col] += 1.0
    W[c00_idx, col] -= 1.0; W[c01_idx, col] += 1.0
    col += 1

    # M6 = (c-a)(e+f)
    U[c_idx, col] += 1.0; U[a_idx, col] -= 1.0
    V[e_idx, col] += 1.0; V[f_idx, col] += 1.0
    W[c11_idx, col] += 1.0
    col += 1

    # M7 = (b-d)(g+h)
    U[b_idx, col] += 1.0; U[d_idx, col] -= 1.0
    V[g_idx, col] += 1.0; V[h_idx, col] += 1.0
    W[c00_idx, col] += 1.0
    col += 1
    
    return col

def _apply_schoolbook_21(A_base, B_col, C_col, n, p, U, V, W, col_offset):
    """
    Apply schoolbook multiplication for 2x2 * 2x1 (single column).
    """
    col = col_offset
    a_idx = 0
    b_idx = 1
    c_idx = 2
    d_idx = 3
    
    b0_idx = 0 * p + B_col # B[0, col]
    b1_idx = 1 * p + B_col # B[1, col]
    
    c0_idx = C_col * n + 0 # C[0, col]
    c1_idx = C_col * n + 1 # C[1, col]

    # a * B[0,col]
    U[a_idx, col] += 1.0
    V[b0_idx, col] += 1.0
    W[c0_idx, col] += 1.0
    col += 1
    
    # b * B[1,col]
    U[b_idx, col] += 1.0
    V[b1_idx, col] += 1.0
    W[c0_idx, col] += 1.0
    col += 1
    
    # c * B[0,col]
    U[c_idx, col] += 1.0
    V[b0_idx, col] += 1.0
    W[c1_idx, col] += 1.0
    col += 1
    
    # d * B[1,col]
    U[d_idx, col] += 1.0
    V[b1_idx, col] += 1.0
    W[c1_idx, col] += 1.0
    col += 1
    
    return col

def construct():
    start = time.time()
    n, m, p = 2, 2, 13
    rank = 46
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    current_col = 0
    
    # Process 6 pairs of columns using Strassen
    # Columns: (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for start_col in range(0, 12, 2):
        current_col = _apply_strassen_22(0, start_col, start_col, 0, n, p, m, U, V, W, current_col)
        
        # Simple timeout guard inside loop (unlikely to trigger for small ops)
        if time.time() - start > 100:
            # If we ran out of time, we must return what we have, but ranks must match.
            # Since this is deterministic and fast, we assume we won't hit this.
            break

    # Process the last column (12) using Schoolbook
    if current_col < rank:
        current_col = _apply_schoolbook_21(0, 12, 12, n, p, U, V, W, current_col)

    # Verify we filled the expected rank
    if current_col != rank:
        # Fill remaining with zeros (should not happen if logic is correct)
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
