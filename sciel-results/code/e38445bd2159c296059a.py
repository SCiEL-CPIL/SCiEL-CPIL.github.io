"""Matrix multiplication 2x2x14: rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    """Return factor matrices for 2x2 by 2x14 matmul using Strassen on 2x2 blocks.
    
    Convention: U has shape (n*m, rank) = (4, rank)
               V has shape (m*p, rank) = (28, rank)
               W has shape (p*n, rank) = (28, rank) with index W[k*n+i, r]
    
    Uses Strassen's algorithm on each 2x2 block of the 2x14 matrix B.
    14 columns -> 7 blocks of 2 columns each.
    Each 2x2 * 2x2 multiplication uses 7 Strassen terms.
    Total rank = 7 blocks * 7 Strassen terms = 49.
    
    Strassen formulas for C = A * B where A, B are 2x2:
    M1 = (A00 + A11) * (B00 + B11)
    M2 = (A10 + A11) * B00
    M3 = A00 * (B10 - B11)
    M4 = A11 * (B01 - B00)
    M5 = (A00 + A01) * B11
    M6 = (A10 - A00) * (B00 + B01)
    M7 = (A01 - A11) * (B10 + B11)
    
    C00 = M1 + M4 - M5 + M7
    C01 = M3 + M5
    C10 = M2 + M4
    C11 = M1 - M2 + M3 + M6
    """
    n, m, p = DIMENSIONS  # 2, 2, 14
    
    num_blocks = p // 2  # 7 blocks of 2 columns
    rank = num_blocks * 7  # 49
    
    U = [[0] * rank for _ in range(n * m)]   # (4, 49)
    V = [[0] * rank for _ in range(m * p)]   # (28, 49)
    W = [[0] * rank for _ in range(p * n)]   # (28, 49)
    
    # Strassen A-forms (linear combinations of A elements)
    # A indexed as A[i*m+j] where i in {0,1}, j in {0,1}
    # A00 = A[0], A01 = A[1], A10 = A[2], A11 = A[3]
    # 
    # M1: (A00 + A11) -> U coeffs: [1, 0, 0, 1]
    # M2: (A10 + A11) -> U coeffs: [0, 0, 1, 1]
    # M3: A00 -> U coeffs: [1, 0, 0, 0]
    # M4: A11 -> U coeffs: [0, 0, 0, 1]
    # M5: (A00 + A01) -> U coeffs: [1, 1, 0, 0]
    # M6: (A10 - A00) -> U coeffs: [-1, 0, 1, 0]
    # M7: (A01 - A11) -> U coeffs: [0, 1, 0, -1]
    
    A_forms = [
        [1, 0, 0, 1],   # M1: A00 + A11
        [0, 0, 1, 1],   # M2: A10 + A11
        [1, 0, 0, 0],   # M3: A00
        [0, 0, 0, 1],   # M4: A11
        [1, 1, 0, 0],   # M5: A00 + A01
        [-1, 0, 1, 0],  # M6: A10 - A00
        [0, 1, 0, -1],  # M7: A01 - A11
    ]
    
    # For each block b (columns 2b, 2b+1) and each Strassen term s:
    # B is 2x2 for this block: B[j][2b+c] where j in {0,1}, c in {0,1}
    # B00 = B[0,2b], B01 = B[0,2b+1], B10 = B[1,2b], B11 = B[1,2b+1]
    # B index in V: j*p + k = j*14 + k
    #
    # Strassen B-forms for each term:
    # M1: (B00 + B11) -> V coeffs: B00:+1, B11:+1
    # M2: B00 -> V coeffs: B00:+1
    # M3: (B10 - B11) -> V coeffs: B10:+1, B11:-1
    # M4: (B01 - B00) -> V coeffs: B01:+1, B00:-1
    # M5: B11 -> V coeffs: B11:+1
    # M6: (B00 + B01) -> V coeffs: B00:+1, B01:+1
    # M7: (B10 + B11) -> V coeffs: B10:+1, B11:+1
    
    # B_forms[s] = list of (col_offset_in_block, coeff)
    B_forms = [
        [(0, 1), (1, 1)],    # M1: B00 + B11  (col 0: B00, col 1: B11)
        [(0, 1)],            # M2: B00
        [(0, 1), (1, -1)],   # M3: B10 - B11  (col 0: B10, col 1: B11)
        [(0, -1), (1, 1)],   # M4: B01 - B00  (col 0: B00, col 1: B01)
        [(1, 1)],            # M5: B11
        [(0, 1), (1, 1)],    # M6: B00 + B01  (col 0: B00, col 1: B01)
        [(0, 1), (1, 1)],    # M7: B10 + B11  (col 0: B10, col 1: B11)
    ]
    
    # Wait, I need to be more careful. Let me re-derive.
    # For block b, B is 2x2:
    #   B[0][0] = B[0, 2b] -> V index 0*14 + 2b = 2b
    #   B[0][1] = B[0, 2b+1] -> V index 0*14 + 2b+1 = 2b+1
    #   B[1][0] = B[1, 2b] -> V index 1*14 + 2b = 14+2b
    #   B[1][1] = B[1, 2b+1] -> V index 1*14 + 2b+1 = 14+2b+1
    
    # Let me redo B_forms with explicit V indices:
    # B_forms[s] = list of (v_index_offset, coeff) where actual index = base + offset
    # base for block b: row 0 col 2b is at 2b, row 1 col 2b is at 14+2b
    
    # Actually let me just use absolute indices.
    
    # Output C for block b:
    #   C[0][2b]   -> W index 2b*2 + 0 = 4b
    #   C[0][2b+1] -> W index (2b+1)*2 + 0 = 4b+2
    #   C[1][2b]   -> W index 2b*2 + 1 = 4b+1
    #   C[1][2b+1] -> W index (2b+1)*2 + 1 = 4b+3
    
    # Strassen output combinations:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for each Strassen term s in block b, the W contributions are:
    # M1: C00:+1, C11:+1
    # M2: C10:+1, C11:-1
    # M3: C01:+1, C11:+1
    # M4: C00:+1, C10:+1
    # M5: C00:-1, C01:+1
    # M6: C11:+1
    # M7: C00:+1
    
    # W_forms[s] = list of (output_col_in_block, coeff)
    # output col 0 -> C00, col 1 -> C01, col 2 -> C10, col 3 -> C11
    # Wait, output is 2x2: C[0][0]=C00, C[0][1]=C01, C[1][0]=C10, C[1][1]=C11
    # W index for block b, output (i, c): (2b+c)*2 + i
    
    W_forms = [
        [(0, 1), (3, 1)],     # M1: C00:+1, C11:+1
        [(2, 1), (3, -1)],    # M2: C10:+1, C11:-1
        [(1, 1), (3, 1)],     # M3: C01:+1, C11:+1
        [(0, 1), (2, 1)],     # M4: C00:+1, C10:+1
        [(0, -1), (1, 1)],    # M5: C00:-1, C01:+1
        [(3, 1)],             # M6: C11:+1
        [(0, 1)],             # M7: C00:+1
    ]
    
    # B_forms[s] = list of (row, col_in_block, coeff)
    # V index = row * p + (2*b + col_in_block)
    B_forms = [
        [(0, 0, 1), (1, 1, 1)],    # M1: B00 + B11
        [(0, 0, 1)],               # M2: B00
        [(1, 0, 1), (1, 1, -1)],   # M3: B10 - B11
        [(0, 1, 1), (0, 0, -1)],   # M4: B01 - B00
        [(1, 1, 1)],               # M5: B11
        [(0, 0, 1), (0, 1, 1)],    # M6: B00 + B01
        [(1, 0, 1), (1, 1, 1)],    # M7: B10 + B11
    ]
    
    col = 0
    for b in range(num_blocks):
        for s in range(7):
            # U: A-form for Strassen term s
            for ai, coeff in enumerate(A_forms[s]):
                if coeff != 0:
                    U[ai][col] += coeff
            
            # V: B-form for Strassen term s in block b
            for (row, col_in_block, coeff) in B_forms[s]:
                v_idx = row * p + (2 * b + col_in_block)
                V[v_idx][col] += coeff
            
            # W: output combination for Strassen term s in block b
            for (out_col, coeff) in W_forms[s]:
                # out_col: 0->C00, 1->C01, 2->C10, 3->C11
                # C[i][c] where i = out_col // 2, c = out_col % 2
                out_i = out_col // 2
                out_c = out_col % 2
                w_idx = (2 * b + out_c) * n + out_i
                W[w_idx][col] += coeff
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
