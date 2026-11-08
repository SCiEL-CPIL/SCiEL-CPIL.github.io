import numpy as np
import time

DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-42 tensor decomposition for 2x4 * 4x6 matrix multiplication.
    
    Strategy: Split the 2x4 and 4x6 matrices into 2x2 blocks and apply Strassen's
    algorithm to each pair of blocks. This gives 6 blocks * 7 Strassen multiplications = 42.
    
    The output factor W follows the trace convention W[k*n+i, r] where:
    - k is the column index in C (0..p-1)
    - i is the row index in C (0..n-1)
    - r is the rank index (0..rank-1)
    """
    n, m, p = 2, 4, 6  # A is n x m, B is m x p, C is n x p
    rank = 42
    
    # Initialize factor matrices
    U = np.zeros((n * m, rank), dtype=float)  # 8 x 42
    V = np.zeros((m * p, rank), dtype=float)  # 24 x 42
    W = np.zeros((p * n, rank), dtype=float)  # 12 x 42
    
    # For each pair of 2x2 blocks, apply Strassen's algorithm
    # A has 2 rows, B has 4 rows (2 groups of 2), C has 6 cols (3 groups of 2)
    
    for group in range(2):  # 2 groups of 2 rows in B
        for block_col in range(3):  # 3 groups of 2 cols in B/C
            
            col_offset = group * 3 + block_col  # 0..5
            base_col = col_offset * 7  # 0, 7, 14, 21, 28, 35
            
            # Define index functions for A, B, W
            def a_idx(i, j):
                """Index in A (2x4): row i (0-1), col j (0-3)"""
                return i * m + j
            
            def b_idx(i, j):
                """Index in B (4x6): row i (0-3), col j (0-5)"""
                return i * p + j
            
            def w_idx(row, col):
                """Index in W following trace convention W[k*n+i, r]"""
                return col * n + row
            
            # Local 2x2 block indices
            # A block: rows 0-1, cols 2*group to 2*group+1
            # B block: rows 2*group to 2*group+1, cols 2*block_col to 2*block_col+1
            # C block: rows 0-1, cols 2*block_col to 2*block_col+1
            
            a00 = a_idx(0, 2 * group)
            a01 = a_idx(0, 2 * group + 1)
            a10 = a_idx(1, 2 * group)
            a11 = a_idx(1, 2 * group + 1)
            
            b00 = b_idx(2 * group, 2 * block_col)
            b01 = b_idx(2 * group, 2 * block_col + 1)
            b10 = b_idx(2 * group + 1, 2 * block_col)
            b11 = b_idx(2 * group + 1, 2 * block_col + 1)
            
            c00 = w_idx(0, 2 * block_col)
            c01 = w_idx(0, 2 * block_col + 1)
            c10 = w_idx(1, 2 * block_col)
            c11 = w_idx(1, 2 * block_col + 1)
            
            # Strassen's 7 multiplications
            # M1 = (A00 + A11) * (B00 + B11)
            U[a00, base_col] += 1
            U[a11, base_col] += 1
            V[b00, base_col] += 1
            V[b11, base_col] += 1
            W[c00, base_col] += 1
            W[c11, base_col] += 1
            
            # M2 = (A10 + A11) * B00
            U[a10, base_col + 1] += 1
            U[a11, base_col + 1] += 1
            V[b00, base_col + 1] += 1
            W[c10, base_col + 1] += 1
            W[c11, base_col + 1] -= 1
            
            # M3 = A00 * (B01 - B11)
            U[a00, base_col + 2] += 1
            V[b01, base_col + 2] += 1
            V[b11, base_col + 2] -= 1
            W[c01, base_col + 2] += 1
            W[c11, base_col + 2] += 1
            
            # M4 = A11 * (B10 - B00)
            U[a11, base_col + 3] += 1
            V[b10, base_col + 3] += 1
            V[b00, base_col + 3] -= 1
            W[c00, base_col + 3] += 1
            W[c10, base_col + 3] += 1
            
            # M5 = (A00 + A01) * B11
            U[a00, base_col + 4] += 1
            U[a01, base_col + 4] += 1
            V[b11, base_col + 4] += 1
            W[c00, base_col + 4] -= 1
            W[c01, base_col + 4] += 1
            
            # M6 = (A10 - A00) * (B00 + B01)
            U[a10, base_col + 5] += 1
            U[a00, base_col + 5] -= 1
            V[b00, base_col + 5] += 1
            V[b01, base_col + 5] += 1
            W[c11, base_col + 5] += 1
            
            # M7 = (A01 - A11) * (B10 + B11)
            U[a01, base_col + 6] += 1
            U[a11, base_col + 6] -= 1
            V[b10, base_col + 6] += 1
            V[b11, base_col + 6] += 1
            W[c00, base_col + 6] += 1
    
    # Convert to lists as required by the interface
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
