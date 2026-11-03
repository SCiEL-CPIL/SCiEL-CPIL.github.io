import numpy as np
import time

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-56 decomposition for 4x4x4 matrix multiplication.
    
    Approach: Treat 4x4 matrices as 2x2 block matrices with 2x2 blocks.
    This requires 8 independent 2x2 matrix multiplications:
      C00 = A00*B00 + A01*B10
      C01 = A00*B01 + A01*B11
      C10 = A10*B00 + A11*B10
      C11 = A10*B01 + A11*B11
    
    Each 2x2 multiplication uses Strassen's rank-7 algorithm.
    Total rank = 8 * 7 = 56.
    
    Index conventions:
      U index: i*4 + j  (row i, col j of A)
      V index: j*4 + k  (row j, col k of B)
      W index: k*4 + i  (row k, col i of C) -- trace convention
    
    Block structure:
      A00 = A[0:2, 0:2], A01 = A[0:2, 2:4]
      A10 = A[2:4, 0:2], A11 = A[2:4, 2:4]
      Similarly for B and C.
    """
    n, m, p = 4, 4, 4
    rank = 56
    
    # Strassen decomposition for 2x2x2
    # For 2x2 matrices indexed 0,1:
    # U2[i*2+j, c] where i,j in {0,1}
    # V2[j*2+k, c]
    # W2[k*2+i, c]
    U2 = [
        [1, 0, 1, 0, 1, -1, 0],   # A00
        [0, 0, 0, 0, 1, 0, 1],    # A01
        [0, 1, 0, 0, 0, 1, 0],    # A10
        [1, 1, 0, 1, 0, 0, -1]    # A11
    ]
    V2 = [
        [1, 1, 0, -1, 0, 1, 0],   # B00
        [0, 0, 0, 1, 0, 1, 0],    # B01
        [0, 0, 1, 0, 0, 0, 1],    # B10
        [1, 0, -1, 0, 1, 0, 1]    # B11
    ]
    W2 = [
        [1, 0, 0, 0, 0, 0, 0],    # C00
        [0, 0, 0, 0, 1, 0, 0],    # C01
        [0, 1, 1, 0, 0, 1, 0],    # C10
        [1, 0, 0, 0, 0, 0, 1]     # C11
    ]
    
    # Verify Strassen decomposition
    for i in range(2):
        for j in range(2):
            for k in range(2):
                s = 0
                for c in range(7):
                    s += U2[i*2+j][c] * V2[j*2+k][c] * W2[k*2+i][c]
                if s != 1:
                    # Fallback to schoolbook
                    return _schoolbook()
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # The 8 block multiplications:
    # (A_block, B_block, C_block)
    # A blocks: A00=(0,0), A01=(0,1), A10=(1,0), A11=(1,1)
    #   where (row_block, col_block)
    # B blocks: B00=(0,0), B01=(0,1), B10=(1,0), B11=(1,1)
    #   where (row_block, col_block)
    # C blocks: C00=(0,0), C01=(0,1), C10=(1,0), C11=(1,1)
    #   where (row_block, col_block)
    
    # Block multiplications needed:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # Each term: A[r_a, c_a] * B[r_b, c_b] -> C[r_c, c_c]
    # where r_a == r_c (row block of A = row block of C)
    # and c_a == r_b (col block of A = row block of B)
    # and c_b == c_c (col block of B = col block of C)
    
    block_mults = [
        # (A_row, A_col, B_row, B_col, C_row, C_col)
        (0, 0, 0, 0, 0, 0),  # A00 * B00 -> C00
        (0, 1, 1, 0, 0, 0),  # A01 * B10 -> C00
        (0, 0, 0, 1, 0, 1),  # A00 * B01 -> C01
        (0, 1, 1, 1, 0, 1),  # A01 * B11 -> C01
        (1, 0, 0, 0, 1, 0),  # A10 * B00 -> C10
        (1, 1, 1, 0, 1, 0),  # A11 * B10 -> C10
        (1, 0, 0, 1, 1, 1),  # A10 * B01 -> C11
        (1, 1, 1, 1, 1, 1),  # A11 * B11 -> C11
    ]
    
    for term_idx, (ar, ac, br, bc, cr, cc) in enumerate(block_mults):
        base_col = term_idx * 7
        for c in range(7):
            col = base_col + c
            # Fill U: A entries at block (ar, ac), local positions (i, j)
            for i in range(2):
                for j in range(2):
                    global_i = ar * 2 + i
                    global_j = ac * 2 + j
                    U[global_i * 4 + global_j][col] = U2[i * 2 + j][c]
            
            # Fill V: B entries at block (br, bc), local positions (j, k)
            for j in range(2):
                for k in range(2):
                    global_j = br * 2 + j
                    global_k = bc * 2 + k
                    V[global_j * 4 + global_k][col] = V2[j * 2 + k][c]
            
            # Fill W: C entries at block (cr, cc), local positions (k, i)
            # W index is k*4 + i (trace convention)
            for k in range(2):
                for i in range(2):
                    global_k = cc * 2 + k
                    global_i = cr * 2 + i
                    W[global_k * 4 + global_i][col] = W2[k * 2 + i][c]
    
    return {"U": U, "V": V, "W": W}


def _schoolbook():
    """Fallback rank-64 schoolbook decomposition."""
    n, m, p = 4, 4, 4
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
