"""Matrix multiplication 6x6x6 with Strassen block decomposition."""
import time
import numpy as np

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-189 decomposition: partition 6x6 into 2x2 grid of 3x3 blocks,
    apply Strassen (7 products) at block level, schoolbook (27 terms) for each 3x3x3.
    Total rank = 7 * 27 = 189.
    
    Convention: W[k*n + i, r] where C[i,k] = sum_j A[i,j]*B[j,k] = sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]
    Actually the standard is: C[i,k] = sum_j A[i,j] B[j,k], and the tensor is
    T[i,j,k] = A[i,j]*B[j,k]*C[i,k]. The decomposition is T = sum_r U[i,j,r] V[j,k,r] W[k,i,r]
    with W indexed as W[k*n+i, r] meaning W index = k*n + i for the (i,k) entry of C.
    """
    n, m, p = DIMENSIONS  # all 6
    block = 3
    
    # Strassen for 2x2 block matrices:
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # strassen_A[si] = dict mapping (block_row, block_col) -> coeff for A side
    # strassen_B[si] = dict mapping (block_row, block_col) -> coeff for B side
    strassen_A = [
        {(0, 0): 1, (1, 1): 1},  # M1
        {(1, 0): 1, (1, 1): 1},  # M2
        {(0, 0): 1},             # M3
        {(1, 1): 1},             # M4
        {(0, 0): 1, (0, 1): 1},  # M5
        {(1, 0): 1, (0, 0): -1}, # M6
        {(0, 1): 1, (1, 1): -1}, # M7
    ]
    strassen_B = [
        {(0, 0): 1, (1, 1): 1},  # M1
        {(0, 0): 1},             # M2
        {(0, 1): 1, (1, 1): -1}, # M3
        {(1, 0): 1, (0, 0): -1}, # M4
        {(1, 1): 1},             # M5
        {(0, 0): 1, (0, 1): 1},  # M6
        {(1, 0): 1, (1, 1): 1},  # M7
    ]
    
    # C output: for each si, which C blocks get what coefficient
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # c_blocks[si] = list of (c_block_row, c_block_col, coeff)
    c_blocks = [
        [(0, 0, 1), (1, 1, 1)],       # M1 -> C00 (+1), C11 (+1)
        [(1, 0, 1), (1, 1, -1)],      # M2 -> C10 (+1), C11 (-1)
        [(0, 1, 1), (1, 1, 1)],       # M3 -> C01 (+1), C11 (+1)
        [(0, 0, 1), (1, 0, 1)],       # M4 -> C00 (+1), C10 (+1)
        [(0, 0, -1), (0, 1, 1)],      # M5 -> C00 (-1), C01 (+1)
        [(1, 1, 1)],                  # M6 -> C11 (+1)
        [(0, 0, 1)],                  # M7 -> C00 (+1)
    ]
    
    rank = 7 * 27  # 189
    U = [[0] * rank for _ in range(n * m)]  # (36, 189)
    V = [[0] * rank for _ in range(m * p)]  # (36, 189)
    W = [[0] * rank for _ in range(p * n)]  # (36, 189)
    
    for si in range(7):
        A_coeff = strassen_A[si]
        B_coeff = strassen_B[si]
        c_targets = c_blocks[si]
        
        for a in range(block):  # row within 3x3 block
            for b in range(block):  # col within A block / row within B block
                for c in range(block):  # col within B block / col within C block
                    col = si * 27 + a * 9 + b * 3 + c
                    
                    # U[row_U, col]: A[i, j] where i = br*3+a, j = bc*3+b
                    for (br, bc), coeff in A_coeff.items():
                        row_U = (br * block + a) * m + (bc * block + b)
                        U[row_U][col] += coeff
                    
                    # V[row_V, col]: B[j, k] where j = br*3+b, k = bc*3+c
                    for (br, bc), coeff in B_coeff.items():
                        row_V = (br * block + b) * p + (bc * block + c)
                        V[row_V][col] += coeff
                    
                    # W[row_W, col]: C[i, k] where i = cbr*3+a, k = cbc*3+c
                    # W index = k * n + i
                    for (cbr, cbc, coeff) in c_targets:
                        i_global = cbr * block + a
                        k_global = cbc * block + c
                        row_W = k_global * n + i_global
                        W[row_W][col] += coeff
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
