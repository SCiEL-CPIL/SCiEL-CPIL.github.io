import time
import random
from fractions import Fraction
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # n=2, m=4, p=4

    # Strassen decomposition for 2x2 * 2x2
    # M1 = (A11 + A22) * (B11 + B22)
    # M2 = (A21 + A22) * B11
    # M3 = A11 * (B12 - B22)
    # M4 = A22 * (B21 - B11)
    # M5 = (A11 + A12) * B22
    # M6 = (A21 - A11) * (B11 + B12)
    # M7 = (A12 - A22) * (B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # We need to map this to the 2x4 * 4x4 case.
    # A is 2x4, split into A1 (2x2) and A2 (2x2).
    # B is 4x4, split into B11, B12, B21, B22 (all 2x2).
    # C is 2x4, split into C1 (2x2) and C2 (2x2).
    # C1 = A1*B11 + A2*B21
    # C2 = A1*B12 + A2*B22

    # We can use 7 mults for A1*B11, 7 for A2*B21, 7 for A1*B12, 7 for A2*B22.
    # Total 28.
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    def emit_strassen(a_offset, b_row, b_col, c_col, start_rank):
        # a_offset: 0 or 2 (which 2x2 block of A)
        # b_row: 0 or 2 (which row of 2x2 blocks in B)
        # b_col: 0 or 2 (which col of 2x2 blocks in B)
        # c_col: 0 or 2 (which 2x2 block of C)
        
        # A block indices
        # A is 2x4. Rows 0-1. Cols 0-3.
        # A1 is cols 0-1, A2 is cols 2-3.
        # a_offset 0 -> cols 0,1. a_offset 2 -> cols 2,3.
        
        # B block indices
        # B is 4x4. Rows 0-3, Cols 0-3.
        # B11 is rows 0-1, cols 0-1.
        # B12 is rows 0-1, cols 2-3.
        # B21 is rows 2-3, cols 0-1.
        # B22 is rows 2-3, cols 2-3.
        # b_row 0 -> rows 0,1. b_row 2 -> rows 2,3.
        # b_col 0 -> cols 0,1. b_col 2 -> cols 2,3.
        
        # C block indices
        # C is 2x4. Rows 0-1. Cols 0-3.
        # C1 is cols 0-1, C2 is cols 2-3.
        # c_col 0 -> cols 0,1. c_col 2 -> cols 2,3.

        # Strassen variables
        # M1 = (A11 + A22) * (B11 + B22)
        # A11 is A[a_offset, a_offset], A22 is A[a_offset+1, a_offset+1]
        # B11 is B[b_row, b_col], B22 is B[b_row+1, b_col+1]
        
        # Indices in U, V, W
        # U[i][r] where i = row * m + col. m=4.
        # V[j][r] where j = row * p + col. p=4.
        # W[k][r] where k = row * n + col. n=2.
        
        # Let's define the 7 terms for this block product.
        # Term 1: (A11 + A22) * (B11 + B22)
        # U: A11 (1), A22 (1)
        # V: B11 (1), B22 (1)
        # W: C11 (1), C12 (1), C21 (1), C22 (1) -- wait, C is 2x2.
        # C11 is C[c_col, c_col]
        # C12 is C[c_col, c_col+1]
        # C21 is C[c_col+1, c_col]
        # C22 is C[c_col+1, c_col+1]
        
        # Let's just use the standard Strassen mapping for C = A*B.
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # M1 = (A11 + A22) * (B11 + B22)
        r1 = start_rank
        U[a_offset * 4 + a_offset][r1] += 1
        U[a_offset * 4 + a_offset + 1][r1] += 1
        V[b_row * 4 + b_col][r1] += 1
        V[b_row * 4 + b_col + 1][r1] += 1
        W[c_col * 2 + c_col][r1] += 1
        W[c_col * 2 + c_col + 1][r1] += 1
        W[c_col * 2 + c_col + 1][r1] += 1
        W[c_col * 2 + c_col + 2][r1] += 1 # Wait, W index is k*n+i. k is col of C? No.
        
        # Let's re-verify W index.
        # W[k*n + i] where k is col of C, i is row of C.
        # C is 2x4. Rows 0-1. Cols 0-3.
        # C1 is cols 0-1. C2 is cols 2-3.
        # C11 is C[0,0], C12 is C[0,1], C21 is C[1,0], C22 is C[1,1].
        # W index for C11: k=0, i=0 -> 0*2+0=0.
        # W index for C12: k=1, i=0 -> 1*2+0=2.
        # W index for C21: k=0, i=1 -> 0*2+1=1.
        # W index for C22: k=1, i=1 -> 1*2+1=3.
        
        # So for block at c_col:
        # C11 is at c_col, C12 at c_col+1, C21 at c_col, C22 at c_col+1.
        # W index for C11: (c_col)*2 + 0
        # W index for C12: (c_col+1)*2 + 0
        # W index for C21: (c_col)*2 + 1
        # W index for C22: (c_col+1)*2 + 1
        
        # Let's restart the W mapping.
        # M1 contributes to C11, C12, C21, C22 with +1.
        W[c_col * 2 + 0][r1] += 1
        W[(c_col + 1) * 2 + 0][r1] += 1
        W[c_col * 2 + 1][r1] += 1
        W[(c_col + 1) * 2 + 1][r1] += 1
        
        # M2 = (A21 + A22) * B11
        r2 = start_rank + 1
        U[a_offset * 4 + a_offset + 1][r2] += 1
        U[a_offset * 4 + a_offset + 2][r2] += 1 # A21 is row 1, col 0 of block.
        # A block is 2x2.
        # A11 is (0,0), A12 is (0,1), A21 is (1,0), A22 is (1,1).
        # In global A, row is 0 or 1. Col is a_offset + 0 or a_offset + 1.
        # A11: row 0, col a_offset. Index 0*4 + a_offset.
        # A12: row 0, col a_offset+1. Index 0*4 + a_offset+1.
        # A21: row 1, col a_offset. Index 1*4 + a_offset.
        # A22: row 1, col a_offset+1. Index 1*4 + a_offset+1.
        
        # Correct U indices:
        # A11: 0*4 + a_offset
        # A12: 0*4 + a_offset + 1
        # A21: 1*4 + a_offset
        # A22: 1*4 + a_offset + 1
        
        # Let's redo U for M1.
        # U[a_offset * 4 + a_offset] -> A11. Correct.
        # U[a_offset * 4 + a_offset + 1] -> A12. Incorrect. Should be A22.
        # A22 is 1*4 + a_offset + 1.
        
        # Let's write a helper to get U index.
        def u_idx(r, c): return r * 4 + c
        def v_idx(r, c): return r * 4 + c
        def w_idx(r, c): return c * 2 + r # k*n+i. k is col, i is row.
        
        # M1 = (A11 + A22) * (B11 + B22)
        r1 = start_rank
        U[u_idx(0, a_offset)][r1] += 1
        U[u_idx(1, a_offset + 1)][r1] += 1
        V[v_idx(b_row, b_col)][r1] += 1
        V[v_idx(b_row + 1, b_col + 1)][r1] += 1
        W[w_idx(0, c_col)][r1] += 1
        W[w_idx(0, c_col + 1)][r1] += 1
        W[w_idx(1, c_col)][r1] += 1
        W[w_idx(1, c_col + 1)][r1] += 1
        
        # M2 = (A21 + A22) * B11
        r2 = start_rank + 1
        U[u_idx(1, a_offset)][r2] += 1
        U[u_idx(1, a_offset + 1)][r2] += 1
        V[v_idx(b_row, b_col)][r2] += 1
        W[w_idx(1, c_col)][r2] += 1
        W[w_idx(1, c_col + 1)][r2] += 1
        
        # M3 = A11 * (B12 - B22)
        r3 = start_rank + 2
        U[u_idx(0, a_offset)][r3] += 1
        V[v_idx(b_row, b_col + 1)][r3] += 1
        V[v_idx(b_row + 1, b_col + 1)][r3] -= 1
        W[w_idx(0, c_col + 1)][r3] += 1
        W[w_idx(1, c_col + 1)][r3] += 1
        
        # M4 = A22 * (B21 - B11)
        r4 = start_rank + 3
        U[u_idx(1, a_offset + 1)][r4] += 1
        V[v_idx(b_row + 1, b_col)][r4] += 1
        V[v_idx(b_row, b_col)][r4] -= 1
        W[w_idx(0, c_col)][r4] += 1
        W[w_idx(1, c_col)][r4] += 1
        
        # M5 = (A11 + A12) * B22
        r5 = start_rank + 4
        U[u_idx(0, a_offset)][r5] += 1
        U[u_idx(0, a_offset + 1)][r5] += 1
        V[v_idx(b_row + 1, b_col + 1)][r5] += 1
        W[w_idx(0, c_col)][r5] -= 1
        W[w_idx(0, c_col + 1)][r5] += 1
        
        # M6 = (A21 - A11) * (B11 + B12)
        r6 = start_rank + 5
        U[u_idx(1, a_offset)][r6] += 1
        U[u_idx(0, a_offset)][r6] -= 1
        V[v_idx(b_row, b_col)][r6] += 1
        V[v_idx(b_row, b_col + 1)][r6] += 1
        W[w_idx(1, c_col + 1)][r6] += 1
        
        # M7 = (A12 - A22) * (B21 + B22)
        r7 = start_rank + 6
        U[u_idx(0, a_offset + 1)][r7] += 1
        U[u_idx(1, a_offset + 1)][r7] -= 1
        V[v_idx(b_row + 1, b_col)][r7] += 1
        V[v_idx(b_row + 1, b_col + 1)][r7] += 1
        W[w_idx(0, c_col)][r7] += 1
        W[w_idx(1, c_col + 1)][r7] += 1

    # C1 = A1*B11 + A2*B21
    # A1 is a_offset=0. B11 is b_row=0, b_col=0. C1 is c_col=0.
    # A2 is a_offset=2. B21 is b_row=2, b_col=0. C1 is c_col=0.
    
    # We can just call emit_strassen 4 times.
    emit_strassen(0, 0, 0, 0, 0)   # A1 * B11
    emit_strassen(2, 2, 0, 0, 7)   # A2 * B21
    emit_strassen(0, 0, 2, 2, 14)  # A1 * B12
    emit_strassen(2, 2, 2, 2, 21)  # A2 * B22

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
