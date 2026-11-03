import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition
    # Split inner dimension m=4 into two blocks of 2.
    # A = [A1 | A2] (2x2 each)
    # B = [B1; B2] (2x8 each)
    # C = A1 B1 + A2 B2
    # Each Ai Bi is a 2x2 * 2x8 multiplication.
    # 2x2 * 2x8 can be done with 28 multiplications (4 blocks of 2x2*2x2 Strassen).
    
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen factors for 2x2 matrix multiplication
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We have two such blocks: Block 1 (A1 B1) and Block 2 (A2 B2)
    # Each block has 4 sub-blocks of 2x2 * 2x2.
    # Total 8 sub-blocks. Each uses 7 multiplications. Total 56.
    
    col = 0
    for block in range(2): # A1B1 and A2B2
        for sub in range(4): # 4 columns of B (2x2 each)
            # Sub-block indices
            # A part: rows 0,1 of A_block (which are rows 0,1 of A if block 0, rows 2,3? No)
            # A is 2x4. A1 is cols 0,1. A2 is cols 2,3.
            # So A1 rows are 0,1. A2 rows are 0,1.
            # Wait, A is 2x4. A1 is A[:, 0:2], A2 is A[:, 2:4].
            # So for Block 1, A indices are 0,1 (rows) and 0,1 (cols of A1).
            # For Block 2, A indices are 0,1 (rows) and 2,3 (cols of A2).
            
            # B part: B1 is rows 0,1 of B. B2 is rows 2,3 of B.
            # B is 4x8. B1 is B[0:2, :], B2 is B[2:4, :].
            # Sub-block sub (0..3) corresponds to cols 2*sub, 2*sub+1 of B.
            
            # A indices for this sub-block:
            # Row 0 of A_block: A[0, block*2 + 0]
            # Row 1 of A_block: A[0, block*2 + 1]
            # Row 0 of A_block: A[1, block*2 + 0]
            # Row 1 of A_block: A[1, block*2 + 1]
            # In U, index is i*m + j.
            # i is 0 or 1. j is block*2 + 0 or 1.
            
            # B indices for this sub-block:
            # Row 0 of B_block: B[block*2 + 0, 2*sub]
            # Row 1 of B_block: B[block*2 + 1, 2*sub]
            # Row 0 of B_block: B[block*2 + 0, 2*sub+1]
            # Row 1 of B_block: B[block*2 + 1, 2*sub+1]
            # In V, index is j*p + k.
            # j is block*2 + 0 or 1. k is 2*sub or 2*sub+1.
            
            # C indices:
            # C is 2x8.
            # C[0, 2*sub] = ...
            # C[0, 2*sub+1] = ...
            # C[1, 2*sub] = ...
            # C[1, 2*sub+1] = ...
            # In W, index is k*n + i.
            # k is 2*sub or 2*sub+1. i is 0 or 1.
            
            a11_idx = 0 * m + (block * 2 + 0)
            a12_idx = 0 * m + (block * 2 + 1)
            a21_idx = 1 * m + (block * 2 + 0)
            a22_idx = 1 * m + (block * 2 + 1)
            
            b11_idx = (block * 2 + 0) * p + (2 * sub)
            b12_idx = (block * 2 + 0) * p + (2 * sub + 1)
            b21_idx = (block * 2 + 1) * p + (2 * sub)
            b22_idx = (block * 2 + 1) * p + (2 * sub + 1)
            
            c11_idx = (2 * sub) * n + 0
            c12_idx = (2 * sub + 1) * n + 0
            c21_idx = (2 * sub) * n + 1
            c22_idx = (2 * sub + 1) * n + 1
            
            # M1
            U[a11_idx][col] += 1
            U[a22_idx][col] += 1
            V[b11_idx][col] += 1
            V[b22_idx][col] += 1
            W[c11_idx][col] += 1
            W[c22_idx][col] += 1
            col += 1
            
            # M2
            U[a21_idx][col] += 1
            U[a22_idx][col] += 1
            V[b11_idx][col] += 1
            W[c21_idx][col] += 1
            W[c22_idx][col] -= 1
            col += 1
            
            # M3
            U[a11_idx][col] += 1
            V[b12_idx][col] += 1
            V[b22_idx][col] -= 1
            W[c12_idx][col] += 1
            W[c22_idx][col] += 1
            col += 1
            
            # M4
            U[a22_idx][col] += 1
            V[b21_idx][col] += 1
            V[b11_idx][col] -= 1
            W[c11_idx][col] += 1
            W[c21_idx][col] += 1
            col += 1
            
            # M5
            U[a11_idx][col] += 1
            U[a12_idx][col] += 1
            V[b22_idx][col] += 1
            W[c11_idx][col] -= 1
            W[c12_idx][col] += 1
            col += 1
            
            # M6
            U[a21_idx][col] += 1
            U[a11_idx][col] -= 1
            V[b11_idx][col] += 1
            V[b12_idx][col] += 1
            W[c22_idx][col] += 1
            col += 1
            
            # M7
            U[a12_idx][col] += 1
            U[a22_idx][col] -= 1
            V[b21_idx][col] += 1
            V[b22_idx][col] += 1
            W[c11_idx][col] += 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
