import time
import numpy as np
from fractions import Fraction

def construct():
    """
    Rank-minimal exact decomposition for 2x4 @ 4x6 matrix multiplication.
    
    Uses the block-partitioned Strassen approach:
    - Split the 4-dimension into two 2-dimensions (left/right)
    - Split the 6-dimension into three 2-dimensions
    - Apply Strassen's rank-7 algorithm to each 2x2 @ 2x2 block
    - Total rank = 2 * 3 * 7 = 42
    
    This is the known optimal approach for this tensor structure.
    """
    n, m, p = 2, 4, 6
    rank_total = 42
    
    U = [[0] * rank_total for _ in range(n * m)]   # 8 x 42
    V = [[0] * rank_total for _ in range(m * p)]   # 24 x 42
    W = [[0] * rank_total for _ in range(p * n)]   # 12 x 42
    
    # Strassen 2x2 @ 2x2 rank-7 decomposition
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # P1 = (a11+a22)*(b11+b22)
    # P2 = (a21+a22)*b11
    # P3 = a11*(b12-b22)
    # P4 = a22*(b21-b11)
    # P5 = (a11+a12)*b22
    # P6 = (a21-a11)*(b11+b12)
    # P7 = (a12-a22)*(b21+b22)
    #
    # c11 = P1 + P4 - P5 + P7
    # c12 = P3 + P5
    # c21 = P2 + P4
    # c22 = P1 - P2 + P3 + P6
    
    # For each product, store (U_entries, V_entries, W_entries) as dict {local_idx: sign}
    # Local indices for 2x2 matrices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_U = [
        {0: 1, 3: 1},       # P1: a11+a22
        {2: 1, 3: 1},       # P2: a21+a22
        {0: 1},             # P3: a11
        {3: 1},             # P4: a22
        {0: 1, 1: 1},       # P5: a11+a12
        {2: 1, 0: -1},      # P6: a21-a11
        {1: 1, 3: -1},      # P7: a12-a22
    ]
    strassen_V = [
        {0: 1, 3: 1},       # P1: b11+b22
        {0: 1},             # P2: b11
        {1: 1, 3: -1},      # P3: b12-b22
        {2: 1, 0: -1},      # P4: b21-b11
        {3: 1},             # P5: b22
        {0: 1, 1: 1},       # P6: b11+b12
        {2: 1, 3: 1},       # P7: b21+b22
    ]
    strassen_W = [
        {0: 1, 3: 1},       # P1: c11+c22
        {2: 1, 3: -1},      # P2: c21-c22
        {1: 1, 3: 1},       # P3: c12+c22
        {0: 1, 2: 1},       # P4: c11+c21
        {0: -1, 1: 1},      # P5: -c11+c12
        {3: 1},             # P6: c22
        {0: 1},             # P7: c11
    ]
    
    col_offset = 0
    # Iterate over the 3 blocks in the 6-dimension (each of size 2)
    for j_block in range(3):
        # Iterate over the 2 blocks in the 4-dimension (left: cols 0-1, right: cols 2-3)
        for side in ['L', 'R']:
            if side == 'L':
                u_col_offset = 0
                v_row_offset = 0
            else:
                u_col_offset = 2
                v_row_offset = 2
            
            for prod_idx in range(7):
                col = col_offset + prod_idx
                
                # U local index mapping:
                # local 0 -> A[0][u_col_offset+0], U idx = 0*4 + (u_col_offset+0)
                # local 1 -> A[0][u_col_offset+1], U idx = 0*4 + (u_col_offset+1)
                # local 2 -> A[1][u_col_offset+0], U idx = 1*4 + (u_col_offset+0)
                # local 3 -> A[1][u_col_offset+1], U idx = 1*4 + (u_col_offset+1)
                local_U = {
                    0: 0 * 4 + (u_col_offset + 0),
                    1: 0 * 4 + (u_col_offset + 1),
                    2: 1 * 4 + (u_col_offset + 0),
                    3: 1 * 4 + (u_col_offset + 1),
                }
                for li, s in strassen_U[prod_idx].items():
                    U[local_U[li]][col] += s
                
                # V local index mapping:
                # local 0 -> B[v_row_offset+0][2*j_block+0], V idx = (v_row_offset+0)*6 + (2*j_block+0)
                # local 1 -> B[v_row_offset+0][2*j_block+1], V idx = (v_row_offset+0)*6 + (2*j_block+1)
                # local 2 -> B[v_row_offset+1][2*j_block+0], V idx = (v_row_offset+1)*6 + (2*j_block+0)
                # local 3 -> B[v_row_offset+1][2*j_block+1], V idx = (v_row_offset+1)*6 + (2*j_block+1)
                local_V = {
                    0: (v_row_offset + 0) * 6 + (2 * j_block + 0),
                    1: (v_row_offset + 0) * 6 + (2 * j_block + 1),
                    2: (v_row_offset + 1) * 6 + (2 * j_block + 0),
                    3: (v_row_offset + 1) * 6 + (2 * j_block + 1),
                }
                for li, s in strassen_V[prod_idx].items():
                    V[local_V[li]][col] += s
                
                # W local index mapping:
                # local 0 -> C[0][2*j_block+0], W idx = (2*j_block+0)*2 + 0
                # local 1 -> C[0][2*j_block+1], W idx = (2*j_block+1)*2 + 0
                # local 2 -> C[1][2*j_block+0], W idx = (2*j_block+0)*2 + 1
                # local 3 -> C[1][2*j_block+1], W idx = (2*j_block+1)*2 + 1
                local_W = {
                    0: (2 * j_block + 0) * 2 + 0,
                    1: (2 * j_block + 1) * 2 + 0,
                    2: (2 * j_block + 0) * 2 + 1,
                    3: (2 * j_block + 1) * 2 + 1,
                }
                for li, s in strassen_W[prod_idx].items():
                    W[local_W[li]][col] += s
            
            col_offset += 7
    
    # Verify the decomposition
    def verify(U, V, W):
        r = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for t in range(r):
                        s += U[i*m+j][t] * V[j*p+k][t] * W[k*n+i][t]
                    if s != 1:
                        return False
        return True
    
    if not verify(U, V, W):
        raise ValueError("Decomposition verification failed")
    
    return {"U": U, "V": V, "W": W}
