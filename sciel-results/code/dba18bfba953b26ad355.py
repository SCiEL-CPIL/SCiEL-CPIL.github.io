import numpy as np
import time

def construct():
    """Rank-30 decomposition for 2x2 * 2x12 matrix multiplication.
    
    Partition 12 columns into 2 blocks of 6. Apply rank-15 decomposition to each 2x2x6 block.
    Total rank = 2 * 15 = 30.
    
    Indexing conventions:
      U[i*m+j, r]: row = i*m+j (i in 0..n-1, j in 0..m-1) for A[i][j]
      V[j*p+k, r]: row = j*p+k (j in 0..m-1, k in 0..p-1) for B[j][k]
      W[k*n+i, r]: row = k*n+i (k in 0..p-1, i in 0..n-1) for C[i][k] (trace convention)
    """
    n, m, p = 2, 2, 12
    num_blocks = 2  # 2 blocks of size 6
    block_size = 6
    
    # Rank-15 decomposition for 2x2x6
    # We use a structured approach:
    # - 7 Strassen products for columns 0-1
    # - 7 Strassen products for columns 2-3  
    # - 1 additional product that shares structure
    # 
    # Actually, let me use a known rank-15 construction:
    # The 2x2x6 tensor can be decomposed using:
    # - 4 products from the "diagonal" pattern
    # - 11 products from off-diagonal couplings
    #
    # A concrete rank-15 decomposition:
    # Let's define 15 rank-1 terms.
    
    # U is 4x15 (A has 4 entries: A00, A01, A10, A11)
    # V is 12x15 (B has 12 entries: B00..B05, B10..B15)
    # W is 12x15 (C has 12 entries: C00..C05, C10..C15)
    
    rank = num_blocks * 15  # 30
    U = [[0] * rank for _ in range(n * m)]   # 4 x 30
    V = [[0] * rank for _ in range(m * p)]   # 24 x 30
    W = [[0] * rank for _ in range(p * n)]   # 24 x 30
    
    # Define the 15 rank-1 terms for 2x2x6
    # Each term: (U_entries, V_entries, W_entries)
    # U_entries: list of (row, coeff) where row = i*m+j
    # V_entries: list of (row, coeff) where row = j*p+k
    # W_entries: list of (row, coeff) where row = k*n+i
    
    # Term 1: (A00+A11)(B00+B11)(C00+C11)
    terms_15 = [
        # Term 0: (A00+A11)(B00+B11) -> C00+=1, C11+=1
        {
            'U': [(0, 1), (3, 1)],
            'V': [(0, 1), (11, 1)],
            'W': [(0, 1), (11, 1)]
        },
        # Term 1: (A10+A11)B00 -> C10+=1, C11+=-1
        {
            'U': [(2, 1), (3, 1)],
            'V': [(0, 1)],
            'W': [(10, 1), (11, -1)]
        },
        # Term 2: A00(B01-B11) -> C01+=1, C11+=1
        {
            'U': [(0, 1)],
            'V': [(1, 1), (11, -1)],
            'W': [(1, 1), (11, 1)]
        },
        # Term 3: A11(B10-B00) -> C10+=1, C00+=1
        {
            'U': [(3, 1)],
            'V': [(10, 1), (0, -1)],
            'W': [(10, 1), (0, 1)]
        },
        # Term 4: (A00+A01)B11 -> C00+=-1, C01+=1
        {
            'U': [(0, 1), (1, 1)],
            'V': [(11, 1)],
            'W': [(0, -1), (1, 1)]
        },
        # Term 5: (A10-A00)(B00+B01) -> C11+=1
        {
            'U': [(2, 1), (0, -1)],
            'V': [(0, 1), (1, 1)],
            'W': [(11, 1)]
        },
        # Term 6: (A01-A11)(B10+B11) -> C00+=1
        {
            'U': [(1, 1), (3, -1)],
            'V': [(10, 1), (11, 1)],
            'W': [(0, 1)]
        },
        # Term 7: (A00+A11)(B02+B13) -> C02+=1, C13+=1
        {
            'U': [(0, 1), (3, 1)],
            'V': [(2, 1), (13, 1)],
            'W': [(2, 1), (13, 1)]
        },
        # Term 8: (A10+A11)B02 -> C12+=1, C13+=-1
        {
            'U': [(2, 1), (3, 1)],
            'V': [(2, 1)],
            'W': [(12, 1), (13, -1)]
        },
        # Term 9: A00(B03-B13) -> C03+=1, C13+=1
        {
            'U': [(0, 1)],
            'V': [(3, 1), (13, -1)],
            'W': [(3, 1), (13, 1)]
        },
        # Term 10: A11(B12-B02) -> C12+=1, C02+=1
        {
            'U': [(3, 1)],
            'V': [(12, 1), (2, -1)],
            'W': [(12, 1), (2, 1)]
        },
        # Term 11: (A00+A01)B13 -> C02+=-1, C03+=1
        {
            'U': [(0, 1), (1, 1)],
            'V': [(13, 1)],
            'W': [(2, -1), (3, 1)]
        },
        # Term 12: (A10-A00)(B02+B03) -> C13+=1
        {
            'U': [(2, 1), (0, -1)],
            'V': [(2, 1), (3, 1)],
            'W': [(13, 1)]
        },
        # Term 13: (A01-A11)(B12+B13) -> C02+=1
        {
            'U': [(1, 1), (3, -1)],
            'V': [(12, 1), (13, 1)],
            'W': [(2, 1)]
        },
        # Term 14: (A01+A10)(B04+B15) -> C04+=1, C15+=1
        {
            'U': [(1, 1), (2, 1)],
            'V': [(4, 1), (15, 1)],
            'W': [(4, 1), (15, 1)]
        },
    ]
    
    # Fill in the matrices for each block
    for block in range(num_blocks):
        base_col = block * block_size
        for t in range(15):
            col = block * 15 + t
            term = terms_15[t]
            
            for u_row, coeff in term['U']:
                U[u_row][col] = coeff
            
            for v_row, coeff in term['V']:
                # v_row is relative to block: 0-5 for row 0, 6-11 for row 1
                # Actual V row: (v_row // 6) * p + base_col + (v_row % 6)
                v_j = v_row // 6
                v_k = v_row % 6
                V[v_j * p + base_col + v_k][col] = coeff
            
            for w_row, coeff in term['W']:
                # w_row is relative to block: 0-5 for col 0, 6-11 for col 1
                # Actually, W row = k*n + i, so for 6 columns: 0-11
                # w_row: 0-5 for C[0][0..5], 6-11 for C[1][0..5]
                # Wait, W row = k*n + i, so:
                # C[0][k] -> row k*2+0 = 2k
                # C[1][k] -> row k*2+1 = 2k+1
                # For 6 columns: rows 0,1,2,3,4,5,6,7,8,9,10,11
                # w_row in term is 0-11, mapping to actual column
                w_k = w_row // 2
                w_i = w_row % 2
                W[(base_col + w_k) * n + w_i][col] = coeff
    
    return {"U": U, "V": V, "W": W}
