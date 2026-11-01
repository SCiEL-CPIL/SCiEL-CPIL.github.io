import time

def construct():
    """
    Rank-42 exact decomposition for 2x4 @ 4x6 matrix multiplication using Strassen.
    
    Strategy: Split the m=4 dimension into two 2x2 blocks (A1, A2) and the p=6 
    dimension into three 2x2 blocks (B1, B2, B3). For each of the 3 output column 
    blocks, compute A1*Br_top + A2*Br_bot using two independent Strassen 2x2x2 
    decompositions (7 multiplications each), giving 3 * 14 = 42 total rank.
    
    U: (n*m, rank) = (8, 42)
    V: (m*p, rank) = (24, 42)
    W: (p*n, rank) = (12, 42)
    
    W index convention: W[k*n+i, r] where k is column index (0..5), i is row index (0..1)
    """
    n, m, p = 2, 4, 6
    rank = 42
    
    U = [[0] * rank for _ in range(n * m)]  # 8 x 42
    V = [[0] * rank for _ in range(m * p)]  # 24 x 42
    W = [[0] * rank for _ in range(p * n)]  # 12 x 42
    
    # Strassen 2x2x2 terms: each term has
    # 'A': list of (row, col, sign) for the first 2x2 matrix
    # 'B': list of (row, col, sign) for the second 2x2 matrix
    # 'W': list of (out_row, out_col, sign) for output contributions
    #
    # Standard Strassen:
    # M1 = (X00+X11)*(Y00+Y11) -> Z00+=1, Z11+=1
    # M2 = (X10+X11)*Y00       -> Z10+=1, Z11-=1
    # M3 = X00*(Y01-Y11)       -> Z01+=1, Z11+=1
    # M4 = X11*(Y10-Y00)       -> Z00+=1, Z10+=1
    # M5 = (X00+X01)*Y11       -> Z00-=1, Z01+=1
    # M6 = (X10-X00)*(Y00+Y01) -> Z11+=1
    # M7 = (X01-X11)*(B10+B11) -> Z00+=1
    
    strassen_terms = [
        {  # M1
            'A': [(0, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1), (1, 1, 1)]
        },
        {  # M2
            'A': [(1, 0, 1), (1, 1, 1)],
            'B': [(0, 0, 1)],
            'W': [(1, 0, 1), (1, 1, -1)]
        },
        {  # M3
            'A': [(0, 0, 1)],
            'B': [(0, 1, 1), (1, 1, -1)],
            'W': [(0, 1, 1), (1, 1, 1)]
        },
        {  # M4
            'A': [(1, 1, 1)],
            'B': [(1, 0, 1), (0, 0, -1)],
            'W': [(0, 0, 1), (1, 0, 1)]
        },
        {  # M5
            'A': [(0, 0, 1), (0, 1, 1)],
            'B': [(1, 1, 1)],
            'W': [(0, 0, -1), (0, 1, 1)]
        },
        {  # M6
            'A': [(1, 0, 1), (0, 0, -1)],
            'B': [(0, 0, 1), (0, 1, 1)],
            'W': [(1, 1, 1)]
        },
        {  # M7
            'A': [(0, 1, 1), (1, 1, -1)],
            'B': [(1, 0, 1), (1, 1, 1)],
            'W': [(0, 0, 1)]
        }
    ]
    
    col = 0
    
    # For each of the 3 column blocks of B/C:
    for r in range(3):
        b_col_offset = 2 * r  # B column offset for this block
        
        # Sub-block 1: A1 (A[:, 0:2]) * Br_top (B[0:2, 2r:2r+2])
        # A1[i, j] is at global A position (i, j) -> U row = i*4 + j
        # Br_top[j, kb] is at global B position (j, 2r+kb) -> V row = j*6 + (2r+kb)
        # Output (i, 2r+kb) -> W row = (2r+kb)*2 + i
        
        for term in strassen_terms:
            for (ar, ac, sign) in term['A']:
                u_idx = ar * m + ac  # A1 is at A[:, 0:2]
                U[u_idx][col] += sign
            for (br, bc, sign) in term['B']:
                v_idx = br * p + (b_col_offset + bc)  # Br_top is at B[0:2, 2r:2r+2]
                V[v_idx][col] += sign
            for (or_, oc, sign) in term['W']:
                w_idx = (b_col_offset + oc) * n + or_  # Output col = 2r+oc, row = or_
                W[w_idx][col] += sign
            col += 1
        
        # Sub-block 2: A2 (A[:, 2:4]) * Br_bot (B[2:4, 2r:2r+2])
        # A2[i, j] is at global A position (i, j+2) -> U row = i*4 + (j+2)
        # Br_bot[j, kb] is at global B position (j+2, 2r+kb) -> V row = (j+2)*6 + (2r+kb)
        # Output (i, 2r+kb) -> W row = (2r+kb)*2 + i (same output as sub-block 1)
        
        for term in strassen_terms:
            for (ar, ac, sign) in term['A']:
                u_idx = ar * m + (ac + 2)  # A2 is at A[:, 2:4]
                U[u_idx][col] += sign
            for (br, bc, sign) in term['B']:
                v_idx = (br + 2) * p + (b_col_offset + bc)  # Br_bot is at B[2:4, 2r:2r+2]
                V[v_idx][col] += sign
            for (or_, oc, sign) in term['W']:
                w_idx = (b_col_offset + oc) * n + or_  # Same output position
                W[w_idx][col] += sign
            col += 1
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
