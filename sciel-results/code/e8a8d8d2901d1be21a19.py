DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=10
    rank = 35
    
    # U: (n*m, rank) = (4, 35) — coefficients on A entries
    # V: (m*p, rank) = (20, 35) — coefficients on B entries
    # W: (p*n, rank) = (20, 35) — coefficients on C entries
    # Indexing: A[i,j] -> i*m+j, B[j,k] -> j*p+k, C[i,k] -> k*n+i
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 7 Strassen terms
    # For each term s and each component t (0..4):
    #   U: linear combo of A entries
    #   V: linear combo of B entries (column t within the 5-col block)
    #   W: contributes to specific C entries
    
    # A coefficients: list of (A_index, coeff) for each Strassen term
    # A indices: A00=0, A01=1, A10=2, A11=3
    A_coeffs = [
        [(0, 1), (3, 1)],      # M1: A00 + A11
        [(2, 1), (3, 1)],      # M2: A10 + A11
        [(0, 1)],               # M3: A00
        [(3, 1)],               # M4: A11
        [(0, 1), (1, 1)],      # M5: A00 + A01
        [(2, 1), (0, -1)],     # M6: A10 - A00
        [(1, 1), (3, -1)],     # M7: A01 - A11
    ]
    
    # B block coefficients: list of (B_row, block_col, coeff)
    # block_col: 0 = left half (cols 0-4), 1 = right half (cols 5-9)
    B_coeffs = [
        [(0, 0, 1), (1, 1, 1)],   # M1: B[0,0] + B[1,1]
        [(0, 0, 1)],               # M2: B[0,0]
        [(0, 1, 1), (1, 1, -1)],  # M3: B[0,1] - B[1,1]
        [(1, 0, 1), (0, 0, -1)],  # M4: B[1,0] - B[0,0]
        [(1, 1, 1)],               # M5: B[1,1]
        [(0, 0, 1), (0, 1, 1)],   # M6: B[0,0] + B[0,1]
        [(1, 0, 1), (1, 1, 1)],   # M7: B[1,0] + B[1,1]
    ]
    
    # Strassen contribution to C blocks: (C00, C01, C10, C11)
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    contrib = [
        (1,  0,  0,  1),  # M1
        (0,  0,  1, -1),  # M2
        (0,  1,  0,  1),  # M3
        (1,  0,  1,  0),  # M4
        (-1, 1,  0,  0),  # M5
        (0,  0,  0,  1),  # M6
        (1,  0,  0,  0),  # M7
    ]
    
    col = 0
    for s in range(7):
        for t in range(5):
            # Set U coefficients
            for a_idx, a_c in A_coeffs[s]:
                U[a_idx][col] = a_c
            
            # Set V coefficients
            for b_row, b_block, b_c in B_coeffs[s]:
                b_col = b_block * 5 + t
                V[b_row * p + b_col][col] = b_c
            
            # Set W coefficients
            c00_c, c01_c, c10_c, c11_c = contrib[s]
            if c00_c != 0:
                W[t * n + 0][col] = c00_c
            if c01_c != 0:
                W[(5 + t) * n + 0][col] = c01_c
            if c10_c != 0:
                W[t * n + 1][col] = c10_c
            if c11_c != 0:
                W[(5 + t) * n + 1][col] = c11_c
            
            col += 1
    
    return {"U": U, "V": V, "W": W}
