# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 2, 7
    # Rank 25: 3 blocks of Strassen (rank 7) + 1 block of Schoolbook (rank 4)
    rank = 25
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen coefficients
    # U (A) coeffs for M1..M7
    # A indices: 0:A00, 1:A01, 2:A10, 3:A11
    strassen_U = [
        [1, 0, 0, 1], # M1: A00 + A11
        [0, 0, 1, 1], # M2: A10 + A11
        [1, 0, 0, 0], # M3: A00
        [0, 0, 0, 1], # M4: A11
        [1, 1, 0, 0], # M5: A00 + A01
        [-1, 0, 1, 0],# M6: -A00 + A10
        [0, 1, 0, -1] # M7: A01 - A11
    ]
    
    # W (C) coeffs for M1..M7
    # C indices in block: 0:C00, 1:C01, 2:C10, 3:C11 (mapping to global k*n+i)
    # Wait, local mapping:
    # C00 -> k=0, i=0
    # C01 -> k=1, i=0
    # C10 -> k=0, i=1
    # C11 -> k=1, i=1
    # Global W index = k*n + i.
    # For block b, k is 2b or 2b+1.
    # So we need to add offsets.
    # Let's define local W coeffs as dict {local_c_idx: coeff}
    # local_c_idx: 0:C00, 1:C01, 2:C10, 3:C11
    strassen_W_local = [
        {0: 1, 3: 1},     # M1: C00 + C11
        {2: 1, 3: -1},    # M2: C10 - C11
        {1: 1, 3: 1},     # M3: C01 + C11
        {0: 1, 2: 1},     # M4: C00 + C10
        {0: -1, 1: 1},    # M5: -C00 + C01
        {3: 1},           # M6: C11
        {0: 1}            # M7: C00
    ]
    
    # V (B) coeffs for M1..M7
    # B indices in block: 0:B00, 1:B01, 2:B10, 3:B11
    # Global V index = j*p + k.
    # For block b, k is 2b or 2b+1.
    # j=0 -> k=2b, 2b+1
    # j=1 -> k=2b, 2b+1
    strassen_V_local = [
        {0: 1, 3: 1},     # M1: B00 + B11
        {0: 1},           # M2: B00
        {1: 1, 3: -1},    # M3: B01 - B11
        {2: 1, 0: -1},    # M4: B10 - B00
        {3: 1},           # M5: B11
        {0: 1, 1: 1},     # M6: B00 + B01
        {2: 1, 3: 1}      # M7: B10 + B11
    ]
    
    # Process 3 Strassen blocks
    for b in range(3):
        for r in range(7):
            # U
            for i in range(4):
                U[i][col] += strassen_U[r][i]
            
            # V
            # local B index 0 -> B[0, 2b] -> V[0*7 + 2b]
            # local B index 1 -> B[0, 2b+1] -> V[0*7 + 2b+1]
            # local B index 2 -> B[1, 2b] -> V[1*7 + 2b]
            # local B index 3 -> B[1, 2b+1] -> V[1*7 + 2b+1]
            for loc_idx, coeff in strassen_V_local[r].items():
                if loc_idx == 0:
                    v_idx = 0 * p + (2 * b)
                elif loc_idx == 1:
                    v_idx = 0 * p + (2 * b + 1)
                elif loc_idx == 2:
                    v_idx = 1 * p + (2 * b)
                elif loc_idx == 3:
                    v_idx = 1 * p + (2 * b + 1)
                V[v_idx][col] += coeff
            
            # W
            # local C index 0 -> C[0, 2b] -> W[2b*2 + 0]
            # local C index 1 -> C[0, 2b+1] -> W[(2b+1)*2 + 0]
            # local C index 2 -> C[1, 2b] -> W[2b*2 + 1]
            # local C index 3 -> C[1, 2b+1] -> W[(2b+1)*2 + 1]
            for loc_idx, coeff in strassen_W_local[r].items():
                if loc_idx == 0:
                    w_idx = (2 * b) * n + 0
                elif loc_idx == 1:
                    w_idx = (2 * b + 1) * n + 0
                elif loc_idx == 2:
                    w_idx = (2 * b) * n + 1
                elif loc_idx == 3:
                    w_idx = (2 * b + 1) * n + 1
                W[w_idx][col] += coeff
            
            col += 1
            
    # Schoolbook for last block (col 6)
    # S1: A00 B0 -> C0
    # S2: A01 B1 -> C0
    # S3: A10 B0 -> C1
    # S4: A11 B1 -> C1
    
    schoolbook_U = [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ]
    
    schoolbook_V_local = [
        {0: 1}, # B0
        {1: 1}, # B1
        {0: 1}, # B0
        {1: 1}  # B1
    ]
    
    schoolbook_W_local = [
        {0: 1}, # C0
        {0: 1}, # C0
        {1: 1}, # C1
        {1: 1}  # C1
    ]
    
    for r in range(4):
        # U
        for i in range(4):
            U[i][col] += schoolbook_U[r][i]
            
        # V
        # col 6
        for loc_idx, coeff in schoolbook_V_local[r].items():
            if loc_idx == 0:
                v_idx = 0 * p + 6
            elif loc_idx == 1:
                v_idx = 1 * p + 6
            V[v_idx][col] += coeff
            
        # W
        for loc_idx, coeff in schoolbook_W_local[r].items():
            if loc_idx == 0:
                w_idx = 6 * n + 0
            elif loc_idx == 1:
                w_idx = 6 * n + 1
            W[w_idx][col] += coeff
            
        col += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
