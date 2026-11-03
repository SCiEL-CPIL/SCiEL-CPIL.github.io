import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    
    # Define the Strassen decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(f-h)
    # M4 = d(g-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # We need to map this to the U, V, W format.
    # U is (n*m, rank) = (16, R)
    # V is (m*p, rank) = (28, R)
    # W is (p*n, rank) = (28, R)
    
    # The indices for U, V, W are:
    # U[i*m + j, r]
    # V[j*p + k, r]
    # W[k*n + i, r]
    
    # For a 2x2x2 block at position (I, J, K) where I, J, K are 0 or 1:
    # A block is at rows 2I, 2I+1 and cols 2J, 2J+1.
    # B block is at rows 2J, 2J+1 and cols 2K, 2K+1.
    # C block is at rows 2I, 2I+1 and cols 2K, 2K+1.
    
    # We will build the full U, V, W matrices by summing contributions from each block.
    
    total_rank = 0
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a decomposition
    def add_decomp(U_block, V_block, W_block, rank):
        # U_block: (4, rank) corresponding to local A indices
        # V_block: (4, rank) corresponding to local B indices? No, B is 2x3 or 2x2.
        # This is getting complicated with the global indices.
        pass

    # Simpler approach:
    # Iterate over all possible rank-1 terms in the block decomposition.
    # For each block, generate its local U, V, W vectors and place them in the global matrices.
    
    # Global dimensions:
    # U: 16 x R
    # V: 28 x R
    # W: 28 x R
    
    # We don't know R in advance, so we collect terms in lists.
    terms_U = []
    terms_V = []
    terms_W = []
    
    # Strassen terms for 2x2x2
    # Local indices:
    # A: 0,1,2,3 (row-major: 00, 01, 10, 11)
    # B: 0,1,2,3 (row-major: 00, 01, 10, 11)
    # C: 0,1,2,3 (row-major: 00, 01, 10, 11) -> W index is k*n+i, so for 2x2, k*2+i.
    # C[0,0] -> W[0]
    # C[0,1] -> W[1]
    # C[1,0] -> W[2]
    # C[1,1] -> W[3]
    
    # Strassen multipliers:
    # M1: A=(1,0,0,1), B=(1,0,0,1) -> (a+d)(e+h)
    # M2: A=(0,0,1,1), B=(1,0,0,0) -> (c+d)e
    # M3: A=(1,0,0,0), B=(0,1,0,-1) -> a(f-h)
    # M4: A=(0,0,0,1), B=(0,0,1,-1) -> d(g-e)
    # M5: A=(1,1,0,0), B=(0,0,0,1) -> (a+b)h
    # M6: A=(0,-1,1,0), B=(1,1,0,0) -> (c-a)(e+f)
    # M7: A=(0,1,0,-1), B=(0,0,1,1) -> (b-d)(g+h)
    
    # C coefficients:
    # C11 (W0): M1(1) + M4(1) - M5(1) + M7(1)
    # C12 (W1): M3(1) + M5(1)
    # C21 (W2): M2(1) + M4(1)
    # C22 (W3): M1(1) - M2(1) + M3(1) + M6(1)
    
    strassen_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [0, -1, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    strassen_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, 0, 1, -1],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    strassen_W = [
        [1, 1, 0, 0], # M1
        [0, 0, 1, 0], # M2
        [0, 0, 0, 1], # M3
        [0, 0, 0, 0], # M4 -- Wait, W is 4x7? No, local W is 4x4 for 2x2x2?
        # W index is k*2+i.
        # C11: k=0, i=0 -> 0
        # C12: k=1, i=0 -> 1
        # C21: k=0, i=1 -> 2
        # C22: k=1, i=1 -> 3
    ]
    # Let's re-verify W coefficients for Strassen.
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # W matrix for Strassen (4 rows, 7 cols):
    # Row 0 (C11): [1, 0, 0, 1, -1, 0, 1]
    # Row 1 (C12): [0, 0, 1, 0, 1, 0, 0]
    # Row 2 (C21): [0, 1, 0, 1, 0, 0, 0]
    # Row 3 (C22): [1, -1, 1, 0, 0, 1, 0]
    
    strassen_W = [
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ]
    
    # Schoolbook for 2x2x3
    # A: 2x2, B: 2x3, C: 2x3
    # A indices: 0,1,2,3
    # B indices: 0..5 (row-major 2x3)
    # C indices: 0..5 (row-major 2x3) -> W index k*2+i
    
    # Terms:
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C02 = A00*B02 + A01*B12
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # C12 = A10*B02 + A11*B12
    
    # 12 terms.
    # Term 0: A00*B00 -> C00
    # U: [1,0,0,0], V: [1,0,0,0,0,0], W: [1,0,0,0,0,0] (C00 is W[0])
    # Term 1: A01*B10 -> C00
    # U: [0,1,0,0], V: [0,0,1,0,0,0], W: [1,0,0,0,0,0]
    # ...
    
    sb_2x2x3_U = [
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1], # A11
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1], # A11
        [1, 0, 0, 0], # A00
        [0, 1, 0, 0], # A01
        [0, 0, 1, 0], # A10
        [0, 0, 0, 1]  # A11
    ]
    sb_2x2x3_V = [
        [1, 0, 0, 0, 0, 0], # B00
        [0, 0, 1, 0, 0, 0], # B10
        [0, 1, 0, 0, 0, 0], # B01
        [0, 0, 0, 1, 0, 0], # B11
        [0, 0, 0, 0, 1, 0], # B02
        [0, 0, 0, 0, 0, 1], # B12
        [1, 0, 0, 0, 0, 0], # B00
        [0, 0, 1, 0, 0, 0], # B10
        [0, 1, 0, 0, 0, 0], # B01
        [0, 0, 0, 1, 0, 0], # B11
        [0, 0, 0, 0, 1, 0], # B02
        [0, 0, 0, 0, 0, 1]  # B12
    ]
    # W for 2x2x3 (6 rows, 12 cols)
    # C00 (W0): T0, T1
    # C01 (W1): T2, T3
    # C02 (W2): T4, T5
    # C10 (W2? No, C10 is row 1, col 0. W index k*2+i. k=0, i=1 -> 1? No.
    # C is 2x3.
    # C[0,0] -> W[0]
    # C[0,1] -> W[1]
    # C[0,2] -> W[2]
    # C[1,0] -> W[3] (k=0, i=1)
    # C[1,1] -> W[4] (k=1, i=1)
    # C[1,2] -> W[5] (k=2, i=1)
    
    sb_2x2x3_W = [
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # C00
        [0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0], # C01
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0], # C02
        [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], # C10
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0], # C11
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1]  # C12
    ]
    
    # Now assemble the global matrices.
    # Blocks:
    # 4x4 A, 4x7 B, 4x7 C
    # A blocks: (0,0), (0,1), (1,0), (1,1)
    # B blocks: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    # C blocks: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2)
    
    # C00 = A00 B00 + A01 B10 (2x2x2 + 2x2x2)
    # C01 = A00 B01 + A01 B11 (2x2x2 + 2x2x2)
    # C02 = A00 B02 + A01 B12 (2x2x3 + 2x2x3)
    # C10 = A10 B00 + A11 B10 (2x2x2 + 2x2x2)
    # C11 = A10 B01 + A11 B11 (2x2x2 + 2x2x2)
    # C12 = A10 B02 + A11 B12 (2x2x3 + 2x2x3)
    
    # We need to map local indices to global indices.
    
    def map_U(local_idx, block_i, block_j):
        # local_idx 0..3
        # block_i 0..1, block_j 0..1
        # global row = (block_i * 2 + local_row) * 4 + (block_j * 2 + local_col)
        # local_row = local_idx // 2
        # local_col = local_idx % 2
        lr = local_idx // 2
        lc = local_idx % 2
        gr = block_i * 2 + lr
        gc = block_j * 2 + lc
        return gr * 4 + gc
        
    def map_V(local_idx, block_j, block_k):
        # local_idx 0..5 for 2x3, 0..3 for 2x2
        # block_j 0..1 (row of B), block_k 0..2 (col block of B)
        # B is 4x7.
        # Block j is rows 2j, 2j+1.
        # Block k:
        # k=0: cols 0,1
        # k=1: cols 2,3
        # k=2: cols 4,5,6
        # local_idx for 2x2:
        # lr = local_idx // 2, lc = local_idx % 2
        # global row = 2*block_j + lr
        # global col:
        # if k=0: 2*lc
        # if k=1: 2 + 2*lc
        # if k=2: 4 + 2*lc (for lc=0,1) or 4+2*lc+1?
        # Wait, 2x3 block has 6 elements.
        # local_idx 0..5.
        # lr = local_idx // 3, lc = local_idx % 3
        # global row = 2*block_j + lr
        # global col:
        # if k=0: 2*lc
        # if k=1: 2 + 2*lc
        # if k=2: 4 + lc
        
        if local_idx < 4:
            # 2x2
            lr = local_idx // 2
            lc = local_idx % 2
            gr = 2 * block_j + lr
            if block_k == 0: gc = 2 * lc
            elif block_k == 1: gc = 2 + 2 * lc
            else: gc = 4 + lc # This is wrong for 2x3 mapping
        else:
            # 2x3
            lr = local_idx // 3
            lc = local_idx % 3
            gr = 2 * block_j + lr
            if block_k == 0: gc = 2 * lc
            elif block_k == 1: gc = 2 + 2 * lc
            else: gc = 4 + lc
        return gr * 7 + gc

    def map_W(local_idx, block_k, block_i):
        # local_idx 0..5 for 2x3, 0..3 for 2x2
        # block_k 0..2 (col of C), block_i 0..1 (row of C)
        # C is 4x7.
        # Block i is rows 2i, 2i+1.
        # Block k:
        # k=0: cols 0,1
        # k=1: cols 2,3
        # k=2: cols 4,5,6
        
        if local_idx < 4:
            # 2x2
            lr = local_idx // 2
            lc = local_idx % 2
            gr = 2 * block_i + lr
            if block_k == 0: gc = 2 * lc
            elif block_k == 1: gc = 2 + 2 * lc
            else: gc = 4 + lc
        else:
            # 2x3
            lr = local_idx // 3
            lc = local_idx % 3
            gr = 2 * block_i + lr
            if block_k == 0: gc = 2 * lc
            elif block_k == 1: gc = 2 + 2 * lc
            else: gc = 4 + lc
            
        # W index is k*n + i.
        # n=4.
        # global col is gc, global row is gr.
        # W index = gc * 4 + gr
        return gc * 4 + gr

    # Build the matrices
    # We will append columns to U, V, W.
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    # Add Strassen for C00
    # A00 B00
    for r in range(7):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 0, 0)] += strassen_U[r][li]
        for li in range(4):
            v[map_V(li, 0, 0)] += strassen_V[r][li]
        for li in range(4):
            w[map_W(li, 0, 0)] += strassen_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    # A01 B10
    for r in range(7):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 0, 1)] += strassen_U[r][li]
        for li in range(4):
            v[map_V(li, 1, 0)] += strassen_V[r][li]
        for li in range(4):
            w[map_W(li, 0, 0)] += strassen_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    # ... repeat for other blocks ...
    
    # This is tedious to write out manually. I will generate it in a loop.
    
    # C01: A00 B01 + A01 B11
    for r in range(7):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 0, 0)] += strassen_U[r][li]
        for li in range(4):
            v[map_V(li, 0, 1)] += strassen_V[r][li]
        for li in range(4):
            w[map_W(li, 1, 0)] += strassen_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    for r in range(7):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 0, 1)] += strassen_U[r][li]
        for li in range(4):
            v[map_V(li, 1, 1)] += strassen_V[r][li]
        for li in range(4):
            w[map_W(li, 1, 0)] += strassen_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    # C02: A00 B02 + A01 B12 (2x2x3)
    for r in range(12):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 0, 0)] += sb_2x2x3_U[r][li]
        for li in range(6):
            v[map_V(li, 0, 2)] += sb_2x2x3_V[r][li]
        for li in range(6):
            w[map_W(li, 2, 0)] += sb_2x2x3_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    for r in range(12):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 0, 1)] += sb_2x2x3_U[r][li]
        for li in range(6):
            v[map_V(li, 1, 2)] += sb_2x2x3_V[r][li]
        for li in range(6):
            w[map_W(li, 2, 0)] += sb_2x2x3_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    # C10: A10 B00 + A11 B10
    for r in range(7):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 1, 0)] += strassen_U[r][li]
        for li in range(4):
            v[map_V(li, 0, 0)] += strassen_V[r][li]
        for li in range(4):
            w[map_W(li, 0, 1)] += strassen_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    for r in range(7):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 1, 1)] += strassen_U[r][li]
        for li in range(4):
            v[map_V(li, 1, 0)] += strassen_V[r][li]
        for li in range(4):
            w[map_W(li, 0, 1)] += strassen_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    # C11: A10 B01 + A11 B11
    for r in range(7):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 1, 0)] += strassen_U[r][li]
        for li in range(4):
            v[map_V(li, 0, 1)] += strassen_V[r][li]
        for li in range(4):
            w[map_W(li, 1, 1)] += strassen_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    for r in range(7):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 1, 1)] += strassen_U[r][li]
        for li in range(4):
            v[map_V(li, 1, 1)] += strassen_V[r][li]
        for li in range(4):
            w[map_W(li, 1, 1)] += strassen_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    # C12: A10 B02 + A11 B12
    for r in range(12):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 1, 0)] += sb_2x2x3_U[r][li]
        for li in range(6):
            v[map_V(li, 0, 2)] += sb_2x2x3_V[r][li]
        for li in range(6):
            w[map_W(li, 2, 1)] += sb_2x2x3_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    for r in range(12):
        u = [0]*16; v = [0]*28; w = [0]*28
        for li in range(4):
            u[map_U(li, 1, 1)] += sb_2x2x3_U[r][li]
        for li in range(6):
            v[map_V(li, 1, 2)] += sb_2x2x3_V[r][li]
        for li in range(6):
            w[map_W(li, 2, 1)] += sb_2x2x3_W[r][li]
        U_cols.append(u); V_cols.append(v); W_cols.append(w)
        
    # Convert to numpy arrays
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

