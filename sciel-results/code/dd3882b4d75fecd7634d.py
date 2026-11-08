import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 65: 49 for 4x4x4 (Strassen) + 16 for 4x4x1
    rank = 65
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # --- Part 1: 4x4x4 Strassen Decomposition ---
    
    # Strassen 2x2x2 decomposition
    # u, v, w are length 4 vectors (flattened 2x2 matrices)
    # Indices in 2x2: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    # M1 = (A00+A11)(B00+B11)
    # A terms: A00(1), A11(1) -> u1 = [1,0,0,1]
    # B terms: B00(1), B11(1) -> v1 = [1,0,0,1]
    # C terms: C00(1), C11(1) -> w1 = [1,0,0,1]
    # Wait, Strassen formulas:
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # We need to decompose each Mk (which is a 2x2x2 tensor) into 7 rank-1 terms.
    # Let's define the 7 rank-1 terms for a generic 2x2x2 multiplication X*Y=Z.
    # The terms are the same as M1..M7 above, but applied to X,Y,Z.
    
    strassen_2x2_terms = [
        # (u, v, w)
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]), # M1
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, 0]), # M2: (X10+X11)Y00 -> Z10
        # Wait, M2 = (A10+A11)B00.
        # A10 is index 2, A11 is index 3. B00 is index 0.
        # Result contributes to Z10 (index 2).
        # u = [0,0,1,1], v = [1,0,0,0], w = [0,0,1,0]
        
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]), # M3: A00(B01-B11) -> Z01
        # A00 index 0. B01 index 1, B11 index 3.
        # u = [1,0,0,0], v = [0,1,0,-1], w = [0,1,0,0]
        
        ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]), # M4: A11(B10-B00) -> Z00
        # A11 index 3. B10 index 2, B00 index 0.
        # u = [0,0,0,1], v = [0,0,1,-1], w = [1,0,0,0]
        
        ([1, 1, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0]), # M5: (A00+A01)B11 -> Z01
        # A00 index 0, A01 index 1. B11 index 3.
        # u = [1,1,0,0], v = [0,0,0,1], w = [0,1,0,0]
        
        ([0, 0, 1, -1], [1, 1, 0, 0], [0, 0, 0, 1]), # M6: (A10-A00)(B00+B01) -> Z11
        # A10 index 2, A00 index 0. B00 index 0, B01 index 1.
        # u = [0,0,1,-1], v = [1,1,0,0], w = [0,0,0,1]
        
        ([0, 1, 0, -1], [0, 0, 1, 1], [1, 0, 0, 0]), # M7: (A01-A11)(B10+B11) -> Z00
        # A01 index 1, A11 index 3. B10 index 2, B11 index 3.
        # u = [0,1,0,-1], v = [0,0,1,1], w = [1,0,0,0]
    ]
    
    # Now, define the 4x4x4 combinations.
    # Blocks: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # A_comb[k]: list of (block_idx, coeff)
    # B_comb[k]: list of (block_idx, coeff)
    # C_contrib[k]: list of (block_idx, coeff) for the result blocks
    
    # M1 = (A0+A3)(B0+B3)
    # A: [(0,1), (3,1)], B: [(0,1), (3,1)]
    # C: M1 contributes to C0 (+1) and C3 (+1)
    A_comb_4x4 = [
        [(0, 1), (3, 1)], # M1
        [(2, 1), (3, 1)], # M2
        [(0, 1)],         # M3
        [(3, 1)],         # M4
        [(0, 1), (1, 1)], # M5
        [(2, 1), (0, -1)],# M6
        [(1, 1), (3, -1)] # M7
    ]
    
    B_comb_4x4 = [
        [(0, 1), (3, 1)], # M1
        [(0, 1)],         # M2
        [(1, 1), (3, -1)],# M3
        [(2, 1), (0, -1)],# M4
        [(3, 1)],         # M5
        [(0, 1), (1, 1)], # M6
        [(2, 1), (3, 1)]  # M7
    ]
    
    C_contrib_4x4 = [
        [(0, 1), (3, 1)], # M1 -> C0, C3
        [(2, 1), (3, -1)],# M2 -> C2, C3
        [(1, 1), (3, 1)], # M3 -> C1, C3
        [(0, 1), (2, 1)], # M4 -> C0, C2
        [(0, -1), (1, 1)],# M5 -> C0, C1
        [(3, 1)],         # M6 -> C3
        [(0, 1)]          # M7 -> C0
    ]
    
    # Generate the 49 terms
    for k in range(7):
        a_terms = A_comb_4x4[k]
        b_terms = B_comb_4x4[k]
        c_terms = C_contrib_4x4[k]
        
        for l in range(7):
            u_2, v_2, w_2 = strassen_2x2_terms[l]
            
            U_row = [0] * (n * m) # 16
            V_row = [0] * (m * p) # 20 (but only first 16 used for cols 0-3)
            W_row = [0] * (p * n) # 20 (but only first 16 used for cols 0-3)
            
            # Fill U
            for block_idx, coeff in a_terms:
                # Block idx 0-3 corresponds to 2x2 blocks in 4x4
                # Global index in U: block_idx * 4 + offset
                for offset in range(4):
                    U_row[block_idx * 4 + offset] += coeff * u_2[offset]
            
            # Fill V
            for block_idx, coeff in b_terms:
                # Block idx 0-3 in B (4x4 part)
                # B is 4x5. Blocks are in the first 4 columns.
                # Global index in V: block_idx * 5 + offset?
                # No. B is 4x5.
                # Block 0: rows 0-1, cols 0-1. Indices: (0,0),(0,1),(1,0),(1,1) -> 0,1,5,6
                # Block 1: rows 0-1, cols 2-3. Indices: (0,2),(0,3),(1,2),(1,3) -> 2,3,7,8
                # Block 2: rows 2-3, cols 0-1. Indices: (2,0),(2,1),(3,0),(3,1) -> 10,11,15,16
                # Block 3: rows 2-3, cols 2-3. Indices: (2,2),(2,3),(3,2),(3,3) -> 12,13,17,18
                
                # V index: j * p + k. j is row of B (0-3), k is col of B (0-4).
                # Block (br, bc) where br, bc in {0,1}.
                # Rows: br*2 to br*2+1. Cols: bc*2 to bc*2+1.
                # Offset 0: (0,0) -> j=0, k=0 -> idx 0
                # Offset 1: (0,1) -> j=0, k=1 -> idx 1
                # Offset 2: (1,0) -> j=1, k=0 -> idx 5
                # Offset 3: (1,1) -> j=1, k=1 -> idx 6
                
                # We need to map block_idx and offset to global V index.
                # block_idx = br * 2 + bc
                br = block_idx // 2
                bc = block_idx % 2
                for offset in range(4):
                    # offset 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
                    # local r, c in 2x2
                    lr = offset // 2
                    lc = offset % 2
                    global_j = br * 2 + lr
                    global_k = bc * 2 + lc
                    v_idx = global_j * p + global_k
                    V_row[v_idx] += coeff * v_2[offset]
            
            # Fill W
            # W index: k * n + i. k is col of C (0-3), i is row of C (0-3).
            # C is 4x5. We are filling cols 0-3.
            # Block (cr, cc) where cr, cc in {0,1}.
            # Rows: cr*2 to cr*2+1. Cols: cc*2 to cc*2+1.
            # Offset 0: (0,0), 1: (0,1), 2: (1,0), 3: (1,1)
            # local r, c
            # global i = cr*2 + lr
            # global k = cc*2 + lc
            # W idx = k * n + i = (cc*2+lc)*4 + (cr*2+lr)
            
            for block_idx, coeff in c_terms:
                cr = block_idx // 2
                cc = block_idx % 2
                for offset in range(4):
                    lr = offset // 2
                    lc = offset % 2
                    global_i = cr * 2 + lr
                    global_k = cc * 2 + lc
                    w_idx = global_k * n + global_i
                    W_row[w_idx] += coeff * w_2[offset]
            
            # Add to global
            for i in range(len(U_row)):
                U[i][col] = U_row[i]
            for i in range(len(V_row)):
                V[i][col] = V_row[i]
            for i in range(len(W_row)):
                W[i][col] = W_row[i]
            
            col += 1
            
    # --- Part 2: 4x4x1 for the 5th column ---
    # A is 4x4, B is 4x1 (5th col), C is 4x1 (5th col)
    # 16 terms
    for i in range(n):
        for j in range(m):
            U[i * m + j][col] = 1
            V[j * p + 4][col] = 1
            W[4 * n + i][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
