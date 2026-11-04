import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    # Target rank 44
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block
    # local_i, local_j, local_k are offsets in global i, j, k
    def add_strassen(oi, oj, ok):
        nonlocal col
        # Strassen coefficients for U (A), V (B), W (C)
        # A indices: i, j. B indices: j, k. C indices: i, k.
        # Local i in {0,1}, j in {0,1}, k in {0,1}
        
        # M1
        # U: (0,0)+1, (1,1)+1
        # V: (0,0)+1, (1,1)+1
        # W: (0,0)+1, (0,1)-1, (1,0)-1, (1,1)+1  -> Wait, W is C.
        # C00 += M1, C01 -= M1, C10 -= M1, C11 += M1?
        # Let's check C formulas again.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # M1: A00+A11, B00+B11
        U[oi*4 + oj][col] += 1 # A00 (i=0, j=0)
        U[(oi+1)*4 + oj+1][col] += 1 # A11 (i=1, j=1)
        
        V[oj*4 + ok][col] += 1 # B00 (j=0, k=0)
        V[(oj+1)*4 + ok+1][col] += 1 # B11 (j=1, k=1)
        
        # W contributions for M1:
        # C00: +1, C01: 0, C10: 0, C11: 0?
        # No, C00 = M1 + ...
        # C11 = M1 - ...
        # So W[ok*3 + oi] += 1 (C00)
        # W[(ok+1)*3 + oi+1] += 1 (C11)
        W[ok*3 + oi][col] += 1
        W[(ok+1)*3 + oi+1][col] += 1
        
        col += 1
        
        # M2: A10+A11, B00
        U[oi*4 + oj+1][col] += 1 # A10 (i=0? No, A10 is i=1, j=0)
        # Wait, local i=1, j=0.
        # Global i = oi+1, j = oj.
        U[(oi+1)*4 + oj][col] += 1
        U[(oi+1)*4 + oj+1][col] += 1 # A11
        
        V[oj*4 + ok][col] += 1 # B00
        
        # C10 += M2
        W[ok*3 + oi+1][col] += 1
        # C11 -= M2
        W[(ok+1)*3 + oi+1][col] -= 1
        
        col += 1
        
        # M3: A00, B10-B11
        U[oi*4 + oj][col] += 1 # A00
        
        V[(oj+1)*4 + ok][col] += 1 # B10 (j=1, k=0)
        V[(oj+1)*4 + ok+1][col] -= 1 # B11 (j=1, k=1)
        
        # C01 += M3
        W[(ok+1)*3 + oi][col] += 1
        # C11 += M3
        W[(ok+1)*3 + oi+1][col] += 1
        
        col += 1
        
        # M4: A11, B01-B00
        U[(oi+1)*4 + oj+1][col] += 1 # A11
        
        V[oj*4 + ok+1][col] += 1 # B01 (j=0, k=1)
        V[oj*4 + ok][col] -= 1 # B00
        
        # C00 += M4
        W[ok*3 + oi][col] += 1
        # C10 += M4
        W[ok*3 + oi+1][col] += 1
        
        col += 1
        
        # M5: A00+A01, B11
        U[oi*4 + oj][col] += 1 # A00
        U[oi*4 + oj+1][col] += 1 # A01 (i=0, j=1)
        
        V[(oj+1)*4 + ok+1][col] += 1 # B11
        
        # C00 -= M5
        W[ok*3 + oi][col] -= 1
        # C01 += M5
        W[(ok+1)*3 + oi][col] += 1
        
        col += 1
        
        # M6: A10-A00, B00+B10
        U[(oi+1)*4 + oj][col] += 1 # A10
        U[oi*4 + oj][col] -= 1 # A00
        
        V[oj*4 + ok][col] += 1 # B00
        V[(oj+1)*4 + ok][col] += 1 # B10
        
        # C11 += M6
        W[(ok+1)*3 + oi+1][col] += 1
        
        col += 1
        
        # M7: A01-A11, B01+B11
        U[oi*4 + oj+1][col] += 1 # A01
        U[(oi+1)*4 + oj+1][col] -= 1 # A11
        
        V[oj*4 + ok+1][col] += 1 # B01
        V[(oj+1)*4 + ok+1][col] += 1 # B11
        
        # C00 += M7
        W[ok*3 + oi][col] += 1
        
        col += 1

    # Helper to add simple 1x1x1 multiplication
    def add_simple(oi, oj, ok):
        nonlocal col
        U[oi*4 + oj][col] = 1
        V[oj*4 + ok][col] = 1
        W[ok*3 + oi][col] = 1
        col += 1

    # Part 1: C_top (i=0,1)
    # A_top_L (i=0,1; j=0,1) * B_T (j=0,1; k=0,3)
    # Split B_T into k=0,1 and k=2,3
    # Block 1.1: A_top_L * B_T_L (j=0,1; k=0,1)
    add_strassen(0, 0, 0)
    # Block 1.2: A_top_L * B_T_R (j=0,1; k=2,3)
    add_strassen(0, 0, 2)
    
    # A_top_R (i=0,1; j=2,3) * B_B (j=2,3; k=0,3)
    # Split B_B into k=0,1 and k=2,3
    # Block 1.3: A_top_R * B_B_L (j=2,3; k=0,1)
    add_strassen(0, 2, 0)
    # Block 1.4: A_top_R * B_B_R (j=2,3; k=2,3)
    add_strassen(0, 2, 2)
    
    # Part 2: C_bot (i=2)
    # A_bot (i=2; j=0,1,2,3) * B (j=0,1,2,3; k=0,1,2,3)
    # 16 simple multiplications
    for j in range(4):
        for k in range(4):
            add_simple(2, j, k)
            
    return {"U": U, "V": V, "W": W}
