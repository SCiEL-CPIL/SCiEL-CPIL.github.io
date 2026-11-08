import numpy as np

def construct():
    n, m, p = 3, 3, 7
    rank = 54
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(u_idx, v_idx, w_idx, val):
        nonlocal col
        U[u_idx][col] += val
        V[v_idx][col] += val
        W[w_idx][col] += val
        col += 1

    # Function to add Strassen M222 decomposition for given index offsets
    def add_strassen_222(i_off, j_off, k_off):
        nonlocal col
        # M1 = (A00+A11)(B00+B11)
        # U: (0,0), (1,1)
        # V: (0,0), (1,1)
        # W: (0,0), (1,1)
        U[i_off*2 + 0][col] = 1
        U[i_off*2 + 1][col] = 1
        V[j_off*2 + 0][col] = 1
        V[j_off*2 + 1][col] = 1
        W[k_off*2 + 0][col] = 1
        W[k_off*2 + 1][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[i_off*2 + 1][col] = 1 # A10
        U[i_off*2 + 1][col] = 1 # A11? No, A10 and A11 are different indices
        # A is 2x2. Indices (i,j).
        # A10 is i=1, j=0. A11 is i=1, j=1.
        # U index is i*m + j.
        # For 2x2, m=2.
        # A10 -> i=1, j=0 -> idx 2.
        # A11 -> i=1, j=1 -> idx 3.
        # My U array is for the full 3x3 matrix.
        # So I need to map local indices to global.
        
    # Let's refactor to use global indices directly in the helper.
    
    def add_strassen_222_global(i_base, j_base, k_base):
        nonlocal col
        # M1
        U[i_base*3 + j_base][col] = 1
        U[(i_base+1)*3 + (j_base+1)][col] = 1
        V[j_base*7 + k_base][col] = 1
        V[(j_base+1)*7 + (k_base+1)][col] = 1
        W[k_base*3 + i_base][col] = 1
        W[(k_base+1)*3 + (i_base+1)][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[(i_base+1)*3 + j_base][col] = 1
        U[(i_base+1)*3 + (j_base+1)][col] = 1
        V[j_base*7 + k_base][col] = 1
        W[k_base*3 + (i_base+1)][col] = 1
        col += 1
        
        # M3 = A00(B01-B11)
        U[i_base*3 + j_base][col] = 1
        V[j_base*7 + (k_base+1)][col] = 1
        V[(j_base+1)*7 + (k_base+1)][col] = -1
        W[(k_base+1)*3 + i_base][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[(i_base+1)*3 + (j_base+1)][col] = 1
        V[(j_base+1)*7 + k_base][col] = 1
        V[j_base*7 + k_base][col] = -1
        W[k_base*3 + (i_base+1)][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[i_base*3 + j_base][col] = 1
        U[i_base*3 + (j_base+1)][col] = 1
        V[(j_base+1)*7 + (k_base+1)][col] = 1
        W[(k_base+1)*3 + i_base][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[(i_base+1)*3 + j_base][col] = 1
        U[i_base*3 + j_base][col] = -1
        V[j_base*7 + k_base][col] = 1
        V[j_base*7 + (k_base+1)][col] = 1
        W[(k_base+1)*3 + (i_base+1)][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[i_base*3 + (j_base+1)][col] = 1
        U[(i_base+1)*3 + (j_base+1)][col] = -1
        V[(j_base+1)*7 + k_base][col] = 1
        V[(j_base+1)*7 + (k_base+1)][col] = 1
        W[k_base*3 + i_base][col] = 1
        col += 1

    # Function to add M332 decomposition
    def add_m332(k_base):
        # i_base=0, j_base=0 for the 2x2 block
        add_strassen_222_global(0, 0, k_base)
        
        # a2x1 b1x2: i in {0,1}, j=2, k in {k_base, k_base+1}
        # Rank 4
        for i in range(2):
            for k in range(2):
                U[i*3 + 2][col] = 1
                V[2*7 + k_base + k][col] = 1
                W[(k_base + k)*3 + i][col] = 1
                col += 1
                
        # a1x2 B2x2: i=2, j in {0,1}, k in {k_base, k_base+1}
        # Rank 2
        for j in range(2):
            U[2*3 + j][col] = 1
            V[j*7 + k_base + j][col] = 1 # B_j, k_base+j
            W[(k_base + j)*3 + 2][col] = 1
            col += 1
            
        # a1x1 b1x2: i=2, j=2, k in {k_base, k_base+1}
        # Rank 2
        for k in range(2):
            U[2*3 + 2][col] = 1
            V[2*7 + k_base + k][col] = 1
            W[(k_base + k)*3 + 2][col] = 1
            col += 1

    # Function to add M331 decomposition
    def add_m331(k_base):
        # i_base=0, j_base=0 for 2x2 block
        # A2x2 B2x1: Rank 4
        # M1 = (A00+A11)B00
        U[0*3 + 0][col] = 1
        U[1*3 + 1][col] = 1
        V[0*7 + k_base][col] = 1
        W[k_base*3 + 0][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00? No, for 2x1, B is 2x1.
        # Strassen for 2x2 by 2x1?
        # We established rank is 4.
        # Let's just use schoolbook for the 2x2 part to be safe, or verify.
        # Schoolbook for 2x2 by 2x1:
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        # Rank 4.
        # My previous Strassen attempt for 2x2 by 2x1:
        # M1 = (A00+A11)B0
        # M2 = A01 B1
        # M3 = A10 B0
        # M4 = A11 B1?
        # Let's stick to schoolbook for M221 to avoid errors.
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        U[0*3 + 0][col] = 1
        V[0*7 + k_base][col] = 1
        W[k_base*3 + 0][col] = 1
        col += 1
        
        U[0*3 + 1][col] = 1
        V[1*7 + k_base][col] = 1
        W[k_base*3 + 0][col] = 1
        col += 1
        
        U[1*3 + 0][col] = 1
        V[0*7 + k_base][col] = 1
        W[k_base*3 + 1][col] = 1
        col += 1
        
        U[1*3 + 1][col] = 1
        V[1*7 + k_base][col] = 1
        W[k_base*3 + 1][col] = 1
        col += 1
        
        # a2x1 b1x1: i in {0,1}, j=2, k=k_base
        # Rank 2
        for i in range(2):
            U[i*3 + 2][col] = 1
            V[2*7 + k_base][col] = 1
            W[k_base*3 + i][col] = 1
            col += 1
            
        # a1x2 B2x1: i=2, j in {0,1}, k=k_base
        # Rank 2
        for j in range(2):
            U[2*3 + j][col] = 1
            V[j*7 + k_base][col] = 1
            W[k_base*3 + 2][col] = 1
            col += 1
            
        # a1x1 b1x1: i=2, j=2, k=k_base
        # Rank 1
        U[2*3 + 2][col] = 1
        V[2*7 + k_base][col] = 1
        W[k_base*3 + 2][col] = 1
        col += 1

    # Add 3 blocks of M332
    add_m332(0)
    add_m332(2)
    add_m332(4)
    
    # Add 1 block of M331
    add_m331(6)
    
    return {"U": U, "V": V, "W": W}
