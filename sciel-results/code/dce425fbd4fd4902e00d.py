import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    n, m, p = 2, 4, 5
    rank = 36
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Helper to add a 2x2x2 Strassen block
    # A is 2x2, B is 2x2, C is 2x2
    # Indices for A: (0,0), (0,1), (1,0), (1,1) -> flattened 0..3
    # Indices for B: (0,0)..(1,1) -> flattened 0..3
    # Indices for C: (0,0)..(1,1) -> flattened 0..3
    
    def add_strassen(a_offset, b_offset, c_offset):
        nonlocal col
        # M1 = (A00+A11)(B00+B11) -> C00 += M1, C01 += M1, C10 += M1, C11 += M1 (wait, standard Strassen formulas)
        # Standard Strassen:
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B10+B11)
        # M4 = A11(B00-B10)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        
        # We need to map these to the global U, V, W.
        # U rows correspond to A elements.
        # V rows correspond to B elements.
        # W rows correspond to C elements.
        
        # For each Strassen term, we add to U, V, W.
        
        # Term 1: (A00+A11)(B00+B11)
        U[a_offset+0, col] += 1
        U[a_offset+3, col] += 1
        V[b_offset+0, col] += 1
        V[b_offset+3, col] += 1
        W[c_offset+0, col] += 1 # C00
        W[c_offset+1, col] += 1 # C01
        W[c_offset+2, col] += 1 # C10
        W[c_offset+3, col] += 1 # C11
        col += 1
        
        # Term 2: (A10+A11)B00
        U[a_offset+2, col] += 1
        U[a_offset+3, col] += 1
        V[b_offset+0, col] += 1
        W[c_offset+0, col] += 1 # C00? No, M2 is in C10 and C11.
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
        # So W[c_offset+2] += 1, W[c_offset+3] -= 1
        W[c_offset+2, col] += 1
        W[c_offset+3, col] -= 1
        col += 1
        
        # Term 3: A00(B10+B11)
        U[a_offset+0, col] += 1
        V[b_offset+2, col] += 1
        V[b_offset+3, col] += 1
        W[c_offset+0, col] += 1 # C00? No.
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # M3 is in C01.
        W[c_offset+1, col] += 1
        col += 1
        
        # Term 4: A11(B00-B10)
        U[a_offset+3, col] += 1
        V[b_offset+0, col] += 1
        V[b_offset+2, col] -= 1
        W[c_offset+0, col] += 1 # C00
        W[c_offset+2, col] += 1 # C10
        col += 1
        
        # Term 5: (A00+A01)B11
        U[a_offset+0, col] += 1
        U[a_offset+1, col] += 1
        V[b_offset+3, col] += 1
        W[c_offset+0, col] -= 1 # C00
        W[c_offset+1, col] += 1 # C01
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        U[a_offset+2, col] += 1
        U[a_offset+0, col] -= 1
        V[b_offset+0, col] += 1
        V[b_offset+1, col] += 1
        W[c_offset+3, col] += 1 # C11
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        U[a_offset+1, col] += 1
        U[a_offset+3, col] -= 1
        V[b_offset+2, col] += 1
        V[b_offset+3, col] += 1
        W[c_offset+0, col] += 1 # C00
        col += 1

    def add_schoolbook_2x2x1(a_offset, b_offset, c_offset):
        # A is 2x2, B is 2x1, C is 2x1
        # C0 = A00 B0 + A01 B1
        # C1 = A10 B0 + A11 B1
        nonlocal col
        # Term 1: A00 B0
        U[a_offset+0, col] = 1
        V[b_offset+0, col] = 1
        W[c_offset+0, col] = 1
        col += 1
        
        # Term 2: A01 B1
        U[a_offset+1, col] = 1
        V[b_offset+1, col] = 1
        W[c_offset+0, col] = 1
        col += 1
        
        # Term 3: A10 B0
        U[a_offset+2, col] = 1
        V[b_offset+0, col] = 1
        W[c_offset+1, col] = 1
        col += 1
        
        # Term 4: A11 B1
        U[a_offset+3, col] = 1
        V[b_offset+1, col] = 1
        W[c_offset+1, col] = 1
        col += 1

    # A is 2x4. Rows 0,1. Cols 0..3.
    # A1 is cols 0,1. Offset 0.
    # A2 is cols 2,3. Offset 2.
    
    # B is 4x5. Rows 0..3. Cols 0..4.
    # B1 is rows 0,1. Offset 0.
    # B2 is rows 2,3. Offset 5 (since 2 rows * 5 cols = 10? No. B is 4x5. Row 0,1 are first 5 cols. Row 2,3 are next 5 cols.
    # V index = row * p + col.
    # B1 rows 0,1. V indices 0..4.
    # B2 rows 2,3. V indices 10..14.
    
    # C is 2x5.
    # C1 is from A1 B1.
    # C2 is from A2 B2.
    # W index = col * n + row.
    # C1 and C2 add to the same C.
    
    # Process A1 B1
    # B1 is 2x5. Split into B1a (2x2), B1b (2x2), B1c (2x1).
    # B1a cols 0,1. V offset 0.
    # B1b cols 2,3. V offset 2.
    # B1c col 4. V offset 4.
    
    # C1a is 2x2. W offset 0.
    # C1b is 2x2. W offset 2.
    # C1c is 2x1. W offset 4.
    
    add_strassen(0, 0, 0) # A1, B1a, C1a
    add_strassen(0, 2, 2) # A1, B1b, C1b
    add_schoolbook_2x2x1(0, 4, 4) # A1, B1c, C1c
    
    # Process A2 B2
    # A2 offset 2.
    # B2 rows 2,3. V offset 10.
    # B2a cols 0,1. V offset 10.
    # B2b cols 2,3. V offset 12.
    # B2c col 4. V offset 14.
    
    # C2 adds to C.
    # C2a is 2x2. W offset 0.
    # C2b is 2x2. W offset 2.
    # C2c is 2x1. W offset 4.
    
    add_strassen(2, 10, 0) # A2, B2a, C2a
    add_strassen(2, 12, 2) # A2, B2b, C2b
    add_schoolbook_2x2x1(2, 14, 4) # A2, B2c, C2c
    
    return {"U": U, "V": V, "W": W}
