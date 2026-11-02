import numpy as np

DIMENSIONS = [4, 5, 5]

def construct():
    n, m, p = 4, 5, 5
    # Total rank estimate: 56 (Part 1) + 16 (Part 2) + 16 (Part 3) + 4 (Part 4) = 92
    rank = 92
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen terms for a 2x2x2 block
    # a_offset, b_offset, c_offset are the starting indices for the block in the global flattened vectors
    # The block is 2x2 in A, 2x2 in B, 2x2 in C
    def add_strassen_block(a_off, b_off, c_off):
        nonlocal col
        # Strassen terms
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Coefficients for A, B, C for each term
        # Term 1
        # A: (1,0,1,0) -> A00, A11
        # B: (1,0,0,1) -> B00, B11
        # C: (1,0,0,1) -> C00, C11 (with signs)
        
        # We define the local indices 0,1 for rows/cols of the 2x2 block
        # Global A index: a_off + i*2 + j (if a_off is row-major offset?)
        # Wait, the global A is flattened row-major: i*m + j.
        # So for a 2x2 block starting at row r, col c:
        # A indices: r*m+c, r*m+c+1, (r+1)*m+c, (r+1)*m+c+1
        
        # Let's define the local indices for the 2x2 block
        # A_local:
        # A00: a_off
        # A01: a_off+1
        # A10: a_off+m (if m=5, then a_off+5? No, a_off is the index of A00)
        # If A00 is at (r,c), A01 is (r,c+1), A10 is (r+1,c), A11 is (r+1,c+1).
        # Indices:
        # A00: r*m + c
        # A01: r*m + c + 1
        # A10: (r+1)*m + c
        # A11: (r+1)*m + c + 1
        
        # B_local:
        # B00: c*p + k? No. B is m x p.
        # B00 is at (c, k).
        # If block starts at col c, row k:
        # B00: c*p + k
        # B01: c*p + k + 1
        # B10: (c+1)*p + k
        # B11: (c+1)*p + k + 1
        
        # C_local:
        # C00 is at (r, k).
        # C00: k*n + r
        # C01: k*n + r + 1
        # C10: (k+1)*n + r
        # C11: (k+1)*n + r + 1
        
        # We need to pass r, c, k to the helper.
        pass

    # It's easier to just loop through the 8 blocks and call a function that takes r, c, k.
    
    def add_strassen(r, c, k):
        nonlocal col
        # A indices
        A00 = r*m + c
        A01 = r*m + c + 1
        A10 = (r+1)*m + c
        A11 = (r+1)*m + c + 1
        
        # B indices
        B00 = c*p + k
        B01 = c*p + k + 1
        B10 = (c+1)*p + k
        B11 = (c+1)*p + k + 1
        
        # C indices
        C00 = k*n + r
        C01 = k*n + r + 1
        C10 = (k+1)*n + r
        C11 = (k+1)*n + r + 1
        
        # Term 1: (A00+A11)(B00+B11)
        # C00 += 1, C11 += 1
        U[A00][col] += 1; U[A11][col] += 1
        V[B00][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1
        
        # Term 2: (A10+A11)B00
        # C10 += 1
        U[A10][col] += 1; U[A11][col] += 1
        V[B00][col] += 1
        W[C10][col] += 1
        col += 1
        
        # Term 3: A00(B01-B11)
        # C01 += 1
        U[A00][col] += 1
        V[B01][col] += 1; V[B11][col] -= 1
        W[C01][col] += 1
        col += 1
        
        # Term 4: A11(B10-B00)
        # C00 += 1, C10 += 1
        U[A11][col] += 1
        V[B10][col] += 1; V[B00][col] -= 1
        W[C00][col] += 1; W[C10][col] += 1
        col += 1
        
        # Term 5: (A00+A01)B11
        # C00 -= 1, C01 += 1
        U[A00][col] += 1; U[A01][col] += 1
        V[B11][col] += 1
        W[C00][col] -= 1; W[C01][col] += 1
        col += 1
        
        # Term 6: (A10-A00)(B00+B01)
        # C11 += 1
        U[A10][col] += 1; U[A00][col] -= 1
        V[B00][col] += 1; V[B01][col] += 1
        W[C11][col] += 1
        col += 1
        
        # Term 7: (A01-A11)(B10+B11)
        # C00 += 1, C11 += 1
        U[A01][col] += 1; U[A11][col] -= 1
        V[B10][col] += 1; V[B11][col] += 1
        W[C00][col] += 1; W[C11][col] += 1
        col += 1

    # Part 1: 8 blocks
    # C00 = A00B00 + A01B10
    add_strassen(0, 0, 0) # A00 B00 -> C00
    add_strassen(0, 2, 0) # A01 B10 -> C00 (A01 is r=0,c=2; B10 is c=2,k=0? No. B10 is row 2, col 0. c=2, k=0. Correct.)
    
    # C01 = A00B01 + A01B11
    add_strassen(0, 0, 1) # A00 B01 -> C01 (A00 r=0,c=0; B01 c=0,k=1. Correct.)
    add_strassen(0, 2, 1) # A01 B11 -> C01 (A01 r=0,c=2; B11 c=2,k=1. Correct.)
    
    # C10 = A10B00 + A11B10
    add_strassen(1, 0, 0) # A10 B00 -> C10 (A10 r=1,c=0; B00 c=0,k=0. Correct.)
    add_strassen(1, 2, 0) # A11 B10 -> C10 (A11 r=1,c=2; B10 c=2,k=0. Correct.)
    
    # C11 = A10B01 + A11B11
    add_strassen(1, 0, 1) # A10 B01 -> C11 (A10 r=1,c=0; B01 c=0,k=1. Correct.)
    add_strassen(1, 2, 1) # A11 B11 -> C11 (A11 r=1,c=2; B11 c=2,k=1. Correct.)

    # Part 2: A2 B21 (4x1x4)
    # A2 is col 4 of A. B21 is row 4 of B.
    # C10..C13 (cols 0..3)
    # C_ik = A_i4 * B_4k
    for i in range(4):
        for k in range(4):
            a_idx = i*m + 4
            b_idx = 4*p + k
            c_idx = k*n + i
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1

    # Part 3: A1 B12 (4x4x1)
    # A1 is cols 0..3. B12 is col 4 of B.
    # C_i4 = sum_j A_ij B_j4
    # This is 4 dot products.
    # We need to decompose sum_j A_ij B_j4.
    # This is rank 16.
    # Each term is A_ij * B_j4.
    for i in range(4):
        for j in range(4):
            a_idx = i*m + j
            b_idx = j*p + 4
            c_idx = 4*n + i
            U[a_idx][col] = 1
            V[b_idx][col] = 1
            W[c_idx][col] = 1
            col += 1

    # Part 4: A2 B22 (4x1x1)
    # C_i4 += A_i4 * B_44
    for i in range(4):
        a_idx = i*m + 4
        b_idx = 4*p + 4
        c_idx = 4*n + i
        U[a_idx][col] = 1
        V[b_idx][col] = 1
        W[c_idx][col] = 1
        col += 1

    return {"U": U, "V": V, "W": W}
