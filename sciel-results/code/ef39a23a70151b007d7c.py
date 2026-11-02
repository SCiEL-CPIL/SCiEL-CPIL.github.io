import numpy as np

DIMENSIONS = [3, 4, 4]

def construct():
    # We construct a rank-28 decomposition for 4x4x4 using Strassen on 2x2 blocks
    # and then restrict it to 3x4x4.
    
    # Strassen's algorithm for 2x2 matrices:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)B11
    # M3 = A11(B12 - B22)
    # M4 = A22(B21 - B11)
    # M5 = (A11 + A12)B22
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6

    # For 4x4, we have 4 blocks of 2x2.
    # A = [[A1, A2], [A3, A4]]
    # B = [[B1, B2], [B3, B4]]
    # C = [[C1, C2], [C3, C4]]
    # C1 = A1 B1 + A2 B3
    # C2 = A1 B2 + A2 B4
    # C3 = A3 B1 + A4 B3
    # C4 = A3 B2 + A4 B4

    # We apply Strassen to each Ci.
    # Total rank 4 * 7 = 28.
    
    rank = 28
    n = 4
    m = 4
    p = 4
    
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Helper to add a Strassen decomposition for a specific block multiplication
    # A_block indices: (r0, r1) rows, (c0, c1) cols
    # B_block indices: (r0, r1) rows, (c0, c1) cols
    # C_block indices: (r0, r1) rows, (c0, c1) cols
    
    def add_strassen(A_r0, A_r1, A_c0, A_c1, B_r0, B_r1, B_c0, B_c1, C_r0, C_r1, C_c0, C_c1, offset):
        # A11, A12, A21, A22 are 2x2 blocks of the 4x4 matrix
        # A11 is at (A_r0, A_c0)
        # A12 is at (A_r0, A_c0+2)
        # A21 is at (A_r0+2, A_c0)
        # A22 is at (A_r0+2, A_c0+2)
        
        # B11 is at (B_r0, B_c0)
        # B12 is at (B_r0, B_c0+2)
        # B21 is at (B_r0+2, B_c0)
        # B22 is at (B_r0+2, B_c0+2)
        
        # C11 is at (C_r0, C_c0)
        # C12 is at (C_r0, C_c0+2)
        # C21 is at (C_r0+2, C_c0)
        # C22 is at (C_r0+2, C_c0+2)
        
        # M1
        U[A_r0*m + A_c0, offset] += 1
        U[A_r0*m + A_c0+1, offset] += 1
        U[(A_r0+1)*m + A_c0, offset] += 1
        U[(A_r0+1)*m + A_c0+1, offset] += 1
        U[A_r0*m + A_c0+2, offset] += 1
        U[A_r0*m + A_c0+3, offset] += 1
        U[(A_r0+1)*m + A_c0+2, offset] += 1
        U[(A_r0+1)*m + A_c0+3, offset] += 1
        
        V[B_r0*p + B_c0, offset] += 1
        V[B_r0*p + B_c0+1, offset] += 1
        V[(B_r0+1)*p + B_c0, offset] += 1
        V[(B_r0+1)*p + B_c0+1, offset] += 1
        V[(B_r0+2)*p + B_c0+2, offset] += 1
        V[(B_r0+2)*p + B_c0+3, offset] += 1
        V[(B_r0+3)*p + B_c0+2, offset] += 1
        V[(B_r0+3)*p + B_c0+3, offset] += 1
        
        W[C_r0*n + C_c0, offset] += 1
        W[C_r0*n + C_c0+1, offset] += 1
        W[(C_r0+1)*n + C_c0, offset] += 1
        W[(C_r0+1)*n + C_c0+1, offset] += 1
        W[(C_r0+2)*n + C_c0+2, offset] += 1
        W[(C_r0+2)*n + C_c0+3, offset] += 1
        W[(C_r0+3)*n + C_c0+2, offset] += 1
        W[(C_r0+3)*n + C_c0+3, offset] += 1
        
        # M2
        U[(A_r0+2)*m + A_c0, offset+1] += 1
        U[(A_r0+2)*m + A_c0+1, offset+1] += 1
        U[(A_r0+3)*m + A_c0, offset+1] += 1
        U[(A_r0+3)*m + A_c0+1, offset+1] += 1
        U[(A_r0+2)*m + A_c0+2, offset+1] += 1
        U[(A_r0+3)*m + A_c0+2, offset+1] += 1
        U[(A_r0+3)*m + A_c0+3, offset+1] += 1
        U[(A_r0+3)*m + A_c0+3, offset+1] += 1 # Wait, A22 is (A_r0+2, A_c0+2) to (A_r0+3, A_c0+3)
        
        # This manual indexing is prone to errors.
        # I will use a more robust construction.
        pass

    # Instead of manual indexing, I will use the known Laderman coefficients if I can find them,
    # or fall back to the block decomposition.
    
    # Let's try to implement the block decomposition correctly.
    # We will create 4 Strassen decompositions.
    
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    
    def idx_A(r, c): return r*m + c
    def idx_B(r, c): return r*p + c
    def idx_C(r, c): return r*n + c
    
    # Strassen terms for 2x2 blocks
    # A11, A12, A21, A22
    # B11, B12, B21, B22
    # C11, C12, C21, C22
    
    # M1: (A11+A22)(B11+B22)
    # M2: (A21+A22)B11
    # M3: A11(B12-B22)
    # M4: A22(B21-B11)
    # M5: (A11+A12)B22
    # M6: (A21-A11)(B11+B12)
    # M7: (A12-A22)(B21+B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    def add_block_strassen(A_r0, A_c0, B_r0, B_c0, C_r0, C_c0, offset):
        # A11: (A_r0, A_c0)
        # A12: (A_r0, A_c0+2)
        # A21: (A_r0+2, A_c0)
        # A22: (A_r0+2, A_c0+2)
        
        # B11: (B_r0, B_c0)
        # B12: (B_r0, B_c0+2)
        # B21: (B_r0+2, B_c0)
        # B22: (B_r0+2, B_c0+2)
        
        # C11: (C_r0, C_c0)
        # C12: (C_r0, C_c0+2)
        # C21: (C_r0+2, C_c0)
        # C22: (C_r0+2, C_c0+2)
        
        # M1
        for dr in range(2):
            for dc in range(2):
                U[idx_A(A_r0+dr, A_c0+dc)][offset] += 1
                U[idx_A(A_r0+2+dr, A_c0+2+dc)][offset] += 1
                V[idx_B(B_r0+dr, B_c0+dc)][offset] += 1
                V[idx_B(B_r0+2+dr, B_c0+2+dc)][offset] += 1
                W[idx_C(C_r0+dr, C_c0+dc)][offset] += 1
                W[idx_C(C_r0+2+dr, C_c0+2+dc)][offset] += 1
        
        # M2
        for dr in range(2):
            for dc in range(2):
                U[idx_A(A_r0+2+dr, A_c0+dc)][offset+1] += 1
                U[idx_A(A_r0+2+dr, A_c0+2+dc)][offset+1] += 1
                V[idx_B(B_r0+dr, B_c0+dc)][offset+1] += 1
                W[idx_C(C_r0+2+dr, C_c0+dc)][offset+1] += 1
        
        # M3
        for dr in range(2):
            for dc in range(2):
                U[idx_A(A_r0+dr, A_c0+dc)][offset+2] += 1
                V[idx_B(B_r0+dr, B_c0+2+dc)][offset+2] += 1
                V[idx_B(B_r0+2+dr, B_c0+2+dc)][offset+2] -= 1
                W[idx_C(C_r0+dr, C_c0+2+dc)][offset+2] += 1
        
        # M4
        for dr in range(2):
            for dc in range(2):
                U[idx_A(A_r0+2+dr, A_c0+2+dc)][offset+3] += 1
                V[idx_B(B_r0+2+dr, B_c0+dc)][offset+3] += 1
                V[idx_B(B_r0+dr, B_c0+dc)][offset+3] -= 1
                W[idx_C(C_r0+dr, C_c0+dc)][offset+3] += 1
                W[idx_C(C_r0+2+dr, C_c0+dc)][offset+3] += 1
                
        # M5
        for dr in range(2):
            for dc in range(2):
                U[idx_A(A_r0+dr, A_c0+dc)][offset+4] += 1
                U[idx_A(A_r0+dr, A_c0+2+dc)][offset+4] += 1
                V[idx_B(B_r0+2+dr, B_c0+2+dc)][offset+4] += 1
                W[idx_C(C_r0+dr, C_c0+dc)][offset+4] -= 1
                W[idx_C(C_r0+2+dr, C_c0+2+dc)][offset+4] += 1
                
        # M6
        for dr in range(2):
            for dc in range(2):
                U[idx_A(A_r0+2+dr, A_c0+dc)][offset+5] += 1
                U[idx_A(A_r0+dr, A_c0+dc)][offset+5] -= 1
                V[idx_B(B_r0+dr, B_c0+dc)][offset+5] += 1
                V[idx_B(B_r0+dr, B_c0+2+dc)][offset+5] += 1
                W[idx_C(C_r0+2+dr, C_c0+2+dc)][offset+5] += 1
                
        # M7
        for dr in range(2):
            for dc in range(2):
                U[idx_A(A_r0+dr, A_c0+2+dc)][offset+6] += 1
                U[idx_A(A_r0+2+dr, A_c0+2+dc)][offset+6] -= 1
                V[idx_B(B_r0+2+dr, B_c0+dc)][offset+6] += 1
                V[idx_B(B_r0+2+dr, B_c0+2+dc)][offset+6] += 1
                W[idx_C(C_r0+dr, C_c0+dc)][offset+6] += 1
                W[idx_C(C_r0+dr, C_c0+2+dc)][offset+6] += 1

    # C1: A1 B1 + A2 B3
    add_block_strassen(0, 0, 0, 0, 0, 0, 0)
    # C2: A1 B2 + A2 B4
    add_block_strassen(0, 0, 0, 2, 0, 2, 7)
    # C3: A3 B1 + A4 B3
    add_block_strassen(2, 0, 0, 0, 2, 0, 14)
    # C4: A3 B2 + A4 B4
    add_block_strassen(2, 0, 0, 2, 2, 2, 21)
    
    # Now restrict to 3x4x4
    # U: keep rows where i < 3 (rows 0,1,2 of A)
    # V: keep all rows
    # W: keep rows where i < 3 (rows 0,1,2 of C)
    
    U_344 = [U[i] for i in range(12)] # 0..11
    V_344 = V # 0..15
    W_344 = [W[i] for i in range(12) if i % 4 < 3] # Rows of W correspond to (k, i) of C. i is col of W?
    # W is (p*n, rank). p=4, n=4.
    # Index in W is k*n + i.
    # We want i < 3.
    # So we keep rows where (idx % 4) < 3.
    
    W_344 = [W[i] for i in range(16) if (i % 4) < 3]
    
    return {"U": U_344, "V": V_344, "W": W_344}

DIMENSIONS = [3, 4, 4]

def construct():
    # ... (same as above)
    pass
