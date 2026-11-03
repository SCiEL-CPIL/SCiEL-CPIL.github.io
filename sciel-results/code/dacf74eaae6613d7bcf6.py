import numpy as np

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    
    # Helper to create zero matrices
    def make_zero(rows, cols):
        return [[0] * cols for _ in range(rows)]
    
    # Strassen 2x2x2
    # Returns U (4x7), V (4x7), W (4x7) for 2x2 matrix mult
    def strassen_222():
        # M1 = (A00+A11)(B00+B11)
        # M2 = (A10+A11)B00
        # M3 = A00(B01-B11)
        # M4 = A11(B10-B00)
        # M5 = (A00+A01)B11
        # M6 = (A10-A00)(B00+B01)
        # M7 = (A01-A11)(B10+B11)
        
        # Local indices:
        # A: 00, 01, 10, 11 -> 0, 1, 2, 3
        # B: 00, 01, 10, 11 -> 0, 1, 2, 3
        # C: 00, 01, 10, 11 -> 0, 1, 2, 3
        
        U = [[0]*7 for _ in range(4)]
        V = [[0]*7 for _ in range(4)]
        W = [[0]*7 for _ in range(4)]
        
        # M1
        # A00, A11
        U[0][0] = 1; U[3][0] = 1
        # B00, B11
        V[0][0] = 1; V[3][0] = 1
        # C00 += M1, C01 -= M3, C10 += M2, C11 += M5
        # Actually, we just define the W coefficients for each C index.
        # C00 = M1 - M3 + M5
        # C01 = M4 + M5
        # C10 = M2 - M4 + M6
        # C11 = M1 - M2 + M7
        
        # Let's fill W based on the formulas:
        # C00 (idx 0): M1(1), M3(-1), M5(1)
        W[0][0] = 1
        W[0][2] = -1
        W[0][4] = 1
        
        # C01 (idx 1): M4(1), M5(1)
        W[1][3] = 1
        W[1][4] = 1
        
        # C10 (idx 2): M2(1), M4(-1), M6(1)
        W[2][1] = 1
        W[2][3] = -1
        W[2][5] = 1
        
        # C11 (idx 3): M1(1), M2(-1), M7(1)
        W[3][0] = 1
        W[3][1] = -1
        W[3][6] = 1
        
        # Fill U and V for all 7
        # M1: A00+A11, B00+B11
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        
        # M2: A10+A11, B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        
        # M3: A00, B01-B11
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        
        # M4: A11, B10-B00
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        
        # M5: A00+A01, B11
        U[0][4] = 1; U[1][4] = 1
        V[3][4] = 1
        
        # M6: A10-A00, B00+B01
        U[2][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[1][5] = 1
        
        # M7: A01-A11, B10+B11
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        
        return U, V, W

    def schoolbook_221():
        # 2x2 A, 2x1 B -> 2x1 C
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        # Rank 4
        U = [[0]*4 for _ in range(4)]
        V = [[0]*4 for _ in range(2)] # B is 2x1 -> 2 rows
        W = [[0]*4 for _ in range(2)] # C is 2x1 -> 2 rows
        
        # C0
        # A00 B00
        U[0][0] = 1; V[0][0] = 1; W[0][0] = 1
        # A01 B10
        U[1][1] = 1; V[1][1] = 1; W[0][1] = 1
        
        # C1
        # A10 B00
        U[2][2] = 1; V[0][2] = 1; W[1][2] = 1
        # A11 B10
        U[3][3] = 1; V[1][3] = 1; W[1][3] = 1
        
        return U, V, W

    # ... other schoolbook helpers ...
    # This is getting long. I will implement a generic schoolbook and use it for small blocks,
    # and specific Strassen for 2x2x2.
    
    # Generic Schoolbook for n x m x p
    def schoolbook_nm_p(n, m, p):
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W

    # Assemble 3x3x3 using blocks
    # We will build U, V, W for the 3x3x3 part.
    # Then 3x3x2 part.
    
    # Part 1: 3x3x3
    # Blocks:
    # A11(2x2), A12(2x1), A21(1x2), A22(1x1)
    # B11(2x2), B12(2x1), B21(1x2), B22(1x1)
    
    # We need to map local block indices to global indices.
    # Global A (3x3):
    # 0 0 1
    # 0 1 2
    # 1 0 3
    # 1 1 4
    # 2 0 5
    # 2 1 6
    # 2 2 7
    
    # This is too complex to hard-code manually in the short time.
    # I will stick to the Schoolbook 45 if I can't easily code the block assembly.
    # But 45 is the seed. I must improve.
    
    # Let's try a simpler improvement:
    # Use Strassen for the 2x2 sub-blocks in a 3x3x3?
    # No, the block assembly is the key.
    
    # I will implement the block assembly for 3x3x3 and 3x3x2.
    
    pass # Placeholder for actual implementation in the edit
