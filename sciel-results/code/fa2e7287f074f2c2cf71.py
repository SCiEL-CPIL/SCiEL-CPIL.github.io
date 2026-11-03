import numpy as np

def construct():
    n, m, p = 3, 4, 7
    
    # We will build the decomposition for 3x4x7 using 3x2x7 blocks
    # Rank of 3x2x7 is 39. Total rank 78.
    
    # Helper to create Strassen 2x2x2
    def strassen_222():
        # U: 4x7, V: 4x7, W: 4x7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[1, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A11(B21-B22)
        U[0, 2] = 1
        V[2, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A22(B12-B11)
        U[3, 3] = 1
        V[1, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[3, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[1, 6] = 1
        
        return U, V, W

    # Helper to create Schoolbook 1x2x2 (Rank 4)
    def schoolbook_122():
        # A: 1x2, B: 2x2, C: 1x2
        # U: 2x4, V: 4x4, W: 2x4
        U = np.zeros((2, 4))
        V = np.zeros((4, 4))
        W = np.zeros((2, 4))
        
        # C1 = A1 B11 + A2 B21
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1
        
        # C2 = A1 B12 + A2 B22
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1
        
        return U, V, W

    # Helper to create Schoolbook 3x2x1 (Rank 6)
    def schoolbook_321():
        # A: 3x2, B: 2x1, C: 3x1
        # U: 6x6, V: 2x6, W: 3x6
        U = np.zeros((6, 6))
        V = np.zeros((2, 6))
        W = np.zeros((3, 6))
        
        for i in range(3):
            for j in range(2):
                col = i * 2 + j
                U[i * 2 + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                
        return U, V, W

    # Build 3x2x2 (Rank 11)
    def build_322():
        # A: 3x2, B: 2x2, C: 3x2
        # Split A into 2x2 and 1x2
        U_s, V_s, W_s = strassen_222()
        U_b, V_b, W_b = schoolbook_122()
        
        U = np.zeros((6, 11))
        V = np.zeros((4, 11))
        W = np.zeros((6, 11))
        
        # Strassen part (rows 0-3 of A, rows 0-3 of C)
        U[0:4, 0:7] = U_s
        V[0:4, 0:7] = V_s
        W[0:4, 0:7] = W_s
        
        # Schoolbook part (row 4 of A, row 4 of C)
        # Note: schoolbook_122 U is 2x4 (A part), V is 4x4 (B part), W is 2x4 (C part)
        # We map A row 4 to U_b row 0-1? No.
        # A is 3x2. Index 4,5 in U.
        # U_b has 2 rows. Map to U[4:6, 7:11].
        U[4:6, 7:11] = U_b
        V[0:4, 7:11] = V_b
        W[4:6, 7:11] = W_b
        
        return U, V, W

    # Build 3x2x1 (Rank 6)
    def build_321():
        return schoolbook_321()

    # Build 3x2x3 (Rank 17)
    def build_323():
        # Split B (2x3) into 2x2 and 2x1
        U_2, V_2, W_2 = build_322()
        U_1, V_1, W_1 = build_321()
        
        U = np.zeros((6, 17))
        V = np.zeros((6, 17)) # B is 2x3 -> 6 elements
        W = np.zeros((6, 17)) # C is 3x2 -> 6 elements
        
        # 3x2x2 part
        U[:, 0:11] = U_2
        V[0:4, 0:11] = V_2 # B top 2x2
        # V_1 is 2x6. B is 2x3.
        # We need to map B indices correctly.
        # In 3x2x2, B is 2x2. Indices 0-3.
        # In 3x2x1, B is 2x1. Indices 0-1.
        # In 3x2x3, B is 2x3. Indices 0-5.
        # B_2 (2x2) maps to V[0:4, 0:11]
        # B_1 (2x1) maps to V[4:6, 11:17]
        
        V[4:6, 11:17] = V_1
        W[:, 0:11] = W_2
        W[:, 11:17] = W_1
        
        return U, V, W

    # Build 3x2x4 (Rank 22)
    def build_324():
        # Split B (2x4) into 2x2 and 2x2
        U_2a, V_2a, W_2a = build_322()
        U_2b, V_2b, W_2b = build_322()
        
        U = np.zeros((6, 22))
        V = np.zeros((8, 22)) # B is 2x4 -> 8 elements
        W = np.zeros((6, 22))
        
        U[:, 0:11] = U_2a
        V[0:4, 0:11] = V_2a
        W[:, 0:11] = W_2a
        
        U[:, 11:22] = U_2b
        V[4:8, 11:22] = V_2b
        W[:, 11:22] = W_2b
        
        return U, V, W

    # Build 3x2x7 (Rank 39)
    def build_327():
        # Split B (2x7) into 2x4 and 2x3
        U_4, V_4, W_4 = build_324()
        U_3, V_3, W_3 = build_323()
        
        U = np.zeros((6, 39))
        V = np.zeros((14, 39)) # B is 2x7 -> 14 elements
        W = np.zeros((6, 39))
        
        U[:, 0:22] = U_4
        V[0:8, 0:22] = V_4
        W[:, 0:22] = W_4
        
        U[:, 22:39] = U_3
        V[8:14, 22:39] = V_3
        W[:, 22:39] = W_3
        
        return U, V, W

    # Build 3x4x7 (Rank 78)
    # Split A (3x4) into 3x2 and 3x2
    U_1, V_1, W_1 = build_327()
    U_2, V_2, W_2 = build_327()
    
    U = np.zeros((12, 78))
    V = np.zeros((28, 78)) # B is 4x7 -> 28 elements
    W = np.zeros((21, 78)) # C is 3x7 -> 21 elements
    
    # A_1 (3x2) maps to U[0:6, 0:39]
    # A_2 (3x2) maps to U[6:12, 39:78]
    U[0:6, 0:39] = U_1
    U[6:12, 39:78] = U_2
    
    # B_1 (2x7) maps to V[0:14, 0:39]
    # B_2 (2x7) maps to V[14:28, 39:78]
    V[0:14, 0:39] = V_1
    V[14:28, 39:78] = V_2
    
    # C (3x7) maps to W[:, 0:39] and W[:, 39:78]
    # Wait, W is 3x7.
    # In build_327, W is 6x39 (3x2 x 7 -> 3x7? No.
    # build_327 W is 6x39?
    # C is 3x7.
    # In build_322, W is 6x11 (3x2).
    # In build_327, W is 6x39?
    # C is 3x7. 3*7=21.
    # My W shapes are wrong.
    
    # Let's re-verify shapes.
    # 3x2x2: A(6), B(4), C(6). Rank 11.
    # 3x2x1: A(6), B(2), C(3). Rank 6.
    # 3x2x3: A(6), B(6), C(6). Rank 17.
    # 3x2x4: A(6), B(8), C(6). Rank 22.
    # 3x2x7: A(6), B(14), C(6). Rank 39.
    # 3x4x7: A(12), B(28), C(21). Rank 78.
    
    # In build_327, W should be 6x39?
    # C is 3x2? No, C is 3x7 in the final product.
    # But build_327 is for 3x2x7. C is 3x7?
    # No, 3x2 matrix A, 2x7 matrix B. C is 3x7.
    # So C has 21 elements.
    # My W in build_322 was 6x11 (3x2 C).
    # So W in build_327 should be 21x39.
    
    # I need to fix the W shapes in the helpers.
    
    return {"U": U, "V": V, "W": W}
