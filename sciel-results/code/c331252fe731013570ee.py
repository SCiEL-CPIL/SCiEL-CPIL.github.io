import time
import numpy as np

def construct():
    n, m, p = 3, 3, 4
    start_time = time.time()
    deadline = start_time + 140
    
    # Rank 34 solution
    # Split B into two 3x2 blocks: B1, B2
    # Compute C1 = A * B1 (3x2), C2 = A * B2 (3x2)
    # Use rank 17 decomposition for 3x3 x 3x2
    
    def strassen_2x2x2():
        # Returns U, V, W for 2x2 x 2x2 multiplication
        # U: (4, 7), V: (4, 7), W: (4, 7)
        # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
        U = np.zeros((4, 7), dtype=int)
        V = np.zeros((4, 7), dtype=int)
        W = np.zeros((4, 7), dtype=int)
        
        # M1 = (A11+A22)(B11+B22)
        # M2 = (A21+A22)B11
        # M3 = A11(B21-B22)
        # M4 = A22(B12-B11)
        # M5 = (A11+A12)B22
        # M6 = (A21-A11)(B11+B12)
        # M7 = (A12-A22)(B21+B22)
        
        # W11 = M1 + M4 - M5 + M7
        # W12 = M3 + M5
        # W21 = M2 + M4
        # W22 = M1 - M2 + M3 + M6
        
        # U (A factors)
        # M1: A11, A22
        U[0, 0] = 1 # A11
        U[3, 0] = 1 # A22
        # M2: A21, A22
        U[2, 1] = 1 # A21
        U[3, 1] = 1 # A22
        # M3: A11
        U[0, 2] = 1 # A11
        # M4: A22
        U[3, 3] = 1 # A22
        # M5: A11, A12
        U[0, 4] = 1 # A11
        U[1, 4] = 1 # A12
        # M6: A21, A11
        U[2, 5] = 1 # A21
        U[0, 5] = -1 # A11
        # M7: A12, A22
        U[1, 6] = 1 # A12
        U[3, 6] = -1 # A22
        
        # V (B factors)
        # M1: B11, B22
        V[0, 0] = 1 # B11
        V[3, 0] = 1 # B22
        # M2: B11
        V[0, 1] = 1 # B11
        # M3: B21, B22
        V[2, 2] = 1 # B21
        V[3, 2] = -1 # B22
        # M4: B12, B11
        V[1, 3] = 1 # B12
        V[0, 3] = -1 # B11
        # M5: B22
        V[3, 4] = 1 # B22
        # M6: B11, B12
        V[0, 5] = 1 # B11
        V[1, 5] = 1 # B12
        # M7: B21, B22
        V[2, 6] = 1 # B21
        V[3, 6] = 1 # B22
        
        # W (C factors)
        # W11 (idx 0): M1, M4, M5, M7
        W[0, 0] = 1
        W[0, 3] = 1
        W[0, 4] = -1
        W[0, 6] = 1
        # W12 (idx 1): M3, M5
        W[1, 2] = 1
        W[1, 4] = 1
        # W21 (idx 2): M2, M4
        W[2, 1] = 1
        W[2, 3] = 1
        # W22 (idx 3): M1, M2, M3, M6
        W[3, 0] = 1
        W[3, 1] = -1
        W[3, 2] = 1
        W[3, 5] = 1
        
        return U, V, W

    def schoolbook_2x1x2():
        # U: (2, 4), V: (2, 4), W: (4, 4)
        # A: 2x1, B: 1x2
        # C: 2x2
        # C11 = A1 B1
        # C12 = A1 B2
        # C21 = A2 B1
        # C22 = A2 B2
        U = np.zeros((2, 4), dtype=int)
        V = np.zeros((2, 4), dtype=int)
        W = np.zeros((4, 4), dtype=int)
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # C11
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1 # C12
        U[1, 2] = 1; V[0, 2] = 1; W[2, 2] = 1 # C21
        U[1, 3] = 1; V[1, 3] = 1; W[3, 3] = 1 # C22
        
        return U, V, W

    def schoolbook_1x2x2():
        # U: (2, 4), V: (4, 4), W: (2, 4)
        # A: 1x2, B: 2x2
        # C: 1x2
        # C1 = A1 B11 + A2 B21
        # C2 = A1 B12 + A2 B22
        U = np.zeros((2, 4), dtype=int)
        V = np.zeros((4, 4), dtype=int)
        W = np.zeros((2, 4), dtype=int)
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1 # C1 from A1 B11
        U[1, 1] = 1; V[2, 1] = 1; W[0, 1] = 1 # C1 from A2 B21
        U[0, 2] = 1; V[1, 2] = 1; W[1, 2] = 1 # C2 from A1 B12
        U[1, 3] = 1; V[3, 3] = 1; W[1, 3] = 1 # C2 from A2 B22
        
        return U, V, W

    def schoolbook_1x1x2():
        # U: (1, 2), V: (2, 2), W: (2, 2)
        # A: 1x1, B: 1x2
        # C: 1x2
        U = np.zeros((1, 2), dtype=int)
        V = np.zeros((2, 2), dtype=int)
        W = np.zeros((2, 2), dtype=int)
        
        U[0, 0] = 1; V[0, 0] = 1; W[0, 0] = 1
        U[0, 1] = 1; V[1, 1] = 1; W[1, 1] = 1
        
        return U, V, W

    # Assemble 3x3 x 3x2 decomposition (Rank 17)
    # A: 3x3, B: 3x2
    # Blocks:
    # A22 (2x2), A21 (2x1), A12 (1x2), A11 (1x1)
    # B22 (2x2), B12 (1x2)
    
    U_332 = np.zeros((9, 17), dtype=int)
    V_332 = np.zeros((6, 17), dtype=int)
    W_332 = np.zeros((6, 17), dtype=int)
    
    # 1. A22 * B22 (Rank 7)
    U_s, V_s, W_s = strassen_2x2x2()
    # Map indices
    # A22 is A[0:2, 0:2] -> rows 0,1,2,3 of U
    # B22 is B[0:2, 0:2] -> rows 0,1,4,5 of V (B is 3x2, B[j,k] -> j*2+k)
    # C22 is C[0:2, 0:2] -> rows 0,1,4,5 of W (C is 3x2, C[i,k] -> k*3+i? No, W is p*n = 2*3=6. W[k*n+i]. C[i,k]. k is 0,1. i is 0,1,2.
    # Wait, for 3x3 x 3x2, C is 3x2.
    # W index is k*n + i. n=3.
    # C[0,0] -> W[0*3+0]=0
    # C[0,1] -> W[1*3+0]=3
    # C[1,0] -> W[0*3+1]=1
    # C[1,1] -> W[1*3+1]=4
    # C[2,0] -> W[0*3+2]=2
    # C[2,1] -> W[1*3+2]=5
    
    # Strassen W is for 2x2 C.
    # W_s[0] -> C11 -> C[0,0] -> W_332[0]
    # W_s[1] -> C12 -> C[0,1] -> W_332[3]
    # W_s[2] -> C21 -> C[1,0] -> W_332[1]
    # W_s[3] -> C22 -> C[1,1] -> W_332[4]
    
    for r in range(7):
        # U
        for i in range(4):
            if U_s[i, r] != 0:
                row_u = i // 2 * 3 + i % 2 # A[i//2, i%2]
                U_332[row_u, r] = U_s[i, r]
        # V
        for j in range(4):
            if V_s[j, r] != 0:
                row_v = j // 2 * 2 + j % 2 # B[j//2, j%2]
                V_332[row_v, r] = V_s[j, r]
        # W
        for k in range(4):
            if W_s[k, r] != 0:
                # k is index in W_s (0:11, 1:12, 2:21, 3:22)
                # Map to C[i, k_c]
                if k == 0:
                    row_w = 0 # C[0,0]
                elif k == 1:
                    row_w = 3 # C[0,1]
                elif k == 2:
                    row_w = 1 # C[1,0]
                elif k == 3:
                    row_w = 4 # C[1,1]
                W_332[row_w, r] = W_s[k, r]
                
    # 2. A21 * B12 (Rank 4)
    U_s, V_s, W_s = schoolbook_2x1x2()
    # A21 is A[0:2, 2] -> rows 2, 5 of U
    # B12 is B[2, 0:2] -> rows 4, 5 of V
    # C21 is C[0:2, 0:2]? No.
    # A21 (2x1) * B12 (1x2) = C (2x2)
    # This contributes to C[0:2, 0:2].
    # C[0,0] += A[0,2]B[2,0]
    # C[0,1] += A[0,2]B[2,1]
    # C[1,0] += A[1,2]B[2,0]
    # C[1,1] += A[1,2]B[2,1]
    for r in range(4):
        for i in range(2):
            if U_s[i, r] != 0:
                row_u = i * 3 + 2 # A[i, 2]
                U_332[row_u, 7+r] = U_s[i, r]
        for j in range(2):
            if V_s[j, r] != 0:
                row_v = 2 * 2 + j # B[2, j]
                V_332[row_v, 7+r] = V_s[j, r]
        for k in range(4):
            if W_s[k, r] != 0:
                if k == 0: row_w = 0
                elif k == 1: row_w = 3
                elif k == 2: row_w = 1
                elif k == 3: row_w = 4
                W_332[row_w, 7+r] = W_s[k, r]

    # 3. A12 * B22 (Rank 4)
    U_s, V_s, W_s = schoolbook_1x2x2()
    # A12 is A[2, 0:2] -> row 6,7 of U
    # B22 is B[0:2, 0:2] -> rows 0,1,4,5 of V
    # C12 is C[2, 0:2]
    for r in range(4):
        for i in range(2):
            if U_s[i, r] != 0:
                row_u = 2 * 3 + i # A[2, i]
                U_332[row_u, 11+r] = U_s[i, r]
        for j in range(4):
            if V_s[j, r] != 0:
                row_v = j // 2 * 2 + j % 2 # B[j//2, j%2]
                V_332[row_v, 11+r] = V_s[j, r]
        for k in range(2):
            if W_s[k, r] != 0:
                # k=0 -> C[2,0] -> W[0*3+2]=2
                # k=1 -> C[2,1] -> W[1*3+2]=5
                row_w = k * 3 + 2
                W_332[row_w, 11+r] = W_s[k, r]

    # 4. A11 * B12 (Rank 2)
    U_s, V_s, W_s = schoolbook_1x1x2()
    # A11 is A[2, 2] -> row 8 of U
    # B12 is B[2, 0:2] -> rows 4,5 of V
    # C11 is C[2, 0:2]
    for r in range(2):
        for i in range(1):
            if U_s[i, r] != 0:
                row_u = 8
                U_332[row_u, 15+r] = U_s[i, r]
        for j in range(2):
            if V_s[j, r] != 0:
                row_v = 2 * 2 + j
                V_332[row_v, 15+r] = V_s[j, r]
        for k in range(2):
            if W_s[k, r] != 0:
                row_w = k * 3 + 2
                W_332[row_w, 15+r] = W_s[k, r]

    # Now assemble 3x3 x 3x4 (Rank 34)
    # B is 3x4. Split into B1 (3x2) and B2 (3x2).
    # C is 3x4. Split into C1 (3x2) and C2 (3x2).
    # C1 = A * B1
    # C2 = A * B2
    
    U_final = np.zeros((9, 34), dtype=int)
    V_final = np.zeros((12, 34), dtype=int)
    W_final = np.zeros((12, 34), dtype=int)
    
    # Copy 3x3x2 decomposition for first block
    for r in range(17):
        U_final[:, r] = U_332[:, r]
        # V: B1 is B[:, 0:2]. Rows 0-5 of V_final.
        V_final[0:6, r] = V_332[:, r]
        # W: C1 is C[:, 0:2]. Rows 0-5 of W_final (k=0,1).
        W_final[0:6, r] = W_332[:, r]
        
    # Copy 3x3x2 decomposition for second block
    for r in range(17):
        U_final[:, 17+r] = U_332[:, r]
        # V: B2 is B[:, 2:4]. Rows 6-11 of V_final.
        V_final[6:12, 17+r] = V_332[:, r]
        # W: C2 is C[:, 2:4]. Rows 6-11 of W_final (k=2,3).
        # W index k*n + i.
        # For C2, k is 2,3.
        # W_332 has k=0,1.
        # We need to map W_332[k, i] to W_final[(k+2)*3 + i, 17+r].
        for i in range(3):
            for k in range(2):
                W_final[(k+2)*3 + i, 17+r] = W_332[k*3 + i, r]

    return {"U": U_final, "V": V_final, "W": W_final}
