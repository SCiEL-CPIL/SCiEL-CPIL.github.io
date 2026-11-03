import numpy as np

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Schoolbook decomposition of rank n*m*p: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c].
    n, m, p = DIMENSIONS
    
    # We will construct the decomposition using a block Strassen-like approach.
    # 4x5x5 -> 2x(2x5x3 + 2x5x2) + 2x(2x5x3 + 2x5x2)
    # Actually, C_top_left (2x3) = A_top (2x5) * B_left (5x3)
    # C_top_right (2x2) = A_top (2x5) * B_right (5x2)
    # C_bot_left (2x3) = A_bot (2x5) * B_left (5x3)
    # C_bot_right (2x2) = A_bot (2x5) * B_right (5x2)
    
    # We need ranks for:
    # R(2, 5, 3) and R(2, 5, 2)
    # R(2, 5, 3) <= R(2, 2, 3) + R(2, 2, 3) + R(2, 1, 3)
    # R(2, 5, 2) <= R(2, 2, 2) + R(2, 2, 2) + R(2, 1, 2)
    
    # Base decompositions:
    # 1. Strassen 2x2x2 (Rank 7)
    # 2. Schoolbook 2x2x1 (Rank 2)
    # 3. Schoolbook 2x1x2 (Rank 2)
    # 4. Schoolbook 2x1x3 (Rank 2)
    # 5. 2x2x3 (Rank 9) - Constructed from 2x2x2 + 2x2x1
    
    # Helper to create schoolbook decomposition for n x m x p
    def schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W

    # Helper to create Strassen 2x2x2 decomposition
    def strassen_2x2x2():
        # Strassen's 7 multiplications
        # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
        # m1 = (a11+a22)(b11+b22)
        # m2 = (a21+a22)b11
        # m3 = a11(b12-b22)
        # m4 = a22(b21-b11)
        # m5 = (a11+a12)b22
        # m6 = (a21-a11)(b11+b12)
        # m7 = (a12-a22)(b21+b22)
        # C11 = m1 + m4 - m5 + m7
        # C12 = m3 + m5
        # C21 = m2 + m4
        # C22 = m1 - m2 + m3 + m6
        
        # U maps A indices (i*m+j) to multiplication inputs
        # V maps B indices (j*p+k) to multiplication inputs
        # W maps C indices (k*n+i) to multiplication outputs
        
        # Input A: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Input B: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        # Output C: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
        
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # m1: (a11+a22) * (b11+b22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # m2: (a21+a22) * b11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # m3: a11 * (b12-b22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # m4: a22 * (b21-b11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[0, 3] = 1; W[2, 3] = 1
        
        # m5: (a11+a12) * b22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[0, 4] = -1; W[1, 4] = 1
        
        # m6: (a21-a11) * (b11+b12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[3, 5] = 1
        
        # m7: (a12-a22) * (b21+b22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = 1; W[3, 6] = -1
        
        return U, V, W

    # Helper to create 2x2x3 decomposition (Rank 9)
    # A is 2x2, B is 2x3, C is 2x3
    # Split B into B_left (2x2) and B_right (2x1)
    # C_left (2x2) = A * B_left (Strassen, Rank 7)
    # C_right (2x1) = A * B_right (Schoolbook, Rank 2)
    def decomp_2x2x3():
        U_strassen, V_strassen, W_strassen = strassen_2x2x2()
        U_school, V_school, W_school = schoolbook(2, 2, 1)
        
        # U is 2x2 -> 4 elements. Same for both.
        # V is 2x3 -> 6 elements.
        # V_strassen uses first 2 cols of B (indices 0,1,2,3 in 2x2? No, 2x2 has 4 elements).
        # B is 2x3. Elements: (0,0), (0,1), (0,2), (1,0), (1,1), (1,2).
        # B_left is cols 0,1. Indices: 0,1,3,4.
        # B_right is col 2. Indices: 2,5.
        
        U_total = np.zeros((4, 9))
        V_total = np.zeros((6, 9))
        W_total = np.zeros((6, 9)) # C is 2x3 -> 6 elements
        
        # Part 1: Strassen for C_left (2x2)
        # U is same.
        U_total[:, 0:7] = U_strassen
        # V maps B_left indices (0,1,3,4) to V_strassen indices (0,1,2,3)
        V_total[0, 0:7] = V_strassen[0, 0:7] # B(0,0) -> V(0)
        V_total[1, 0:7] = V_strassen[1, 0:7] # B(0,1) -> V(1)
        V_total[3, 0:7] = V_strassen[2, 0:7] # B(1,0) -> V(2)
        V_total[4, 0:7] = V_strassen[3, 0:7] # B(1,1) -> V(3)
        # W maps C_left indices (0,1,2,3) to W_strassen indices (0,1,2,3)
        W_total[0, 0:7] = W_strassen[0, 0:7] # C(0,0)
        W_total[1, 0:7] = W_strassen[1, 0:7] # C(0,1)
        W_total[2, 0:7] = W_strassen[2, 0:7] # C(1,0)
        W_total[3, 0:7] = W_strassen[3, 0:7] # C(1,1)
        
        # Part 2: Schoolbook for C_right (2x1)
        # U is same.
        U_total[:, 7:9] = U_school
        # V maps B_right indices (2,5) to V_school indices (0,1)
        # V_school is for 2x2x1. Input B is 2x1? No, schoolbook(2,2,1) means A 2x2, B 2x1.
        # B elements: (0,0), (1,0). Indices in 2x1: 0, 1.
        # In 2x3, B_right is col 2. Elements (0,2), (1,2). Indices 2, 5.
        V_total[2, 7:9] = V_school[0, 0:2]
        V_total[5, 7:9] = V_school[1, 0:2]
        # W maps C_right indices (4,5) to W_school indices (0,1)
        # W_school is for 2x2x1. Output C is 2x1. Indices 0, 1.
        W_total[4, 7:9] = W_school[0, 0:2]
        W_total[5, 7:9] = W_school[1, 0:2]
        
        return U_total, V_total, W_total

    # Helper to create 2x5x3 decomposition (Rank 20)
    # A is 2x5, B is 5x3, C is 2x3
    # Split A into A1 (2x2), A2 (2x2), A3 (2x1)
    # Split B into B1 (2x3), B2 (2x3), B3 (1x3)
    # C = A1*B1 + A2*B2 + A3*B3
    # R(2,2,3) = 9, R(2,1,3) = 2
    def decomp_2x5x3():
        U_223, V_223, W_223 = decomp_2x2x3()
        U_213, V_213, W_213 = schoolbook(2, 1, 3)
        
        U_total = np.zeros((10, 20))
        V_total = np.zeros((15, 20))
        W_total = np.zeros((6, 20))
        
        # Part 1: A1 (rows 0,1) * B1 (cols 0,1)
        # U indices: 0,1,2,3 (A1 is 2x2)
        # V indices: 0,1,2,3,4,5 (B1 is 2x3)
        U_total[0:4, 0:9] = U_223
        V_total[0:6, 0:9] = V_223
        W_total[:, 0:9] = W_223
        
        # Part 2: A2 (rows 2,3) * B2 (cols 2,3)
        # U indices: 4,5,6,7
        # V indices: 6,7,8,9,10,11
        U_total[4:8, 9:18] = U_223
        V_total[6:12, 9:18] = V_223
        W_total[:, 9:18] = W_223
        
        # Part 3: A3 (row 4) * B3 (col 4)
        # U indices: 8,9
        # V indices: 12,13,14
        U_total[8:10, 18:20] = U_213
        V_total[12:15, 18:20] = V_213
        W_total[:, 18:20] = W_213
        
        return U_total, V_total, W_total

    # Helper to create 2x5x2 decomposition (Rank 16)
    # A is 2x5, B is 5x2, C is 2x2
    # Split A into A1 (2x2), A2 (2x2), A3 (2x1)
    # Split B into B1 (2x2), B2 (2x2), B3 (1x2)
    # C = A1*B1 + A2*B2 + A3*B3
    # R(2,2,2) = 7, R(2,1,2) = 2
    def decomp_2x5x2():
        U_222, V_222, W_222 = strassen_2x2x2()
        U_212, V_212, W_212 = schoolbook(2, 1, 2)
        
        U_total = np.zeros((10, 16))
        V_total = np.zeros((10, 16))
        W_total = np.zeros((4, 16))
        
        # Part 1: A1 * B1
        U_total[0:4, 0:7] = U_222
        V_total[0:4, 0:7] = V_222
        W_total[:, 0:7] = W_222
        
        # Part 2: A2 * B2
        U_total[4:8, 7:14] = U_222
        V_total[4:8, 7:14] = V_222
        W_total[:, 7:14] = W_222
        
        # Part 3: A3 * B3
        U_total[8:10, 14:16] = U_212
        V_total[8:10, 14:16] = V_212
        W_total[:, 14:16] = W_212
        
        return U_total, V_total, W_total

    # Main construction for 4x5x5
    # A is 4x5. Split into A_top (2x5), A_bot (2x5).
    # B is 5x5. Split into B_left (5x3), B_right (5x2).
    # C is 4x5. Split into C_top_left (2x3), C_top_right (2x2), C_bot_left (2x3), C_bot_right (2x2).
    
    U_253, V_253, W_253 = decomp_2x5x3()
    U_252, V_252, W_252 = decomp_2x5x2()
    
    rank_253 = 20
    rank_252 = 16
    
    U_total = np.zeros((20, 72))
    V_total = np.zeros((25, 72))
    W_total = np.zeros((20, 72))
    
    # C_top_left (2x3) = A_top (2x5) * B_left (5x3)
    # U indices: 0-9 (A_top)
    # V indices: 0-14 (B_left)
    U_total[0:10, 0:20] = U_253
    V_total[0:15, 0:20] = V_253
    W_total[0:6, 0:20] = W_253
    
    # C_top_right (2x2) = A_top (2x5) * B_right (5x2)
    # U indices: 0-9
    # V indices: 15-24 (B_right)
    U_total[0:10, 20:36] = U_252
    V_total[15:25, 20:36] = V_252
    W_total[6:10, 20:36] = W_252
    
    # C_bot_left (2x3) = A_bot (2x5) * B_left (5x3)
    # U indices: 10-19 (A_bot)
    # V indices: 0-14
    U_total[10:20, 36:56] = U_253
    V_total[0:15, 36:56] = V_253
    W_total[10:16, 36:56] = W_253
    
    # C_bot_right (2x2) = A_bot (2x5) * B_right (5x2)
    # U indices: 10-19
    # V indices: 15-24
    U_total[10:20, 56:72] = U_252
    V_total[15:25, 56:72] = V_252
    W_total[16:20, 56:72] = W_252
    
    return {"U": U_total.tolist(), "V": V_total.tolist(), "W": W_total.tolist()}
# EVOLVE-BLOCK-END
