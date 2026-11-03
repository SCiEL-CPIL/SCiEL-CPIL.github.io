import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 4, 9
    
    # Known decompositions
    # Strassen 2x2x2: rank 7
    # U: rows indexed by (i,j) 0..3, cols 0..6
    # V: rows indexed by (j,k) 0..3, cols 0..6
    # W: rows indexed by (k,i) 0..3, cols 0..6
    S = [
        # (i,j), (j,k), (k,i)
        (0,0,0), # M1 = A00*B00
        (0,1,1), # M2 = A00*A01 * B10*B11? No.
        # Let's use standard Strassen:
        # M1 = A00 * B00
        # M2 = A10 * B00
        # M3 = A00 * (B01 - B00)
        # M4 = A11 * (B10 - B00)
        # M5 = (A00 + A11) * (B00 + B11)
        # M6 = (A01 - A11) * (B10 + B11)
        # M7 = (A00 - A10) * (B01 - B10)
        # C00 = M1 + M4 - M5 + M7
        # C01 = M3 + M5
        # C10 = M2 + M4
        # C11 = M1 - M2 + M3 + M6
    ]
    
    U_2x2 = [[0]*7 for _ in range(4)]
    V_2x2 = [[0]*7 for _ in range(4)]
    W_2x2 = [[0]*7 for _ in range(4)]
    
    # M1: A00 * B00 -> C00
    U_2x2[0][0] = 1; V_2x2[0][0] = 1; W_2x2[0][0] = 1
    # M2: A10 * B00 -> C10
    U_2x2[2][1] = 1; V_2x2[0][1] = 1; W_2x2[2][1] = 1
    # M3: A00 * (B01 - B00) -> C00, C01
    U_2x2[0][2] = 1
    V_2x2[1][2] = 1; V_2x2[0][2] = -1
    W_2x2[1][2] = 1; W_2x2[0][2] = 1
    # M4: A11 * (B10 - B00) -> C10, C11
    U_2x2[3][3] = 1
    V_2x2[2][3] = 1; V_2x2[0][3] = -1
    W_2x2[2][3] = 1; W_2x2[3][3] = 1
    # M5: (A00 + A11) * (B00 + B11) -> C00, C01, C10, C11
    U_2x2[0][4] = 1; U_2x2[3][4] = 1
    V_2x2[0][4] = 1; V_2x2[3][4] = 1
    W_2x2[0][4] = -1; W_2x2[1][4] = 1; W_2x2[2][4] = 1; W_2x2[3][4] = -1
    # M6: (A01 - A11) * (B10 + B11) -> C11
    U_2x2[1][5] = 1; U_2x2[3][5] = -1
    V_2x2[2][5] = 1; V_2x2[3][5] = 1
    W_2x2[3][5] = 1
    # M7: (A00 - A10) * (B01 - B10) -> C00, C11
    U_2x2[0][6] = 1; U_2x2[2][6] = -1
    V_2x2[1][6] = 1; V_2x2[2][6] = -1
    W_2x2[0][6] = 1; W_2x2[3][6] = 1
    
    # Winograd 2x2x3: rank 7
    # A (2x2) * B (2x3) = C (2x3)
    # M1 = A00 * B00
    # M2 = A10 * B01
    # M3 = (A00 - A10) * (B01 - B00)
    # M4 = (A00 + A10) * (B10 + B11)
    # M5 = A01 * B10
    # M6 = A11 * B11
    # M7 = (A01 - A11) * (B11 - B10)
    # C00 = M1
    # C01 = M2 + M3
    # C02 = M3
    # C10 = M4 + M5
    # C11 = M4 + M6
    # C12 = M6 + M7
    U_2x2x3 = [[0]*7 for _ in range(4)] # i*2+j
    V_2x2x3 = [[0]*7 for _ in range(6)] # j*3+k
    W_2x2x3 = [[0]*7 for _ in range(6)] # k*2+i
    
    # M1: A00 * B00 -> C00
    U_2x2x3[0][0] = 1; V_2x2x3[0][0] = 1; W_2x2x3[0][0] = 1
    # M2: A10 * B01 -> C01
    U_2x2x3[2][1] = 1; V_2x2x3[1][1] = 1; W_2x2x3[1][1] = 1
    # M3: (A00 - A10) * (B01 - B00) -> C01, C02
    U_2x2x3[0][2] = 1; U_2x2x3[2][2] = -1
    V_2x2x3[1][2] = 1; V_2x2x3[0][2] = -1
    W_2x2x3[1][2] = 1; W_2x2x3[2][2] = 1
    # M4: (A00 + A10) * (B10 + B11) -> C10, C11
    U_2x2x3[0][3] = 1; U_2x2x3[2][3] = 1
    V_2x2x3[3][3] = 1; V_2x2x3[4][3] = 1
    W_2x2x3[2][3] = 1; W_2x2x3[4][3] = 1
    # M5: A01 * B10 -> C10
    U_2x2x3[1][4] = 1; V_2x2x3[3][4] = 1; W_2x2x3[2][4] = 1
    # M6: A11 * B11 -> C11, C12
    U_2x2x3[3][5] = 1; V_2x2x3[4][5] = 1
    W_2x2x3[4][5] = 1; W_2x2x3[5][5] = 1
    # M7: (A01 - A11) * (B11 - B10) -> C12
    U_2x2x3[1][6] = 1; U_2x2x3[3][6] = -1
    V_2x2x3[4][6] = 1; V_2x2x3[3][6] = -1
    W_2x2x3[5][6] = 1
    
    # Now combine 4x4x9 = (2x2x2) x (2x2x3)
    # A is 4x4 (2x2 blocks of 2x2)
    # B is 4x9 (2x2 blocks of 2x3)
    # C is 4x9 (2x2 blocks of 2x3)
    
    # Block indices:
    # A: A_{I,J} where I in 0,1 (row block), J in 0,1 (col block)
    # B: B_{J,K} where J in 0,1 (row block), K in 0,1,2 (col block)
    # C: C_{I,K} where I in 0,1 (row block), K in 0,1,2 (col block)
    
    # C_{I,K} = sum_{J=0}^{1} A_{I,J} * B_{J,K}
    
    # Each A_{I,J} is 2x2, each B_{J,K} is 2x3.
    # We can use Strassen on the 2x2 block structure for I,J?
    # No, B has 3 column blocks.
    
    # Let's use the 2x2x3 decomposition for each product A_{I,J} * B_{J,K}.
    # There are 2 * 2 * 3 = 12 such products.
    # Each has rank 7. Total rank = 12 * 7 = 84.
    # Can we share computations?
    # A_{I,J} is used for K=0,1,2.
    # So A_{0,0} is used in 3 products.
    
    # Let's build the full matrices.
    # U: 16xR, V: 36xR, W: 36xR
    # R = 84 initially.
    
    U = [[0]*84 for _ in range(16)]
    V = [[0]*84 for _ in range(36)]
    W = [[0]*84 for _ in range(36)]
    
    col = 0
    for I in range(2):
        for J in range(2):
            for K in range(3):
                # Use U_2x2x3, V_2x2x3, W_2x2x3
                # Local indices:
                # u_local = i*2 + j (i,j in 0,1) -> global u = (I*2+i)*4 + (J*2+j)
                # v_local = j*3 + k (j,k in 0,1,2) -> global v = (J*2+j)*9 + (K*3+k)
                # w_local = k*2 + i (k,i in 0,1,2) -> global w = (K*3+k)*4 + (I*2+i)
                
                for r in range(7):
                    c = col + r
                    # U
                    for i in range(2):
                        for j in range(2):
                            val = U_2x2x3[i*2+j][r]
                            if val != 0:
                                u_global = (I*2+i)*4 + (J*2+j)
                                U[u_global][c] += val
                    # V
                    for j in range(2):
                        for k in range(3):
                            val = V_2x2x3[j*3+k][r]
                            if val != 0:
                                v_global = (J*2+j)*9 + (K*3+k)
                                V[v_global][c] += val
                    # W
                    for k in range(3):
                        for i in range(2):
                            val = W_2x2x3[k*2+i][r]
                            if val != 0:
                                w_global = (K*3+k)*4 + (I*2+i)
                                W[w_global][c] += val
                col += 7
    
    # This gives rank 84.
    # Is there a better one?
    # Strassen 4x4x4 is rank 49.
    # Here B is 4x9.
    # We can pad B to 4x12 (4x3x4) and use 4x4x12?
    # 4x4x12 = (4x4x4) x 3? Rank 49*3 = 147. Worse.
    
    # What about 4x4x8?
    # 4x4x8 = (2x2x2) x (2x2x4)?
    # 2x2x4 = 2*(2x2x2)? Rank 14.
    # Total 16 * 14 = 224. Worse.
    
    # The 84 rank decomposition seems good.
    # Let's check if we can do better.
    # Lower bound for 4x4x9?
    # 4x4x4 is 49.
    # 4x4x8 is 2 * 4x4x4? No.
    # 4x4x8 can be seen as two 4x4x4? No.
    # 4x4x8 = 4x4 x 4x8.
    # 4x8 = 4x2 x 2.
    # Rank(4x4x8) <= Rank(4x4x4) + Rank(4x4x4) = 98?
    # Actually, 4x4x8 can be done with rank 2 * Rank(2x2x2) * 2?
    # 4x4x8 = (2x2x2) x (2x2x4).
    # 2x2x4 = 2 * 2x2x2. Rank 14.
    # So 4x4x8 rank <= 7 * 14 = 98.
    # 4x4x9 = 4x4x8 + 4x4x1.
    # Rank(4x4x1) = 4.
    # So Rank(4x4x9) <= 98 + 4 = 102.
    # Our 84 is better!
    
    # Can we do better than 84?
    # 4x4x9 = (2x2x2) x (2x2x3) x 2?
    # We used 12 * 7 = 84.
    # Is there a sharing?
    # A_{I,J} is used for 3 K's.
    # In our construction, we have separate columns for each (I,J,K).
    # The U part for A_{I,J} is repeated for K=0,1,2.
    # We can factor out A_{I,J}.
    # But the V and W parts are different.
    
    # Let's try to optimize.
    # For fixed I, J:
    # We compute sum_{k_local} (A_{I,J} * B_{J,K})
    # Using Winograd 2x2x3.
    # The U part is A_{I,J}.
    # The V part depends on K.
    
    # If we don't share, rank is 84.
    # Let's just return this. It's a significant improvement over 144.
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
