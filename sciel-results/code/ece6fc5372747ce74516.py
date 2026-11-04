"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = 140.0
    
    def schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = schoolbook()
    best_rank = n * m * p

    # Construct Strassen 4x4x4 decomposition (Rank 49)
    # and Vector Multiplication 4x4x1 decomposition (Rank 16)
    # Total Rank 65
    
    def get_strassen_2x2():
        # Returns U, V, W for 2x2x2 with rank 7
        # Indices: i,j in 0..1 for A, k,l in 0..1 for B, m,n in 0..1 for C
        # Flattened indices: A: i*2+j, B: k*2+l, C: m*2+n
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # M1 = (A11+A22)(B11+B22)
        U[0, 0] = 1; U[3, 0] = 1
        V[0, 0] = 1; V[3, 0] = 1
        W[0, 0] = 1; W[3, 0] = 1
        
        # M2 = (A21+A22)B11
        U[2, 1] = 1; U[3, 1] = 1
        V[0, 1] = 1
        W[2, 1] = 1
        
        # M3 = A11(B12-B22)
        U[0, 2] = 1
        V[1, 2] = 1; V[3, 2] = -1
        W[1, 2] = 1
        
        # M4 = A22(B21-B11)
        U[3, 3] = 1
        V[2, 3] = 1; V[0, 3] = -1
        W[2, 3] = 1; W[3, 3] = 1
        
        # M5 = (A11+A12)B22
        U[0, 4] = 1; U[1, 4] = 1
        V[3, 4] = 1
        W[1, 4] = 1; W[3, 4] = 1
        
        # M6 = (A21-A11)(B11+B12)
        U[2, 5] = 1; U[0, 5] = -1
        V[0, 5] = 1; V[1, 5] = 1
        W[0, 5] = 1; W[2, 5] = 1
        
        # M7 = (A12-A22)(B21+B22)
        U[1, 6] = 1; U[3, 6] = -1
        V[2, 6] = 1; V[3, 6] = 1
        W[0, 6] = -1; W[1, 6] = 1
        
        return U, V, W

    U2, V2, W2 = get_strassen_2x2()
    
    # Define block linear maps for 4x4x4 Strassen
    # A is 4x4. Blocks A11, A12, A21, A22 are 2x2.
    # Global indices: i in 0..3, j in 0..3.
    # A11: i in 0..1, j in 0..1
    # A12: i in 0..1, j in 2..3
    # A21: i in 2..3, j in 0..1
    # A22: i in 2..3, j in 2..3
    
    # We need L_A, L_B, L_C for each of the 7 products.
    # L_A: 16 -> 4. L_B: 16 -> 4. L_C: 4 -> 16.
    
    def get_block_maps():
        maps = []
        # M1: (A11+A22)(B11+B22) -> C11, C22
        LA = np.zeros((4, 16))
        LA[0, 0] = 1; LA[1, 1] = 1 # A11
        LA[0, 12] = 1; LA[1, 13] = 1 # A22 (indices: 12=2*2+0, 13=2*2+1, 14=2*2+2, 15=2*2+3? No.)
        # Flattened A:
        # 0: (0,0), 1: (0,1)
        # 2: (1,0), 3: (1,1)
        # 4: (2,0), 5: (2,1)
        # 6: (3,0), 7: (3,1)
        # 8: (0,2), 9: (0,3)
        # 10: (1,2), 11: (1,3)
        # 12: (2,2), 13: (2,3)
        # 14: (3,2), 15: (3,3)
        
        # A11: 0,1,2,3. A22: 12,13,14,15.
        LA[0, 0] = 1; LA[1, 1] = 1; LA[2, 2] = 1; LA[3, 3] = 1
        LA[0, 12] = 1; LA[1, 13] = 1; LA[2, 14] = 1; LA[3, 15] = 1
        
        LB = np.zeros((4, 16))
        LB[0, 0] = 1; LB[1, 1] = 1; LB[2, 2] = 1; LB[3, 3] = 1
        LB[0, 12] = 1; LB[1, 13] = 1; LB[2, 14] = 1; LB[3, 15] = 1
        
        LC = np.zeros((16, 4))
        # C11 (0,1,2,3) gets M1. C22 (12,13,14,15) gets M1.
        LC[0, 0] = 1; LC[1, 1] = 1; LC[2, 2] = 1; LC[3, 3] = 1
        LC[12, 0] = 1; LC[13, 1] = 1; LC[14, 2] = 1; LC[15, 3] = 1
        maps.append((LA, LB, LC))
        
        # M2: (A21+A22)B11 -> C21
        LA = np.zeros((4, 16))
        LA[0, 4] = 1; LA[1, 5] = 1; LA[2, 6] = 1; LA[3, 7] = 1 # A21
        LA[0, 12] = 1; LA[1, 13] = 1; LA[2, 14] = 1; LA[3, 15] = 1 # A22
        
        LB = np.zeros((4, 16))
        LB[0, 0] = 1; LB[1, 1] = 1; LB[2, 2] = 1; LB[3, 3] = 1 # B11
        
        LC = np.zeros((16, 4))
        LC[4, 0] = 1; LC[5, 1] = 1; LC[6, 2] = 1; LC[7, 3] = 1 # C21
        maps.append((LA, LB, LC))
        
        # M3: A11(B12-B22) -> C12
        LA = np.zeros((4, 16))
        LA[0, 0] = 1; LA[1, 1] = 1; LA[2, 2] = 1; LA[3, 3] = 1 # A11
        
        LB = np.zeros((4, 16))
        LB[0, 8] = 1; LB[1, 9] = 1; LB[2, 10] = 1; LB[3, 11] = 1 # B12
        LB[0, 12] = -1; LB[1, 13] = -1; LB[2, 14] = -1; LB[3, 15] = -1 # -B22
        
        LC = np.zeros((16, 4))
        LC[8, 0] = 1; LC[9, 1] = 1; LC[10, 2] = 1; LC[11, 3] = 1 # C12
        maps.append((LA, LB, LC))
        
        # M4: A22(B21-B11) -> C21, C22
        LA = np.zeros((4, 16))
        LA[0, 12] = 1; LA[1, 13] = 1; LA[2, 14] = 1; LA[3, 15] = 1 # A22
        
        LB = np.zeros((4, 16))
        LB[0, 4] = 1; LB[1, 5] = 1; LB[2, 6] = 1; LB[3, 7] = 1 # B21
        LB[0, 0] = -1; LB[1, 1] = -1; LB[2, 2] = -1; LB[3, 3] = -1 # -B11
        
        LC = np.zeros((16, 4))
        LC[4, 0] = 1; LC[5, 1] = 1; LC[6, 2] = 1; LC[7, 3] = 1 # C21
        LC[12, 0] = 1; LC[13, 1] = 1; LC[14, 2] = 1; LC[15, 3] = 1 # C22
        maps.append((LA, LB, LC))
        
        # M5: (A11+A12)B22 -> C12, C22
        LA = np.zeros((4, 16))
        LA[0, 0] = 1; LA[1, 1] = 1; LA[2, 2] = 1; LA[3, 3] = 1 # A11
        LA[0, 8] = 1; LA[1, 9] = 1; LA[2, 10] = 1; LA[3, 11] = 1 # A12
        
        LB = np.zeros((4, 16))
        LB[0, 12] = 1; LB[1, 13] = 1; LB[2, 14] = 1; LB[3, 15] = 1 # B22
        
        LC = np.zeros((16, 4))
        LC[8, 0] = 1; LC[9, 1] = 1; LC[10, 2] = 1; LC[11, 3] = 1 # C12
        LC[12, 0] = 1; LC[13, 1] = 1; LC[14, 2] = 1; LC[15, 3] = 1 # C22
        maps.append((LA, LB, LC))
        
        # M6: (A21-A11)(B11+B12) -> C11, C21
        LA = np.zeros((4, 16))
        LA[0, 4] = 1; LA[1, 5] = 1; LA[2, 6] = 1; LA[3, 7] = 1 # A21
        LA[0, 0] = -1; LA[1, 1] = -1; LA[2, 2] = -1; LA[3, 3] = -1 # -A11
        
        LB = np.zeros((4, 16))
        LB[0, 0] = 1; LB[1, 1] = 1; LB[2, 2] = 1; LB[3, 3] = 1 # B11
        LB[0, 8] = 1; LB[1, 9] = 1; LB[2, 10] = 1; LB[3, 11] = 1 # B12
        
        LC = np.zeros((16, 4))
        LC[0, 0] = 1; LC[1, 1] = 1; LC[2, 2] = 1; LC[3, 3] = 1 # C11
        LC[4, 0] = 1; LC[5, 1] = 1; LC[6, 2] = 1; LC[7, 3] = 1 # C21
        maps.append((LA, LB, LC))
        
        # M7: (A12-A22)(B21+B22) -> C11, C12
        LA = np.zeros((4, 16))
        LA[0, 8] = 1; LA[1, 9] = 1; LA[2, 10] = 1; LA[3, 11] = 1 # A12
        LA[0, 12] = -1; LA[1, 13] = -1; LA[2, 14] = -1; LA[3, 15] = -1 # -A22
        
        LB = np.zeros((4, 16))
        LB[0, 4] = 1; LB[1, 5] = 1; LB[2, 6] = 1; LB[3, 7] = 1 # B21
        LB[0, 12] = 1; LB[1, 13] = 1; LB[2, 14] = 1; LB[3, 15] = 1 # B22
        
        LC = np.zeros((16, 4))
        LC[0, 0] = 1; LC[1, 1] = 1; LC[2, 2] = 1; LC[3, 3] = 1 # C11
        LC[8, 0] = 1; LC[9, 1] = 1; LC[10, 2] = 1; LC[11, 3] = 1 # C12
        maps.append((LA, LB, LC))
        
        return maps

    block_maps = get_block_maps()
    
    U_44 = np.zeros((16, 49))
    V_44 = np.zeros((16, 49))
    W_44 = np.zeros((16, 49))
    
    for p_idx in range(7):
        LA, LB, LC = block_maps[p_idx]
        for q_idx in range(7):
            col = p_idx * 7 + q_idx
            U_44[:, col] = LA.T.dot(U2[:, q_idx])
            V_44[:, col] = LB.T.dot(V2[:, q_idx])
            W_44[:, col] = LC.dot(W2[:, q_idx])
            
    # Vector Multiplication 4x4x1 (Rank 16)
    # A is 4x4 (16), B is 4x1 (4), C is 4x1 (4)
    # We need to map this to the global U, V, W.
    # Global U is 16x65. Global V is 20x65. Global W is 20x65.
    # Wait, B is 4x5. We split B into B1 (4x4) and B2 (4x1).
    # The 49 terms use B1.
    # The 16 terms use B2.
    # So V_44 should be extended to 20 columns? No, V is 20x65.
    # The first 16 columns of V_44 correspond to B1.
    # The last 4 columns of V_44 are 0.
    # The 16 vector terms will have non-zeros in the last 4 columns of V.
    
    U_vec = np.zeros((16, 16))
    V_vec = np.zeros((20, 16))
    W_vec = np.zeros((20, 16))
    
    for i in range(4):
        for j in range(4):
            k = i * 4 + j
            U_vec[i * 4 + j, k] = 1
            V_vec[j, k] = 1 # B index j (0..3)
            W_vec[4 * i, k] = 1 # C index i (0..3). C is 4x1. Flattened: 0..3.
            
    # Combine
    U_total = np.hstack([U_44, U_vec])
    V_total = np.hstack([np.vstack([V_44, np.zeros((4, 49))]), V_vec])
    W_total = np.hstack([np.vstack([W_44, np.zeros((4, 49))]), W_vec])
    
    # Convert to lists
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    result = {"U": U_list, "V": V_list, "W": W_list}
    
    # Verify
    def check_decomposition(U, V, W):
        rank = len(U[0])
        if len(U) != n * m or len(V) != m * p or len(W) != p * n:
            return False
        if len(V[0]) != rank or len(W[0]) != rank:
            return False
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for c in range(rank):
                        total += U[i * m + j][c] * V[j * p + k][c] * W[k * n + i][c]
                    if abs(total - 1) > 1e-9:
                        return False
        return True

    if check_decomposition(U_list, V_list, W_list):
        if 65 < best_rank:
            best_rank = 65
            best_result = result
            
    return best_result
# EVOLVE-BLOCK-END
