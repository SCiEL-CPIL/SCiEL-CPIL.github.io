import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Helper to create schoolbook decomposition for n x m x p
    def get_schoolbook(n, m, p):
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    # Helper to get Strassen 2x2x2 decomposition
    def get_strassen_222():
        # Rank 7
        U = np.zeros((4, 7))
        V = np.zeros((4, 7))
        W = np.zeros((4, 7))
        
        # Strassen's 7 multiplications
        # M1 = (A11 + A22)(B11 + B22)
        # M2 = (A21 + A22)B11
        # M3 = A11(B21 - B22)
        # M4 = A22(B12 - B11)
        # M5 = (A11 + A12)B22
        # M6 = (A21 - A11)(B11 + B12)
        # M7 = (A12 - A22)(B21 + B22)
        
        # C11 = M1 + M4 - M5 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Indices:
        # A: 0,1 (row 1), 2,3 (row 2) -> 0,1,2,3
        # B: 0,1 (col 1), 2,3 (col 2) -> 0,1,2,3
        # C: 0,1 (col 1), 2,3 (col 2) -> 0,1,2,3
        
        # M1: A11+A22, B11+B22
        U[:, 0] = [1, 0, 0, 1]
        V[:, 0] = [1, 0, 0, 1]
        W[0, 0] = 1; W[1, 0] = 0; W[2, 0] = 0; W[3, 0] = 1
        
        # M2: A21+A22, B11
        U[:, 1] = [0, 0, 1, 1]
        V[:, 1] = [1, 0, 0, 0]
        W[0, 1] = 0; W[1, 1] = 0; W[2, 1] = 1; W[3, 1] = 0
        
        # M3: A11, B21-B22
        U[:, 2] = [1, 0, 0, 0]
        V[:, 2] = [0, 0, 1, -1]
        W[0, 2] = 0; W[1, 2] = 1; W[2, 2] = 1; W[3, 2] = 0
        
        # M4: A22, B12-B11
        U[:, 3] = [0, 0, 0, 1]
        V[:, 3] = [-1, 1, 0, 0]
        W[0, 3] = 1; W[1, 3] = 0; W[2, 3] = 0; W[3, 3] = 1
        
        # M5: A11+A12, B22
        U[:, 4] = [1, 1, 0, 0]
        V[:, 4] = [0, 0, 0, 1]
        W[0, 4] = -1; W[1, 4] = 0; W[2, 4] = 0; W[3, 4] = 1
        
        # M6: A21-A11, B11+B12
        U[:, 5] = [-1, 0, 1, 0]
        V[:, 5] = [1, 1, 0, 0]
        W[0, 5] = 0; W[1, 5] = 0; W[2, 5] = 0; W[3, 5] = 1
        
        # M7: A12-A22, B21+B22
        U[:, 6] = [0, 1, 0, -1]
        V[:, 6] = [0, 0, 1, 1]
        W[0, 6] = 1; W[1, 6] = 0; W[2, 6] = 0; W[3, 6] = 0
        
        return U, V, W, 7

    # Get base decompositions
    U_s, V_s, W_s, R_s = get_strassen_222()
    U_sb, V_sb, W_sb, R_sb = get_schoolbook(2, 2, 1)
    
    # Component 1: 4x4x4 (Strassen x Strassen)
    U_c1 = np.kron(U_s, U_s)
    V_c1 = np.kron(V_s, V_s)
    W_c1 = np.kron(W_s, W_s)
    R_c1 = R_s * R_s # 49
    
    # Component 2: 4x4x2 (Schoolbook 2x2x1 x Strassen 2x2x2)
    # Note: Dimensions of M_221 are 2x2x1. M_222 are 2x2x2.
    # Kronecker product gives 4x4x2.
    U_c2 = np.kron(U_sb, U_s)
    V_c2 = np.kron(V_sb, V_s)
    W_c2 = np.kron(W_sb, W_s)
    R_c2 = R_sb * R_s # 4 * 7 = 28
    
    # Component 3: 4x1x6 (Schoolbook)
    U_c3, V_c3, W_c3, R_c3 = get_schoolbook(4, 1, 6)
    
    Total_Rank = R_c1 + R_c2 + R_c3
    # 49 + 28 + 24 = 101
    
    U_global = np.zeros((n * m, Total_Rank))
    V_global = np.zeros((m * p, Total_Rank))
    W_global = np.zeros((p * n, Total_Rank))
    
    # Fill Component 1: 4x4x4
    # A[0:4, 0:4], B[0:4, 0:4], C[0:4, 0:4]
    # U indices: i*4+j (0..15)
    # V indices: j*4+k (0..15)
    # W indices: k*4+i (0..15)
    
    for r in range(R_c1):
        # Map local U_c1 indices to global
        # Local U_c1 is 16x49. Global U is 20x101.
        # Local row i_l corresponds to global row i_l (since i_l < 16 and global rows 0..15 are A[0:4,0:4])
        U_global[0:16, r] = U_c1[:, r]
        
        # Local V_c1 is 16x49. Global V is 30x101.
        # Local V row j_l corresponds to global row j_l (since j_l < 16 and global rows 0..15 are B[0:4,0:4])
        V_global[0:16, r] = V_c1[:, r]
        
        # Local W_c1 is 16x49. Global W is 24x101.
        # Local W row k_l corresponds to global row k_l (since k_l < 16 and global rows 0..15 are C[0:4,0:4])
        W_global[0:16, r] = W_c1[:, r]
        
    # Fill Component 2: 4x4x2
    # A[0:4, 0:4], B[0:4, 4:6], C[0:4, 4:6]
    # U indices: i*4+j (0..15) -> Global 0..15
    # V indices: j*4+k_l (0..7) where k_l in 0,1 -> Global j*6 + (4+k_l)
    # W indices: k_l*4+i (0..7) where k_l in 0,1 -> Global (4+k_l)*4 + i
    
    offset_r = R_c1
    for r in range(R_c2):
        U_global[0:16, offset_r + r] = U_c2[:, r]
        
        # V mapping
        for j in range(4):
            for k_l in range(2):
                local_v_idx = j * 2 + k_l # Wait, V_c2 is from M_221 x M_222.
                # M_221 V is 2x2x1 -> 4 rows.
                # M_222 V is 2x2x2 -> 4 rows.
                # Kron V is 16 rows?
                # No, V_c2 shape is (2*2 * 2*2, 28) = (16, 28)?
                # M_221 is 2x2x1. V size 2*1=2? No, V is m*p. 2*1=2.
                # M_222 is 2x2x2. V size 2*2=4.
                # Kron V size 2*4=8.
                # Wait, get_schoolbook(2,2,1) returns V of size (2*1, 4) = (2, 4).
                # get_strassen_222 returns V of size (4, 7).
                # np.kron(V_sb, V_s) will be (2*4, 4*7) = (8, 28).
                # So V_c2 has 8 rows.
                # Local V_c2 row index corresponds to (j, k) of 4x2 matrix?
                # The Kronecker product structure:
                # V_c2[i1, i2] where i1 from V_sb (2x1) and i2 from V_s (2x2).
                # V_sb index: j (0,1).
                # V_s index: j_s, k_s.
                # This is getting complicated to map manually.
                
                # Let's just use the full matrix mapping.
                # V_c2 is 8x28.
                # Global V is 30x101.
                # We need to map the 8 rows of V_c2 to the correct rows in Global V.
                # The 4x4x2 tensor uses B[0:4, 4:6].
                # B is 5x6. Rows 0..3, Cols 4..5.
                # Global V index for B[j, k] is j*6 + k.
                # The local tensor is 4x4x2.
                # Local B is 4x2.
                # Local B index (j, k_l) maps to Global B index (j, 4+k_l).
                # How are the rows of V_c2 ordered?
                # V_c2 = kron(V_sb, V_s).
                # V_sb is for M_221 (2x2x1). Rows 0,1 correspond to j=0,1?
                # Actually, V_sb from get_schoolbook(2,2,1) has rows for (j, k).
                # j in 0,1. k in 0.
                # So V_sb rows are (0,0), (1,0).
                # V_s is for M_222 (2x2x2). Rows are (j_s, k_s).
                # j_s in 0,1. k_s in 0,1.
                # V_s rows are (0,0), (0,1), (1,0), (1,1).
                # Kron V_c2 rows are (j, k_s, k_l)?
                # No, kron(A, B) flattens (i_A, i_B).
                # So row index is i_A * N_B + i_B.
                # i_A is from V_sb (2 rows). i_B is from V_s (4 rows).
                # So 8 rows.
                # i_A corresponds to j (0,1).
                # i_B corresponds to (j_s, k_s).
                # This implies the local 4x4x2 tensor is constructed from blocks.
                # The "4" in 4x4 comes from 2 (from M_221) x 2 (from M_222)?
                # No, M_221 is 2x2x1. M_222 is 2x2x2.
                # Kron product is (2*2) x (2*2) x (1*2) = 4x4x2.
                # The first dimension 4 is (i_A, i_B) where i_A from M_221 A (2x2) and i_B from M_222 A (2x2).
                # So i = i_A * 2 + i_B.
                # The second dimension 4 is (j_A, j_B).
                # j = j_A * 2 + j_B.
                # The third dimension 2 is (k_A, k_B).
                # k = k_A * 2 + k_B.
                # M_221 has k_A=0 only.
                # So k = 0*2 + k_B = k_B.
                # So k is 0,1.
                
                # Mapping Global V:
                # Global V index for B[j, k_global].
                # j = j_A * 2 + j_B.
                # k_global = 4 + k_B.
                # V_c2 row index: i_A * 4 + i_B.
                # i_A is j_A. i_B is (j_B, k_B) index in V_s.
                # V_s row index for (j_B, k_B) is j_B * 2 + k_B.
                # So V_c2 row index = j_A * 4 + (j_B * 2 + k_B).
                
                for j_A in range(2):
                    for j_B in range(2):
                        for k_B in range(2):
                            local_v_row = j_A * 4 + (j_B * 2 + k_B)
                            global_j = j_A * 2 + j_B
                            global_k = 4 + k_B
                            global_v_row = global_j * 6 + global_k
                            V_global[global_v_row, offset_r + r] = V_c2[local_v_row, r]
                            
        # W mapping
        # W_c2 is 8x28.
        # W_c2 = kron(W_sb, W_s).
        # W_sb rows: (k_A, i_A). k_A=0. i_A in 0,1.
        # W_s rows: (k_B, i_B). k_B in 0,1. i_B in 0,1.
        # Kron W_c2 rows: (k_A, i_A, k_B, i_B).
        # Index = (k_A * 2 + i_A) * 4 + (k_B * 2 + i_B).
        # k_A=0. So index = i_A * 4 + (k_B * 2 + i_B).
        # Global W index: k_global * 4 + i_global.
        # k_global = 4 + k_B.
        # i_global = i_A * 2 + i_B.
        
        for i_A in range(2):
            for i_B in range(2):
                for k_B in range(2):
                    local_w_row = i_A * 4 + (k_B * 2 + i_B)
                    global_k = 4 + k_B
                    global_i = i_A * 2 + i_B
                    global_w_row = global_k * 4 + global_i
                    W_global[global_w_row, offset_r + r] = W_c2[local_w_row, r]

    # Fill Component 3: 4x1x6
    offset_r = R_c1 + R_c2
    # U_c3 is 4x24. Rows i (0..3).
    # Global U rows: i*5 + 4.
    for i in range(4):
        U_global[i * 5 + 4, offset_r:offset_r + R_c3] = U_c3[i, :]
        
    # V_c3 is 6x24. Rows k (0..5).
    # Global V rows: 4*6 + k.
    for k in range(6):
        V_global[4 * 6 + k, offset_r:offset_r + R_c3] = V_c3[k, :]
        
    # W_c3 is 24x24. Rows (k, i).
    # Global W rows: k*4 + i.
    # W_c3 row index is k*4 + i?
    # get_schoolbook(4,1,6) returns W of size (6*4, 24).
    # W[k*n + i, col].
    # So W_c3 row index is already k*4 + i.
    W_global[0:24, offset_r:offset_r + R_c3] = W_c3[:, :]
    
    # Verify
    U_r = U_global.reshape(n, m, Total_Rank)
    V_r = V_global.reshape(m, p, Total_Rank)
    W_r = W_global.reshape(p, n, Total_Rank)
    
    T_recon = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
    T_target = np.ones((n, m, p), dtype=np.float64)
    
    if np.allclose(T_recon, T_target, atol=1e-9):
        return {"U": U_global.tolist(), "V": V_global.tolist(), "W": W_global.tolist()}
    else:
        # Fallback to schoolbook
        U_school, V_school, W_school, R_school = get_schoolbook(n, m, p)
        return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
