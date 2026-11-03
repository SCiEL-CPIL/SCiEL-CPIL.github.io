"""Rewritten program: exact decomposition using Strassen-based rank-114 algorithm for 4x4 times 4x9."""

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen decomposition for M_{2,2,2}
    # U_S, V_S, W_S are 4x7 matrices
    # Indices: U: (i,j), V: (j,k), W: (i,k)
    # 0:00, 1:01, 2:10, 3:11
    
    U_S = [[0]*7 for _ in range(4)]
    V_S = [[0]*7 for _ in range(4)]
    W_S = [[0]*7 for _ in range(4)]
    
    # Term 1: (a+d)(e+h) -> C00, C11
    U_S[0][0] = 1; U_S[3][0] = 1
    V_S[0][0] = 1; V_S[3][0] = 1
    W_S[0][0] = 1; W_S[3][0] = 1
    
    # Term 2: (c+d)e -> C10, C11 (neg)
    U_S[2][1] = 1; U_S[3][1] = 1
    V_S[0][1] = 1
    W_S[2][1] = 1; W_S[3][1] = -1
    
    # Term 3: a(f-h) -> C01, C11
    U_S[0][2] = 1
    V_S[1][2] = 1; V_S[3][2] = -1
    W_S[1][2] = 1; W_S[3][2] = 1
    
    # Term 4: d(g-e) -> C00, C10
    U_S[3][3] = 1
    V_S[2][3] = 1; V_S[0][3] = -1
    W_S[0][3] = 1; W_S[2][3] = 1
    
    # Term 5: (a+b)h -> C00 (neg), C01
    U_S[0][4] = 1; U_S[1][4] = 1
    V_S[3][4] = 1
    W_S[0][4] = -1; W_S[1][4] = 1
    
    # Term 6: (c-d)(e+f) -> C11
    U_S[2][5] = 1; U_S[3][5] = -1
    V_S[0][5] = 1; V_S[1][5] = 1
    W_S[3][5] = 1
    
    # Term 7: (b-d)(g+h) -> C00, C11 (neg)
    U_S[1][6] = 1; U_S[3][6] = -1
    V_S[2][6] = 1; V_S[3][6] = 1
    W_S[0][6] = 1; W_S[3][6] = -1
    
    # 2. Construct M_{4,4,4} decomposition using Kronecker product
    # Rank 49
    U_444 = []
    V_444 = []
    W_444 = []
    
    for k in range(7):
        for l in range(7):
            # u = U_S[:,k] (kron) U_S[:,l]
            # Indices for 4x4: (i1,i2,j1,j2) -> i1*4 + i2? No.
            # A is 4x4. Index (i,j) where i=0..3, j=0..3.
            # i = 2*i1 + i2, j = 2*j1 + j2.
            # U_S index is i*2 + j? No, U_S is 4x7. Index 0..3 corresponds to (0,0), (0,1), (1,0), (1,1).
            # So U_S[i2*2 + j2][k].
            
            u_vec = [0]*16
            v_vec = [0]*16
            w_vec = [0]*16
            
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            # A index: i = 2*i1 + i2, j = 2*j1 + j2
                            # U_S index for (i1, j1) is i1*2 + j1
                            # U_S index for (i2, j2) is i2*2 + j2
                            
                            val_u = U_S[i1*2+j1][k] * U_S[i2*2+j2][l]
                            if val_u != 0:
                                u_vec[(2*i1+i2)*4 + (2*j1+j2)] = val_u
                            
                            # B index: j = 2*j1 + j2, k = 2*k1 + k2
                            # V_S index for (j1, k1) is j1*2 + k1
                            # V_S index for (j2, k2) is j2*2 + k2
                            
                            val_v = V_S[j1*2+k1][k] * V_S[j2*2+k2][l] # Wait, k1,k2 are loop vars?
                            # We need to sum over all k1,k2
                            # Actually, the Kronecker product of vectors u, v is:
                            # (u kron v)_{(i1,i2), (j1,j2)} = u_{i1,j1} v_{i2,j2}
                            # Here indices are (i,j) for U, (j,k) for V, (i,k) for W.
                            
                            # Let's just compute the full vectors.
                            pass
            
            # Simpler: Compute u_vec, v_vec, w_vec by iterating all 16 indices
            for idx_u in range(16):
                i1 = idx_u // 8
                rem = idx_u % 8
                i2 = rem // 4
                j1 = rem % 4 // 2
                j2 = rem % 2
                # Wait, mapping:
                # i = 2*i1 + i2, j = 2*j1 + j2
                # idx_u = i*4 + j = (2*i1+i2)*4 + (2*j1+j2) = 8*i1 + 4*i2 + 2*j1 + j2
                # So i1 = idx_u // 8
                # rem = idx_u % 8
                # i2 = rem // 4
                # rem2 = rem % 4
                # j1 = rem2 // 2
                # j2 = rem2 % 2
                
                i1 = idx_u // 8
                rem = idx_u % 8
                i2 = rem // 4
                rem2 = rem % 4
                j1 = rem2 // 2
                j2 = rem2 % 2
                
                u_val = U_S[i1*2+j1][k] * U_S[i2*2+j2][l]
                u_vec[idx_u] = u_val
                
                # V index: (j, k) where j = 2*j1 + j2, k = 2*k1 + k2
                # idx_v = j*4 + k = (2*j1+j2)*4 + (2*k1+k2) = 8*j1 + 4*j2 + 2*k1 + k2
                # We need to sum? No, the vector is fixed for the term.
                # The term is defined by u_vec, v_vec, w_vec.
                # v_vec[idx_v] = V_S[j1*2+k1][k] * V_S[j2*2+k2][l]
                # But idx_v depends on k1, k2.
                # So we iterate idx_v from 0 to 15.
                
            v_vec = [0]*16
            for idx_v in range(16):
                j1 = idx_v // 8
                rem = idx_v % 8
                j2 = rem // 4
                rem2 = rem % 4
                k1 = rem2 // 2
                k2 = rem2 % 2
                
                v_val = V_S[j1*2+k1][k] * V_S[j2*2+k2][l]
                v_vec[idx_v] = v_val
                
            w_vec = [0]*16
            for idx_w in range(16):
                i1 = idx_w // 8
                rem = idx_w % 8
                i2 = rem // 4
                rem2 = rem % 4
                k1 = rem2 // 2
                k2 = rem2 % 2
                
                w_val = W_S[i1*2+k1][k] * W_S[i2*2+k2][l]
                w_vec[idx_w] = w_val
                
            U_444.append(u_vec)
            V_444.append(v_vec)
            W_444.append(w_vec)
            
    # 3. Construct M_{4,4,9}
    # Rank 49 (cols 0-3) + 49 (cols 4-7) + 16 (col 8)
    rank = 49 + 49 + 16
    U = [[0]*rank for _ in range(16)]
    V = [[0]*rank for _ in range(36)] # 4*9
    W = [[0]*rank for _ in range(36)] # 9*4
    
    col_offset = 0
    
    # Part 1: Cols 0-3
    for r in range(49):
        # U is same
        for i in range(16):
            U[i][col_offset + r] = U_444[r][i]
            
        # V: indices (j, k) where k=0..3
        # V_444[r] has indices (j, k) for k=0..3.
        # In V matrix, index is j*9 + k.
        for idx_v in range(16):
            j = idx_v // 4
            k = idx_v % 4
            V[j*9 + k][col_offset + r] = V_444[r][idx_v]
            
        # W: indices (k, i) where k=0..3
        # W_444[r] has indices (i, k) for i=0..3, k=0..3.
        # In W matrix, index is k*4 + i.
        for idx_w in range(16):
            i = idx_w // 4
            k = idx_w % 4
            W[k*4 + i][col_offset + r] = W_444[r][idx_w]
            
    col_offset += 49
    
    # Part 2: Cols 4-7
    for r in range(49):
        for i in range(16):
            U[i][col_offset + r] = U_444[r][i]
            
        # V: indices (j, k) where k=4..7
        # Map local k (0..3) to global k (4..7)
        for idx_v in range(16):
            j = idx_v // 4
            k_local = idx_v % 4
            k_global = 4 + k_local
            V[j*9 + k_global][col_offset + r] = V_444[r][idx_v]
            
        # W: indices (k, i) where k=4..7
        for idx_w in range(16):
            i = idx_w // 4
            k_local = idx_w % 4
            k_global = 4 + k_local
            W[k_global*4 + i][col_offset + r] = W_444[r][idx_w]
            
    col_offset += 49
    
    # Part 3: Col 8 (Schoolbook M_{4,4,1})
    # 16 terms
    for i in range(4):
        for j in range(4):
            # Term for A_{i,j} * B_{j,8} -> C_{i,8}
            U[i*4 + j][col_offset] = 1
            V[j*9 + 8][col_offset] = 1
            W[8*4 + i][col_offset] = 1
            col_offset += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
