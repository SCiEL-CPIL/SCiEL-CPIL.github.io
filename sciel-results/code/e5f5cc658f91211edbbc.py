"""
Program to find a rank-minimal exact tensor decomposition for 2x5x6 matrix multiplication.
Approach: Block decomposition using Strassen's algorithm for 2x2x2 blocks and schoolbook for smaller ones.
"""

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 5, 6
    
    # 1. Define Strassen decomposition for 2x2x2
    # U: (i, j) -> 4 rows, 7 cols
    # V: (j, k) -> 4 rows, 7 cols
    # W: (k, i) -> 4 rows, 7 cols (indices k*n + i)
    
    # Strassen variables M1..M7:
    # M1 = (A11 + A22)(B11 + B22)
    # M2 = (A21 + A22)(B11)
    # M3 = (A11)(B21 - B22)
    # M4 = (A22)(B12 - B11)
    # M5 = (A11 + A12)(B22)
    # M6 = (A21 - A11)(B11 + B12)
    # M7 = (A12 - A22)(B21 + B22)
    
    # C11 = M1 + M4 - M5 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M6
    
    # U[i, j] contributes to A[i][j] in factor t.
    # V[j, k] contributes to B[j][k] in factor t.
    # W[k, i] contributes to C[i][k] in factor t (with sign).
    
    # Let's build U_str, V_str, W_str as 4x7 matrices.
    # Rows of U: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # Rows of V: (0,0), (0,1), (1,0), (1,1) -> indices 0,1,2,3
    # Rows of W: (0,0), (1,0), (0,1), (1,1) -> indices k*2+i.
    # k=0,i=0 -> 0
    # k=0,i=1 -> 1
    # k=1,i=0 -> 2
    # k=1,i=1 -> 3
    
    # Column 0: M1
    # A parts: A00(+1), A11(+1)
    # B parts: B00(+1), B11(+1)
    # C parts: C00(+1), C11(+1)
    
    # Let's fill columns 0-6 for U, V, W
    
    U_str = [[0]*7 for _ in range(4)]
    V_str = [[0]*7 for _ in range(4)]
    W_str = [[0]*7 for _ in range(4)]
    
    # Col 0: M1
    U_str[0][0] = 1; U_str[3][0] = 1 # A00, A11
    V_str[0][0] = 1; V_str[3][0] = 1 # B00, B11
    W_str[0][0] = 1 # C00
    W_str[3][0] = 1 # C11 (C[k=1,i=1] is index 3)
    
    # Col 1: M2
    # A parts: A10(+1), A11(+1)
    # B parts: B00(+1)
    # C parts: C10(+1)
    U_str[2][1] = 1; U_str[3][1] = 1
    V_str[0][1] = 1
    W_str[2][1] = 1 # C10 (k=1,i=0 -> idx 2)
    
    # Col 2: M3
    # A parts: A00(+1)
    # B parts: B10(+1), B11(-1)
    # C parts: C01(+1), C11(+1)
    U_str[0][2] = 1
    V_str[2][2] = 1; V_str[3][2] = -1
    W_str[1][2] = 1 # C01 (k=0,i=1 -> idx 1)
    W_str[3][2] = 1 # C11
    
    # Col 3: M4
    # A parts: A11(+1)
    # B parts: B01(+1), B00(-1)
    # C parts: C00(+1), C10(+1)
    U_str[3][3] = 1
    V_str[1][3] = 1; V_str[0][3] = -1
    W_str[0][3] = 1 # C00
    W_str[2][3] = 1 # C10
    
    # Col 4: M5
    # A parts: A00(+1), A01(+1)
    # B parts: B11(+1)
    # C parts: C00(-1), C01(+1)
    U_str[0][4] = 1; U_str[1][4] = 1
    V_str[3][4] = 1
    W_str[0][4] = -1 # C00
    W_str[1][4] = 1 # C01
    
    # Col 5: M6
    # A parts: A10(+1), A00(-1)
    # B parts: B00(+1), B01(+1)
    # C parts: C11(+1)
    U_str[2][5] = 1; U_str[0][5] = -1
    V_str[0][5] = 1; V_str[1][5] = 1
    W_str[3][5] = 1 # C11
    
    # Col 6: M7
    # A parts: A01(+1), A11(-1)
    # B parts: B10(+1), B11(+1)
    # C parts: C00(+1), C11(+1)
    U_str[1][6] = 1; U_str[3][6] = -1
    V_str[2][6] = 1; V_str[3][6] = 1
    W_str[0][6] = 1 # C00
    W_str[3][6] = 1 # C11
    
    # 2. Define Schoolbook decomposition for general n, m', p'
    # This is used for the 1x2 or 2x1 blocks? No, we have 2x1x2 and 2x2x1?
    # Actually my split was:
    # m groups: 2, 2, 1
    # p groups: 2, 2, 2
    # So we have blocks:
    # (2, 2, 2) -> Strassen
    # (2, 2, 2) -> Strassen
    # (2, 2, 2) -> Strassen
    # (2, 2, 2) -> Strassen
    # (2, 2, 2) -> Strassen
    # (2, 2, 2) -> Strassen
    # (2, 1, 2) -> Schoolbook Rank 4
    # (2, 1, 2) -> Schoolbook Rank 4
    # (2, 1, 2) -> Schoolbook Rank 4
    
    def get_schoolbook(n, m_prime, p_prime):
        rank = n * m_prime * p_prime
        U_s = [[0]*rank for _ in range(n * m_prime)]
        V_s = [[0]*rank for _ in range(m_prime * p_prime)]
        W_s = [[0]*rank for _ in range(p_prime * n)]
        
        col = 0
        for i in range(n):
            for j in range(m_prime):
                for k in range(p_prime):
                    U_s[i * m_prime + j][col] = 1
                    V_s[j * p_prime + k][col] = 1
                    W_s[k * n + i][col] = 1
                    col += 1
        return U_s, V_s, W_s

    # 3. Construct Global Matrices
    total_rank = 54
    U_global = [[0]*total_rank for _ in range(n * m)] # 10 x 54
    V_global = [[0]*total_rank for _ in range(m * p)] # 30 x 54
    W_global = [[0]*total_rank for _ in range(p * n)] # 12 x 54
    
    col_offset = 0
    # m_indices: [ [0,1], [2,3], [4] ]
    # p_indices: [ [0,1], [2,3], [4,5] ]
    m_groups = [[0,1], [2,3], [4]]
    p_groups = [[0,1], [2,3], [4,5]]
    
    for g_m, m_idx_list in enumerate(m_groups):
        m_prime = len(m_idx_list)
        m_start = m_idx_list[0]
        for g_p, p_idx_list in enumerate(p_groups):
            p_prime = len(p_idx_list)
            p_start = p_idx_list[0]
            
            if m_prime == 2 and p_prime == 2:
                # Use Strassen 2x2x2
                U_b, V_b, W_b = U_str, V_str, W_str
                b_rank = 7
            else:
                # m_prime is 1, p_prime is 2 (since all p_groups are size 2)
                # Or if m_prime 2, p_prime 1? No, p groups are all 2.
                # So this is 2x1x2
                U_b, V_b, W_b = get_schoolbook(n, m_prime, p_prime)
                b_rank = n * m_prime * p_prime # 2*1*2 = 4
            
            # Copy to global with offsets
            for r in range(b_rank):
                gc = col_offset + r
                # Update U
                for i in range(n):
                    for j_loc in range(m_prime):
                        u_loc_idx = i * m_prime + j_loc
                        val = U_b[u_loc_idx][r]
                        if val != 0:
                            j_glob = m_start + j_loc
                            u_glob_idx = i * m + j_glob
                            U_global[u_glob_idx][gc] = val
                
                # Update V
                for j_loc in range(m_prime):
                    for k_loc in range(p_prime):
                        v_loc_idx = j_loc * p_prime + k_loc
                        val = V_b[v_loc_idx][r]
                        if val != 0:
                            j_glob = m_start + j_loc
                            k_glob = p_start + k_loc
                            v_glob_idx = j_glob * p + k_glob
                            V_global[v_glob_idx][gc] = val
                
                # Update W
                # W is p_prime * n rows. Index k * n + i
                for k_loc in range(p_prime):
                    for i in range(n):
                        w_loc_idx = k_loc * n + i
                        val = W_b[w_loc_idx][r]
                        if val != 0:
                            k_glob = p_start + k_loc
                            w_glob_idx = k_glob * n + i
                            W_global[w_glob_idx][gc] = val
            
            col_offset += b_rank

    return {"U": U_global, "V": V_global, "W": W_global}
# EVOLVE-BLOCK-END
