import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Total rank calculation
    # Block 1 (M327): 39
    # Block 2 (M327): 39
    # Block 3 (M317): 21
    rank_total = 99
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    def add_strassen(rank_offset, i0, j0, k0):
        # Adds M222 Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
        # n_s=2, m_s=2, p_s=2
        # U local: 4x7
        # V local: 4x7
        # W local: 4x7
        
        # Coefficients defined in analysis
        # r, u_local_idx, v_local_idx, w_local_idx, val_u, val_v, val_w
        # Actually easier to just fill the 4x7 matrices and then map
        
        u_loc = [[0]*7 for _ in range(4)]
        v_loc = [[0]*7 for _ in range(4)]
        w_loc = [[0]*7 for _ in range(4)]
        
        # M1
        u_loc[0][0] = 1; u_loc[3][0] = 1
        v_loc[0][0] = 1; v_loc[3][0] = 1
        w_loc[0][0] = 1; w_loc[3][0] = 1
        
        # M2
        u_loc[2][1] = 1; u_loc[3][1] = 1
        v_loc[0][1] = 1
        w_loc[1][1] = 1; w_loc[3][1] = -1
        
        # M3
        u_loc[0][2] = 1
        v_loc[1][2] = 1; v_loc[3][2] = -1
        w_loc[2][2] = 1; w_loc[3][2] = 1
        
        # M4
        u_loc[3][3] = 1
        v_loc[2][3] = 1; v_loc[0][3] = -1
        w_loc[0][3] = 1; w_loc[1][3] = 1
        
        # M5
        u_loc[0][4] = 1; u_loc[1][4] = 1
        v_loc[3][4] = 1
        w_loc[0][4] = -1; w_loc[2][4] = 1
        
        # M6
        u_loc[2][5] = 1; u_loc[0][5] = -1
        v_loc[0][5] = 1; v_loc[1][5] = 1
        w_loc[3][5] = 1
        
        # M7
        u_loc[1][6] = 1; u_loc[3][6] = -1
        v_loc[2][6] = 1; v_loc[3][6] = 1
        w_loc[0][6] = 1
        
        # Map to global
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(7):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(2):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(7):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*2 + k_l][r]
        
        for k_l in range(2):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(7):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]

    def add_m221(rank_offset, i0, j0, k0):
        # Adds M221 for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+1]
        # Rank 4
        # C0 = A00 B00 + A01 B10
        # C1 = A10 B00 + A11 B10
        
        u_loc = [[0]*4 for _ in range(4)]
        v_loc = [[0]*4 for _ in range(4)]
        w_loc = [[0]*4 for _ in range(4)]
        
        # r=0: A00 B00
        u_loc[0][0] = 1
        v_loc[0][0] = 1
        w_loc[0][0] = 1 # C0 (k=0, i=0)
        
        # r=1: A01 B10
        u_loc[1][1] = 1
        v_loc[2][1] = 1 # B10 is j=1, k=0 -> 1*1+0 = 1? No, p_s=1.
        # V index: j_l * p_s + k_l. p_s=1.
        # B00: j=0, k=0 -> 0.
        # B10: j=1, k=0 -> 1.
        # So v_loc[1][1] = 1?
        # Wait. V index is j*p+k.
        # j_l=1, k_l=0. p_s=1. Index = 1*1+0 = 1.
        # So v_loc[1][1] = 1.
        
        # Let's re-verify V index for M221.
        # m_s=2, p_s=1.
        # V rows: 0, 1.
        # V[0] corresponds to B[0,0] (j=0, k=0).
        # V[1] corresponds to B[1,0] (j=1, k=0).
        
        # My previous mapping:
        # v_loc[0][0] = 1 (B00)
        # v_loc[1][1] = 1 (B10)
        
        # C0 = A00 B00 + A01 B10
        # r=0: A00 B00. u_loc[0][0]=1, v_loc[0][0]=1. w_loc[0][0]=1 (C0).
        # r=1: A01 B10. u_loc[1][1]=1 (A01 is i=0, j=1 -> 1). v_loc[1][1]=1 (B10 is j=1, k=0 -> 1). w_loc[0][1]=1 (C0).
        
        # C1 = A10 B00 + A11 B10
        # r=2: A10 B00. u_loc[2][2]=1 (A10 is i=1, j=0 -> 2). v_loc[0][2]=1 (B00). w_loc[1][2]=1 (C1 is k=0, i=1 -> 1).
        # r=3: A11 B10. u_loc[3][3]=1 (A11 is i=1, j=1 -> 3). v_loc[1][3]=1 (B10). w_loc[1][3]=1 (C1).
        
        # Correcting u_loc indices:
        # U index: i_l * m_s + j_l. m_s=2.
        # A00: 0. A01: 1. A10: 2. A11: 3.
        
        # Correcting w_loc indices:
        # W index: k_l * n_s + i_l. n_s=2.
        # C0 (i=0, k=0): 0.
        # C1 (i=1, k=0): 1.
        
        # Let's fill:
        # r=0
        u_loc[0][0] = 1
        v_loc[0][0] = 1
        w_loc[0][0] = 1
        
        # r=1
        u_loc[1][1] = 1
        v_loc[1][1] = 1
        w_loc[0][1] = 1
        
        # r=2
        u_loc[2][2] = 1
        v_loc[0][2] = 1
        w_loc[1][2] = 1
        
        # r=3
        u_loc[3][3] = 1
        v_loc[1][3] = 1
        w_loc[1][3] = 1
        
        # Map to global
        for i_l in range(2):
            for j_l in range(2):
                u_g_idx = (i0 + i_l) * m + (j0 + j_l)
                for r in range(4):
                    U[u_g_idx][rank_offset + r] += u_loc[i_l*2 + j_l][r]
        
        for j_l in range(2):
            for k_l in range(1):
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                for r in range(4):
                    V[v_g_idx][rank_offset + r] += v_loc[j_l*1 + k_l][r]
        
        for k_l in range(1):
            for i_l in range(2):
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                for r in range(4):
                    W[w_g_idx][rank_offset + r] += w_loc[k_l*2 + i_l][r]

    def add_m127(rank_offset, i0, j0, k0):
        # Adds M127 for A[i0:i0+1, j0:j0+2] * B[j0:j0+2, k0:k0+7]
        # Rank 14
        # C_k = A0 B0k + A1 B1k for k=0..6
        # 7 dot products of length 2.
        # Each dot product rank 2.
        
        # We can just use schoolbook for these 14 terms.
        # Term r: A[0, j] * B[j, k]
        # There are 2 * 7 = 14 terms.
        
        for j_l in range(2):
            for k_l in range(7):
                r = j_l * 7 + k_l
                # U: A[0, j_l] -> i_l=0, j_l. Index 0*2 + j_l = j_l.
                u_g_idx = (i0 + 0) * m + (j0 + j_l)
                U[u_g_idx][rank_offset + r] += 1
                
                # V: B[j_l, k_l] -> j_l, k_l. Index j_l*7 + k_l.
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] += 1
                
                # W: C[0, k_l] -> i_l=0, k_l. Index k_l*1 + 0 = k_l?
                # W index: k_l * n_s + i_l. n_s=1.
                # Index = k_l * 1 + 0 = k_l.
                w_g_idx = (k0 + k_l) * n + (i0 + 0)
                W[w_g_idx][rank_offset + r] += 1

    def add_m317(rank_offset, i0, j0, k0):
        # Adds M317 for A[i0:i0+3, j0:j0+1] * B[j0:j0+1, k0:k0+7]
        # Rank 21
        # C_ik = A_i0 B_0k
        # 3 * 7 = 21 terms.
        
        for i_l in range(3):
            for k_l in range(7):
                r = i_l * 7 + k_l
                # U: A[i_l, 0] -> i_l, 0. Index i_l*1 + 0 = i_l.
                u_g_idx = (i0 + i_l) * m + (j0 + 0)
                U[u_g_idx][rank_offset + r] += 1
                
                # V: B[0, k_l] -> 0, k_l. Index 0*7 + k_l = k_l.
                v_g_idx = (j0 + 0) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] += 1
                
                # W: C[i_l, k_l] -> i_l, k_l. Index k_l*3 + i_l.
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                W[w_g_idx][rank_offset + r] += 1

    # Build the blocks
    # Block 1: M327 (cols 0,1)
    # M227 (rows 0,1)
    add_strassen(0, 0, 0, 0)   # C cols 0,1
    add_strassen(7, 0, 0, 2)   # C cols 2,3
    add_strassen(14, 0, 0, 4)  # C cols 4,5
    add_m221(21, 0, 0, 6)      # C col 6
    
    # M127 (row 2)
    add_m127(25, 2, 0, 0)      # C cols 0-6
    
    # Block 2: M327 (cols 2,3)
    # M227 (rows 0,1)
    add_strassen(39, 0, 2, 0)  # C cols 0,1
    add_strassen(46, 0, 2, 2)  # C cols 2,3
    add_strassen(53, 0, 2, 4)  # C cols 4,5
    add_m221(60, 0, 2, 6)      # C col 6
    
    # M127 (row 2)
    add_m127(64, 2, 2, 0)      # C cols 0-6
    
    # Block 3: M317 (col 4)
    add_m317(78, 0, 4, 0)      # C cols 0-6
    
    return {"U": U, "V": V, "W": W}
