import numpy as np
import time
from fractions import Fraction

START_TIME = time.time()
TIME_LIMIT = 110  # Safety margin for 120s limit

def construct():
    n, m, p = 3, 5, 7
    
    # Total rank calculation
    # Block 1 (M327): 39
    # Block 2 (M327): 39
    # Block 3 (M317): 21
    rank_total = 99
    
    # Time check - if we've spent too long, return immediately
    if time.time() - START_TIME > TIME_LIMIT - 5:
        # Return minimal valid schoolbook as fallback
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][idx] = 1
                    V_s[j*p+k][idx] = 1
                    W_s[k*n+i][idx] = 1
                    idx += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    def add_strassen(rank_offset, i0, j0, k0):
        # Adds M222 Strassen for A[i0:i0+2, j0:j0+2] * B[j0:j0+2, k0:k0+2]
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
        u_loc = [[0]*4 for _ in range(4)]
        v_loc = [[0]*4 for _ in range(2)]
        w_loc = [[0]*4 for _ in range(2)]
        
        # r=0: A00 B00
        u_loc[0][0] = 1
        v_loc[0][0] = 1
        w_loc[0][0] = 1
        
        # r=1: A01 B10
        u_loc[1][1] = 1
        v_loc[1][1] = 1
        w_loc[0][1] = 1
        
        # r=2: A10 B00
        u_loc[2][2] = 1
        v_loc[0][2] = 1
        w_loc[1][2] = 1
        
        # r=3: A11 B10
        u_loc[3][3] = 1
        v_loc[1][3] = 1
        w_loc[1][3] = 1
        
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
        for j_l in range(2):
            for k_l in range(7):
                r = j_l * 7 + k_l
                u_g_idx = (i0 + 0) * m + (j0 + j_l)
                U[u_g_idx][rank_offset + r] += 1
                
                v_g_idx = (j0 + j_l) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] += 1
                
                w_g_idx = (k0 + k_l) * n + (i0 + 0)
                W[w_g_idx][rank_offset + r] += 1

    def add_m317(rank_offset, i0, j0, k0):
        # Adds M317 for A[i0:i0+3, j0:j0+1] * B[j0:j0+1, k0:k0+7]
        for i_l in range(3):
            for k_l in range(7):
                r = i_l * 7 + k_l
                u_g_idx = (i0 + i_l) * m + (j0 + 0)
                U[u_g_idx][rank_offset + r] += 1
                
                v_g_idx = (j0 + 0) * p + (k0 + k_l)
                V[v_g_idx][rank_offset + r] += 1
                
                w_g_idx = (k0 + k_l) * n + (i0 + i_l)
                W[w_g_idx][rank_offset + r] += 1

    # Build the blocks
    # Block 1: M327 (cols 0,1)
    add_strassen(0, 0, 0, 0)
    add_strassen(7, 0, 0, 2)
    add_strassen(14, 0, 0, 4)
    add_m221(21, 0, 0, 6)
    add_m127(25, 2, 0, 0)
    
    # Block 2: M327 (cols 2,3)
    add_strassen(39, 0, 2, 0)
    add_strassen(46, 0, 2, 2)
    add_strassen(53, 0, 2, 4)
    add_m221(60, 0, 2, 6)
    add_m127(64, 2, 2, 0)
    
    # Block 3: M317 (col 4)
    add_m317(78, 0, 4, 0)
    
    # Verify the decomposition
    U_np = np.array(U, dtype=float)
    V_np = np.array(V, dtype=float)
    W_np = np.array(W, dtype=float)
    
    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank_total):
                    s += U_np[i*m+j, r] * V_np[j*p+k, r] * W_np[k*n+i, r]
                if abs(s - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break
    
    if not valid:
        # Fallback to schoolbook if something went wrong
        rank_school = n * m * p
        U_s = [[0] * rank_school for _ in range(n * m)]
        V_s = [[0] * rank_school for _ in range(m * p)]
        W_s = [[0] * rank_school for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_s[i*m+j][idx] = 1
                    V_s[j*p+k][idx] = 1
                    W_s[k*n+i][idx] = 1
                    idx += 1
        return {"U": U_s, "V": V_s, "W": W_s}
    
    return {"U": U, "V": V, "W": W}
