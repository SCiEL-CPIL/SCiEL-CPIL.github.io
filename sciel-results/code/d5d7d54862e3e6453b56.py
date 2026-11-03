"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]
import time

# EVOLVE-BLOCK-START

def construct():
    """Construct rank-98 tensor decomposition for 4x4 times 4x8 matrix multiplication.
    
    Uses double Strassen: [4,4,4] has rank 49 (Strassen at both 4x4 and 2x2 levels),
    and [4,4,8] = two independent [4,4,4] problems, giving rank 98.
    """
    n, m, p = DIMENSIONS  # 4, 4, 8
    
    # Strassen [2,2,2] decomposition (rank 7)
    # U_str[i*2+j][c], V_str[j*2+k][c], W_str[k*2+i][c] for c in 0..6
    U_str = [[0]*7 for _ in range(4)]
    V_str = [[0]*7 for _ in range(4)]
    W_str = [[0]*7 for _ in range(4)]
    
    # M1 = (A11+A22)(B11+B22)
    U_str[0][0] = 1; U_str[3][0] = 1
    V_str[0][0] = 1; V_str[3][0] = 1
    W_str[0][0] = 1; W_str[3][0] = 1  # C11+=M1, C22+=M1
    
    # M2 = (A21+A22)B11
    U_str[2][1] = 1; U_str[3][1] = 1
    V_str[0][1] = 1
    W_str[2][1] = 1; W_str[3][1] = -1  # C21+=M2, C22-=M2
    
    # M3 = A11(B21-B22)
    U_str[0][2] = 1
    V_str[2][2] = 1; V_str[3][2] = -1
    W_str[1][2] = 1; W_str[3][2] = 1  # C12+=M3, C22+=M3
    
    # M4 = A22(B12-B11)
    U_str[3][3] = 1
    V_str[1][3] = 1; V_str[0][3] = -1
    W_str[0][3] = 1; W_str[2][3] = 1  # C11+=M4, C21+=M4
    
    # M5 = (A11+A12)B22
    U_str[0][4] = 1; U_str[1][4] = 1
    V_str[3][4] = 1
    W_str[0][4] = -1; W_str[1][4] = 1  # C11-=M5, C12+=M5
    
    # M6 = (A21-A11)(B11+B12)
    U_str[2][5] = 1; U_str[0][5] = -1
    V_str[0][5] = 1; V_str[1][5] = 1
    W_str[3][5] = 1  # C22+=M6
    
    # M7 = (A12-A22)(B21+B22)
    U_str[1][6] = 1; U_str[3][6] = -1
    V_str[2][6] = 1; V_str[3][6] = 1
    W_str[0][6] = 1  # C11+=M7
    
    # Verify Strassen [2,2,2]:
    # C[0,0] = M1+M4-M5+M7: W[0][0]=1, W[0][3]=1, W[0][4]=-1, W[0][6]=1 ✓
    # C[0,1] = M3+M5: W[1][2]=1, W[1][4]=1 ✓
    # C[1,0] = M2+M4: W[2][1]=1, W[2][3]=1 ✓
    # C[1,1] = M1-M2+M3+M6: W[3][0]=1, W[3][1]=-1, W[3][2]=1, W[3][5]=1 ✓
    
    # Build [4,4,4] via tensor product of two Strassen [2,2,2] decompositions
    # Rank = 7*7 = 49
    # U_4[i*4+j][c_big*7+c_small] = U_str[i_big*2+j_big][c_big] * U_str[i_small*2+j_small][c_small]
    # where i = i_big*2+i_small, j = j_big*2+j_small
    U_4 = [[0]*49 for _ in range(16)]
    V_4 = [[0]*49 for _ in range(16)]
    W_4 = [[0]*49 for _ in range(16)]
    
    for i_big in range(2):
        for i_small in range(2):
            for j_big in range(2):
                for j_small in range(2):
                    i = i_big*2 + i_small
                    j = j_big*2 + j_small
                    u_idx = i*4 + j
                    for c_big in range(7):
                        ub = U_str[i_big*2+j_big][c_big]
                        if ub == 0:
                            continue
                        for c_small in range(7):
                            us = U_str[i_small*2+j_small][c_small]
                            if us == 0:
                                continue
                            col = c_big*7 + c_small
                            U_4[u_idx][col] = ub * us
    
    for j_big in range(2):
        for j_small in range(2):
            for k_big in range(2):
                for k_small in range(2):
                    j = j_big*2 + j_small
                    k = k_big*2 + k_small
                    v_idx = j*4 + k
                    for c_big in range(7):
                        vb = V_str[j_big*2+k_big][c_big]
                        if vb == 0:
                            continue
                        for c_small in range(7):
                            vs = V_str[j_small*2+k_small][c_small]
                            if vs == 0:
                                continue
                            col = c_big*7 + c_small
                            V_4[v_idx][col] = vb * vs
    
    for k_big in range(2):
        for k_small in range(2):
            for i_big in range(2):
                for i_small in range(2):
                    k = k_big*2 + k_small
                    i = i_big*2 + i_small
                    w_idx = k*4 + i
                    for c_big in range(7):
                        wb = W_str[k_big*2+i_big][c_big]
                        if wb == 0:
                            continue
                        for c_small in range(7):
                            ws = W_str[k_small*2+i_small][c_small]
                            if ws == 0:
                                continue
                            col = c_big*7 + c_small
                            W_4[w_idx][col] = wb * ws
    
    # Now assemble [4,4,8] = [4,4,4] (cols 0-3) + [4,4,4] (cols 4-7)
    rank = 98
    U = [[0] * rank for _ in range(n * m)]  # 16 x 98
    V = [[0] * rank for _ in range(m * p)]  # 32 x 98
    W = [[0] * rank for _ in range(p * n)]  # 32 x 98
    
    # First [4,4,4]: B cols 0-3, C cols 0-3
    for i_idx in range(16):
        for c in range(49):
            U[i_idx][c] = U_4[i_idx][c]
    
    for v_idx in range(16):
        for c in range(49):
            V[v_idx][c] = V_4[v_idx][c]
    
    for w_idx in range(16):
        for c in range(49):
            W[w_idx][c] = W_4[w_idx][c]
    
    # Second [4,4,4]: B cols 4-7, C cols 4-7
    for i_idx in range(16):
        for c in range(49):
            U[i_idx][c+49] = U_4[i_idx][c]
    
    for j in range(4):
        for k_prime in range(4):
            v_idx = j*4 + (k_prime+4)
            v_idx_4 = j*4 + k_prime
            for c in range(49):
                V[v_idx][c+49] = V_4[v_idx_4][c]
    
    for i_prime in range(4):
        for k_prime in range(4):
            w_idx = (k_prime+4)*4 + i_prime
            w_idx_4 = k_prime*4 + i_prime
            for c in range(49):
                W[w_idx][c+49] = W_4[w_idx_4][c]
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END

