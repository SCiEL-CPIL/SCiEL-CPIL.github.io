import time
import sys

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    # Target: Rank 63 (7 * 9)
    # Structure: 2x2 blocks of 3x3 matrices.
    # T_6 = T_2 (Strassen, rank 7) x T_3 (Schoolbook, rank 9)
    
    n, m, p = DIMENSIONS
    
    # 1. Define Strassen (2x2) factors
    # U2, V2, W2 of size 4x7
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # P1 = (a+d)(e+h)
    U2[0][0] = 1
    U2[3][0] = 1
    V2[0][0] = 1
    V2[3][0] = 1
    W2[0][0] = 1
    W2[3][0] = 1
    
    # P2 = (c+d)e
    U2[2][1] = 1
    U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1
    W2[3][1] = -1
    
    # P3 = a(f-h)
    U2[0][2] = 1
    V2[1][2] = 1
    V2[3][2] = -1
    W2[2][2] = 1
    W2[3][2] = 1
    
    # P4 = d(g-e)
    U2[3][3] = 1
    V2[2][3] = 1
    V2[0][3] = -1
    W2[0][3] = 1
    W2[1][3] = 1
    
    # P5 = (a+b)h
    U2[0][4] = 1
    U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1
    W2[2][4] = 1
    
    # P6 = (c-a)(e+f)
    U2[2][5] = 1
    U2[0][5] = -1
    V2[0][5] = 1
    V2[1][5] = 1
    W2[3][5] = 1
    
    # P7 = (b-d)(g+h)
    U2[1][6] = 1
    U2[3][6] = -1
    V2[2][6] = 1
    V2[3][6] = 1
    W2[0][6] = 1
    
    # 2. Define Schoolbook (3x3) factors
    # U3, V3, W3 of size 9x9
    U3 = [[0]*9 for _ in range(9)]
    V3 = [[0]*9 for _ in range(9)]
    W3 = [[0]*9 for _ in range(9)]
    
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U3[i*3 + j][col] = 1
                V3[j*3 + k][col] = 1
                W3[k*3 + i][col] = 1
                col += 1
                
    # 3. Combine
    rank_2 = 7
    rank_3 = 9
    total_rank = rank_2 * rank_3 # 63
    
    U = [[0] * total_rank for _ in range(36)]
    V = [[0] * total_rank for _ in range(36)]
    W = [[0] * total_rank for _ in range(36)]
    
    # Mapping:
    # i = i2*3 + i3
    # j = j2*3 + j3
    # k = k2*3 + k3
    # u_idx = i*6 + j
    # v_idx = j*6 + k
    # w_idx = k*6 + i
    
    for r2 in range(rank_2):
        for r3 in range(rank_3):
            r = r2 * rank_3 + r3
            
            # U[u_idx, r] = U2[i2*2+j2, r2] * U3[i3*3+j3, r3]
            for i2 in range(2):
                for j2 in range(2):
                    val_u2 = U2[i2*2 + j2][r2]
                    if val_u2 == 0: continue
                    for i3 in range(3):
                        for j3 in range(3):
                            val_u3 = U3[i3*3 + j3][r3]
                            if val_u3 == 0: continue
                            u_idx = (i2*3 + i3)*6 + (j2*3 + j3)
                            U[u_idx][r] = val_u2 * val_u3
                            
            # V[v_idx, r] = V2[j2*2+k2, r2] * V3[j3*3+k3, r3]
            for j2 in range(2):
                for k2 in range(2):
                    val_v2 = V2[j2*2 + k2][r2]
                    if val_v2 == 0: continue
                    for j3 in range(3):
                        for k3 in range(3):
                            val_v3 = V3[j3*3 + k3][r3]
                            if val_v3 == 0: continue
                            v_idx = (j2*3 + j3)*6 + (k2*3 + k3)
                            V[v_idx][r] = val_v2 * val_v3
                            
            # W[w_idx, r] = W2[k2*2+i2, r2] * W3[k3*3+i3, r3]
            for k2 in range(2):
                for i2 in range(2):
                    val_w2 = W2[k2*2 + i2][r2]
                    if val_w2 == 0: continue
                    for k3 in range(3):
                        for i3 in range(3):
                            val_w3 = W3[k3*3 + i3][r3]
                            if val_w3 == 0: continue
                            w_idx = (k2*3 + k3)*6 + (i2*3 + i3)
                            W[w_idx][r] = val_w2 * val_w3

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
