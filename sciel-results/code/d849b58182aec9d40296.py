"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U2 = [[0] * 7 for _ in range(4)]
    V2 = [[0] * 7 for _ in range(4)]
    W2 = [[0] * 7 for _ in range(4)]
    
    # M1
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[2][0] = -1; W2[1][0] = 1; W2[3][0] = 1
    
    # M2
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[1][1] = 1; W2[3][1] = -1
    
    # M3
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1; W2[3][2] = 1
    
    # M4
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[1][3] = 1
    
    # M5
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[2][4] = 1
    
    # M6
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # M7
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1; W2[2][6] = -1
    
    for t in range(27):
        ip = t // 9
        jp = (t // 3) % 3
        kp = t % 3
        
        for s in range(7):
            r = s * 27 + t
            
            for bi in range(2):
                for bj in range(2):
                    i = 3 * bi + ip
                    j = 3 * bj + jp
                    u_row = i * 6 + j
                    val = U2[bi * 2 + bj][s]
                    if val != 0:
                        U[u_row][r] = val
                        
            for bj in range(2):
                for bk in range(2):
                    j = 3 * bj + jp
                    k = 3 * bk + kp
                    v_row = j * 6 + k
                    val = V2[bj * 2 + bk][s]
                    if val != 0:
                        V[v_row][r] = val
                        
            for bk in range(2):
                for bi in range(2):
                    k = 3 * bk + kp
                    i = 3 * bi + ip
                    w_row = k * 6 + i
                    val = W2[bk * 2 + bi][s]
                    if val != 0:
                        W[w_row][r] = val
                        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
