"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # U2, V2, W2 for 2x2 Strassen
    # U2[i*2+j, r], V2[j*2+k, r], W2[k*2+i, r]
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # r=0 (M1)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1
    
    # r=1 (M2)
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1
    
    # r=2 (M3)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[2][2] = 1
    
    # r=3 (M4)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[3][3] = 1
    
    # r=4 (M5)
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[3][4] = 1
    
    # r=5 (M6)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[1][5] = 1
    
    # r=6 (M7)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = -1; W2[1][6] = 1
    
    # Global coefficients
    # alpha[l][IJ], beta[l][JK], gamma[l][IK]
    # I, J, K in 0..1
    alpha = [[0]*4 for _ in range(7)]
    beta = [[0]*4 for _ in range(7)]
    gamma = [[0]*4 for _ in range(7)]
    
    # l=0 (M1)
    alpha[0][0] = 1; alpha[0][3] = 1
    beta[0][0] = 1; beta[0][3] = 1
    gamma[0][0] = 1; gamma[0][3] = 1
    
    # l=1 (M2)
    alpha[1][2] = 1; alpha[1][3] = 1
    beta[1][0] = 1
    gamma[1][1] = 1
    
    # l=2 (M3)
    alpha[2][0] = 1
    beta[2][1] = 1; beta[2][3] = -1
    gamma[2][1] = 1
    
    # l=3 (M4)
    alpha[3][3] = 1
    beta[3][2] = 1; beta[3][0] = -1
    gamma[3][0] = 1; gamma[3][3] = 1
    
    # l=4 (M5)
    alpha[4][0] = 1; alpha[4][1] = 1
    beta[4][3] = 1
    gamma[4][0] = -1; gamma[4][3] = 1
    
    # l=5 (M6)
    alpha[5][2] = 1; alpha[5][0] = -1
    beta[5][0] = 1; beta[5][1] = 1
    gamma[5][2] = 1
    
    # l=6 (M7)
    alpha[6][1] = 1; alpha[6][3] = -1
    beta[6][2] = 1; beta[6][3] = 1
    gamma[6][0] = -1; gamma[6][2] = 1
    
    # Build U, V, W
    for l in range(7):
        for r in range(7):
            col = l * 7 + r
            for I in range(2):
                for J in range(2):
                    a = alpha[l][I*2+J]
                    if a != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u2_val = U2[i_prime*2+j_prime][r]
                                if u2_val != 0:
                                    i = I*2 + i_prime
                                    j = J*2 + j_prime
                                    U[i*4+j][col] += a * u2_val
                                    
            for J in range(2):
                for K in range(2):
                    b = beta[l][J*2+K]
                    if b != 0:
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v2_val = V2[j_prime*2+k_prime][r]
                                if v2_val != 0:
                                    j = J*2 + j_prime
                                    k = K*2 + k_prime
                                    V[j*4+k][col] += b * v2_val
                                    
            for I in range(2):
                for K in range(2):
                    g = gamma[l][I*2+K]
                    if g != 0:
                        for i_prime in range(2):
                            for k_prime in range(2):
                                w2_val = W2[k_prime*2+i_prime][r]
                                if w2_val != 0:
                                    k = K*2 + k_prime
                                    i = I*2 + i_prime
                                    W[k*4+i][col] += g * w2_val
                                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
