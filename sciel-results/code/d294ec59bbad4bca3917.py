import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120  # seconds
    
    n, m, p = 4, 4, 9
    rank_444 = 49
    rank_441 = 4
    total_rank = 2 * rank_444 + rank_441
    
    # Precompute Strassen coefficients for 2x2x2
    # f_t(x) and g_t(y) coefficients
    # t: 0..6
    # p, q: 0..1
    beta = [[0]*2 for _ in range(2)]  # beta[t][p][q]
    delta = [[0]*2 for _ in range(2)] # delta[t][p][q]
    
    # t=0
    beta[0][0][0] = 1; beta[0][1][1] = 1
    delta[0][0][0] = 1; delta[0][1][1] = 1
    # t=1
    beta[1][1][0] = 1; beta[1][1][1] = 1
    delta[1][0][0] = 1
    # t=2
    beta[2][0][0] = 1
    delta[2][0][1] = 1; delta[2][1][1] = -1
    # t=3
    beta[3][1][1] = 1
    delta[3][1][0] = 1; delta[3][0][0] = -1
    # t=4
    beta[4][0][0] = 1; beta[4][0][1] = 1
    delta[4][1][1] = 1
    # t=5
    beta[5][1][0] = 1; beta[5][0][0] = -1
    delta[5][0][0] = 1; delta[5][0][1] = 1
    # t=6
    beta[6][0][1] = 1; beta[6][1][1] = -1
    delta[6][1][0] = 1; delta[6][1][1] = 1
    
    # gamma_s[a][b] for X_s
    gamma = [[[0]*2 for _ in range(2)] for _ in range(7)]
    # eta_s[c][d] for Y_s
    eta = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # s=0
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    eta[0][0][0] = 1; eta[0][1][1] = 1
    # s=1
    gamma[1][1][0] = 1; gamma[1][1][1] = 1
    eta[1][0][0] = 1
    # s=2
    gamma[2][0][0] = 1
    eta[2][0][1] = 1; eta[2][1][1] = -1
    # s=3
    gamma[3][1][1] = 1
    eta[3][1][0] = 1; eta[3][0][0] = -1
    # s=4
    gamma[4][0][0] = 1; gamma[4][0][1] = 1
    eta[4][1][1] = 1
    # s=5
    gamma[5][1][0] = 1; gamma[5][0][0] = -1
    eta[5][0][0] = 1; eta[5][0][1] = 1
    # s=6
    gamma[6][0][1] = 1; gamma[6][1][1] = -1
    eta[6][1][0] = 1; eta[6][1][1] = 1
    
    # c_ij[s] for block reconstruction
    c = [[0]*7 for _ in range(4)]
    # c[0][0] is C00, c[0][1] is C01, etc.
    # C00 = M0 + M3 - M4 + M6
    c[0][0] = 1; c[0][3] = 1; c[0][4] = -1; c[0][6] = 1
    # C01 = M2 + M4
    c[0][2] = 1; c[0][4] = 1
    # C10 = M1 + M3
    c[1][1] = 1; c[1][3] = 1
    # C11 = M0 - M1 + M2 + M5
    c[1][0] = 1; c[1][1] = -1; c[1][2] = 1; c[1][5] = 1
    
    # W_t matrices for 2x2 reconstruction
    W_mat = [[[0]*2 for _ in range(2)] for _ in range(7)]
    # W0
    W_mat[0][0][0] = 1; W_mat[0][1][1] = 1
    # W1
    W_mat[1][1][0] = 1; W_mat[1][1][1] = -1
    # W2
    W_mat[2][0][1] = 1; W_mat[2][1][1] = 1
    # W3
    W_mat[3][0][0] = 1; W_mat[3][1][0] = 1
    # W4
    W_mat[4][0][0] = -1; W_mat[4][0][1] = 1; W_mat[4][1][1] = 1
    # W5
    W_mat[5][1][1] = 1
    # W6
    W_mat[6][0][0] = 1
    
    # Build rank-49 decomposition for 4x4x4
    U44 = [[0]*49 for _ in range(16)]
    V44 = [[0]*49 for _ in range(16)]
    W44 = [[0]*49 for _ in range(16)]
    
    for s in range(7):
        for t in range(7):
            r = s * 7 + t
            # U vector
            for a in range(2):
                for b in range(2):
                    if gamma[s][a][b] != 0:
                        for p in range(2):
                            for q in range(2):
                                if beta[t][p][q] != 0:
                                    i = 2 * a + p
                                    j = 2 * b + q
                                    U44[i * 4 + j][r] = beta[t][p][q] * gamma[s][a][b]
            # V vector
            for c_ in range(2):
                for d in range(2):
                    if eta[s][c_][d] != 0:
                        for p in range(2):
                            for q in range(2):
                                if delta[t][p][q] != 0:
                                    j = 2 * c_ + p
                                    k = 2 * d + q
                                    V44[j * 4 + k][r] = delta[t][p][q] * eta[s][c_][d]
            # W vector
            for i_block in range(2):
                for k_block in range(2):
                    if c[i_block][k_block] != 0: # Wait, c is indexed by [i_block][k_block]?
                        # My c array is c[i][j] where i,j are 0,1.
                        # c[0] is for C00, C01? No.
                        # C is 2x2 blocks. C[0][0], C[0][1], C[1][0], C[1][1].
                        # My c array: c[0] is C00 coefficients? No.
                        # I defined c[0][0]=1, c[0][3]=1...
                        # This is c[block_row][block_col] = [coeffs for s=0..6]
                        # So c[i_block][k_block] is a list of 7 coeffs.
                        pass
            
            # Correct W vector construction
            for i_block in range(2):
                for k_block in range(2):
                    coeff_s = c[i_block][k_block][s]
                    if coeff_s != 0:
                        for p in range(2):
                            for q in range(2):
                                if W_mat[t][p][q] != 0:
                                    i = 2 * i_block + p
                                    k = 2 * k_block + q
                                    W44[k * 4 + i][r] = coeff_s * W_mat[t][p][q]

    # Build rank-4 decomposition for 4x4x1
    U41 = [[0]*4 for _ in range(16)]
    V41 = [[0]*4 for _ in range(4)]
    W41 = [[0]*4 for _ in range(4)]
    
    for j in range(4):
        r = j
        for i in range(4):
            U41[i * 4 + j][r] = 1
        V41[j][r] = 1
        for i in range(4):
            W41[i][r] = 1

    # Embed into 4x4x9
    U = [[0]*total_rank for _ in range(16)]
    V = [[0]*total_rank for _ in range(36)]
    W = [[0]*total_rank for _ in range(36)]
    
    # First 4x4x4
    offset = 0
    for i in range(16):
        for r in range(49):
            U[i][offset + r] = U44[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j * 9 + k][offset + r] = V44[j * 4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[k * 4 + i][offset + r] = W44[k * 4 + i][r]
                
    # Second 4x4x4
    offset = 49
    for i in range(16):
        for r in range(49):
            U[i][offset + r] = U44[i][r]
    for j in range(4):
        for k in range(4):
            for r in range(49):
                V[j * 9 + (k + 4)][offset + r] = V44[j * 4 + k][r]
    for k in range(4):
        for i in range(4):
            for r in range(49):
                W[(k + 4) * 4 + i][offset + r] = W44[k * 4 + i][r]
                
    # 4x4x1
    offset = 98
    for i in range(16):
        for r in range(4):
            U[i][offset + r] = U41[i][r]
    for j in range(4):
        for r in range(4):
            V[j * 9 + 8][offset + r] = V41[j][r]
    for i in range(4):
        for r in range(4):
            W[8 * 4 + i][offset + r] = W41[i][r]
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
