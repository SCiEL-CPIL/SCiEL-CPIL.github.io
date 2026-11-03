import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # 1. Construct 2x2 Strassen coefficients
    # U22: 4x7, V22: 4x7, W22: 4x7
    U22 = np.zeros((4, 7), dtype=int)
    V22 = np.zeros((4, 7), dtype=int)
    W22 = np.zeros((4, 7), dtype=int)
    
    # M1: (A00+A11)(B00+B11)
    U22[0, 0] = 1; U22[3, 0] = 1
    V22[0, 0] = 1; V22[3, 0] = 1
    W22[0, 0] = 1; W22[3, 0] = 1
    
    # M2: (A10+A11)B00
    U22[2, 1] = 1; U22[3, 1] = 1
    V22[0, 1] = 1
    W22[2, 1] = 1; W22[3, 1] = -1
    
    # M3: A00(B01-B11)
    U22[0, 2] = 1
    V22[1, 2] = 1; V22[3, 2] = -1
    W22[1, 2] = 1; W22[3, 2] = 1
    
    # M4: A11(B10-B00)
    U22[3, 3] = 1
    V22[2, 3] = 1; V22[0, 3] = -1
    W22[0, 3] = 1; W22[2, 3] = 1
    
    # M5: (A00+A01)B11
    U22[0, 4] = 1; U22[1, 4] = 1
    V22[3, 4] = 1
    W22[0, 4] = -1; W22[1, 4] = 1
    
    # M6: (A10-A00)(B00+B01)
    U22[2, 5] = 1; U22[0, 5] = -1
    V22[0, 5] = 1; V22[1, 5] = 1
    W22[3, 5] = 1
    
    # M7: (A01-A11)(B10+B11)
    U22[1, 6] = 1; U22[3, 6] = -1
    V22[2, 6] = 1; V22[3, 6] = 1
    W22[0, 6] = 1; W22[3, 6] = 1

    # 2. Define block operation coefficients alpha, beta, gamma
    # alpha[b, r, c] for A blocks
    # beta[b, c, d] for B blocks
    # gamma[b, r, c] for C blocks
    # b in 0..6
    alpha = np.zeros((7, 2, 2), dtype=int)
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    
    # M1
    alpha[0, 0, 0] = 1; alpha[0, 1, 1] = 1
    beta[0, 0, 0] = 1; beta[0, 1, 1] = 1
    gamma[0, 0, 0] = 1; gamma[0, 1, 1] = 1
    
    # M2
    alpha[1, 1, 0] = 1; alpha[1, 1, 1] = 1
    beta[1, 0, 0] = 1
    gamma[1, 1, 0] = 1; gamma[1, 1, 1] = -1
    
    # M3
    alpha[2, 0, 0] = 1
    beta[2, 0, 1] = 1; beta[2, 1, 1] = -1
    gamma[2, 0, 1] = 1; gamma[2, 1, 1] = 1
    
    # M4
    alpha[3, 1, 1] = 1
    beta[3, 1, 0] = 1; beta[3, 0, 0] = -1
    gamma[3, 0, 0] = 1; gamma[3, 1, 0] = 1
    
    # M5
    alpha[4, 0, 0] = 1; alpha[4, 0, 1] = 1
    beta[4, 1, 1] = 1
    gamma[4, 0, 0] = -1; gamma[4, 0, 1] = 1
    
    # M6
    alpha[5, 1, 0] = 1; alpha[5, 0, 0] = -1
    beta[5, 0, 0] = 1; beta[5, 0, 1] = 1
    gamma[5, 1, 1] = 1
    
    # M7
    alpha[6, 0, 1] = 1; alpha[6, 1, 1] = -1
    beta[6, 1, 0] = 1; beta[6, 1, 1] = 1
    gamma[6, 0, 0] = 1; gamma[6, 1, 1] = 1

    # 3. Construct full U, V, W
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = np.zeros((16, total_rank), dtype=int)
    V = np.zeros((20, total_rank), dtype=int)
    W = np.zeros((20, total_rank), dtype=int)
    
    col = 0
    for b in range(7):
        for s in range(7):
            # U coefficients
            for I in range(4):
                for J in range(4):
                    r = I // 2
                    i = I % 2
                    c = J // 2
                    j = J % 2
                    coeff = alpha[b, r, c] * U22[i*2+j, s]
                    if coeff != 0:
                        U[I*4+J, col] += coeff
            
            # V coefficients
            for J in range(4):
                for K in range(4):
                    c = J // 2
                    j = J % 2
                    d = K // 2
                    k = K % 2
                    coeff = beta[b, c, d] * V22[j*2+k, s]
                    if coeff != 0:
                        V[J*5+K, col] += coeff
            
            # W coefficients
            for I in range(4):
                for K in range(4):
                    r = I // 2
                    i = I % 2
                    c = K // 2
                    k = K % 2
                    coeff = gamma[b, r, c] * W22[k*2+i, s]
                    if coeff != 0:
                        W[K*4+I, col] += coeff
            
            col += 1
    
    # 4. Add 4x4x1 part
    for I in range(4):
        for J in range(4):
            U[I*4+J, col] = 1
            V[J*5+4, col] = 1
            W[4*4+I, col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
