import time

def construct():
    start_time = time.time()
    deadline = start_time + 10.0
    
    n, m, p = 4, 4, 4
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 2x2 coefficients
    # beta[s][I][J]: coeff of A[I][J] in s-th outer mult
    # gamma[s][J][K]: coeff of B[J][K] in s-th outer mult  
    # delta[s][I][K]: coeff of C[I][K] in s-th outer mult
    
    beta = [[[0]*2 for _ in range(2)] for _ in range(7)]
    gamma = [[[0]*2 for _ in range(2)] for _ in range(7)]
    delta = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # s=0: (A00+A11)(B00+B11) -> C00+=, C11+=
    beta[0][0][0]=1; beta[0][1][1]=1
    gamma[0][0][0]=1; gamma[0][1][1]=1
    delta[0][0][0]=1; delta[0][1][1]=1
    
    # s=1: (A10+A11)B00 -> C10+=, C11-=
    beta[1][1][0]=1; beta[1][1][1]=1
    gamma[1][0][0]=1
    delta[1][1][0]=1; delta[1][1][1]=-1
    
    # s=2: A00(B01-B11) -> C01+=, C11+=
    beta[2][0][0]=1
    gamma[2][0][1]=1; gamma[2][1][1]=-1
    delta[2][0][1]=1; delta[2][1][1]=1
    
    # s=3: A11(B10-B00) -> C00+=, C10+=
    beta[3][1][1]=1
    gamma[3][1][0]=1; gamma[3][0][0]=-1
    delta[3][0][0]=1; delta[3][1][0]=1
    
    # s=4: (A00+A01)B11 -> C00-=, C01+=
    beta[4][0][0]=1; beta[4][0][1]=1
    gamma[4][1][1]=1
    delta[4][0][0]=-1; delta[4][0][1]=1
    
    # s=5: (A10-A00)(B00+B01) -> C11+=
    beta[5][1][0]=1; beta[5][0][0]=-1
    gamma[5][0][0]=1; gamma[5][0][1]=1
    delta[5][1][1]=1
    
    # s=6: (A01-A11)(B10+B11) -> C00+=
    beta[6][0][1]=1; beta[6][1][1]=-1
    gamma[6][1][0]=1; gamma[6][1][1]=1
    delta[6][0][0]=1
    
    # Inner Strassen 2x2 coefficients (same structure, applied to 2x2 blocks)
    U2 = [[0]*7 for _ in range(4)]  # U2[local_idx][t]
    V2 = [[0]*7 for _ in range(4)]  # V2[local_idx][t]
    W2 = [[0]*7 for _ in range(4)]  # W2[local_idx][t]
    
    # t=0
    U2[0][0]=1; U2[3][0]=1
    V2[0][0]=1; V2[3][0]=1
    W2[0][0]=1; W2[3][0]=1
    
    # t=1
    U2[2][1]=1; U2[3][1]=1
    V2[0][1]=1
    W2[1][1]=1; W2[3][1]=-1
    
    # t=2
    U2[0][2]=1
    V2[1][2]=1; V2[3][2]=-1
    W2[2][2]=1; W2[3][2]=1
    
    # t=3
    U2[3][3]=1
    V2[2][3]=1; V2[0][3]=-1
    W2[0][3]=1; W2[1][3]=1
    
    # t=4
    U2[0][4]=1; U2[1][4]=1
    V2[3][4]=1
    W2[0][4]=-1; W2[2][4]=1
    
    # t=5
    U2[2][5]=1; U2[0][5]=-1
    V2[0][5]=1; V2[1][5]=1
    W2[3][5]=1
    
    # t=6
    U2[1][6]=1; U2[3][6]=-1
    V2[2][6]=1; V2[3][6]=1
    W2[0][6]=1
    
    # Build the 49-rank decomposition via tensor product of outer and inner Strassen
    for s in range(7):
        for t in range(7):
            c = 7 * s + t
            
            # U[i*4+j, c] = beta[s][I][J] * U2[i2*2+j2][t]
            # where i = 2*I + i2, j = 2*J + j2
            for I in range(2):
                for J in range(2):
                    b_val = beta[s][I][J]
                    if b_val == 0:
                        continue
                    for i2 in range(2):
                        for j2 in range(2):
                            u2_val = U2[i2 * 2 + j2][t]
                            if u2_val == 0:
                                continue
                            i = 2 * I + i2
                            j = 2 * J + j2
                            U[i * 4 + j][c] = b_val * u2_val
            
            # V[j*4+k, c] = gamma[s][J][K] * V2[j2*2+k2][t]
            # where j = 2*J + j2, k = 2*K + k2
            for J in range(2):
                for K in range(2):
                    g_val = gamma[s][J][K]
                    if g_val == 0:
                        continue
                    for j2 in range(2):
                        for k2 in range(2):
                            v2_val = V2[j2 * 2 + k2][t]
                            if v2_val == 0:
                                continue
                            j = 2 * J + j2
                            k = 2 * K + k2
                            V[j * 4 + k][c] = g_val * v2_val
            
            # W[k*4+i, c] = delta[s][I][K] * W2[k2*2+i2][t]
            # where k = 2*K + k2, i = 2*I + i2
            # W index: k*n+i = k*4+i (trace convention)
            for I in range(2):
                for K in range(2):
                    d_val = delta[s][I][K]
                    if d_val == 0:
                        continue
                    for i2 in range(2):
                        for k2 in range(2):
                            w2_val = W2[k2 * 2 + i2][t]
                            if w2_val == 0:
                                continue
                            k = 2 * K + k2
                            i = 2 * I + i2
                            W[k * 4 + i][c] = d_val * w2_val
            
            if time.time() > deadline:
                return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
