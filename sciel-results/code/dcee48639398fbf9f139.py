import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = 4, 4, 4

    # Standard rank-49 decomposition via Kronecker product of two rank-7 Strassen algorithms
    # This gives rank = 7 * 7 = 49
    rank = 49

    # Build the 4x4x4 tensor decomposition using Strassen at two levels
    # Level 1: 2x2 block decomposition with rank 7 (Strassen)
    # Level 2: Within each block, 2x2 scalar decomposition with rank 7 (Strassen)
    
    # Strassen 2x2 coefficients
    # For A (2x2), B (2x2) -> C = A*B
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # Strassen's 7 multiplications:
    # m1 = (a11+a22)*(b11+b22)
    # m2 = (a21+a22)*b11
    # m3 = a11*(b12-b22)
    # m4 = a22*(b21-b11)
    # m5 = (a11+a12)*b22
    # m6 = (a21-a11)*(b11+b12)
    # m7 = (a12-a22)*(b21+b22)
    
    # C11 = m1 + m4 - m5 + m7
    # C12 = m3 + m5
    # C21 = m2 + m4
    # C22 = m1 - m2 + m3 + m6
    
    # Build U2, V2, W2 for 2x2 Strassen (each is 4x7)
    # U2[i,j] for A element at position (i,j), V2[j,k] for B element at (j,k), W2[i,k] for C element at (i,k)
    
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # Flatten 2x2 matrices: index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    # A: a11=0, a12=1, a21=2, a22=3
    # B: b11=0, b12=1, b21=2, b22=3
    # C: c11=0, c12=1, c21=2, c22=3
    
    # m1 = (a11+a22)*(b11+b22)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[3][0] = 1  # c11 += m1, c22 += m1
    
    # m2 = (a21+a22)*b11
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1  # c21 += m2
    W2[3][1] = -1  # c22 -= m2
    
    # m3 = a11*(b12-b22)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1  # c12 += m3
    W2[3][2] = 1  # c22 += m3
    
    # m4 = a22*(b21-b11)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1  # c11 += m4
    W2[2][3] = 1  # c21 += m4
    
    # m5 = (a11+a12)*b22
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1  # c11 -= m5
    W2[1][4] = 1  # c12 += m5
    
    # m6 = (a21-a11)*(b11+b12)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1  # c22 += m6
    
    # m7 = (a12-a22)*(b21+b22)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1  # c11 += m7
    
    # Build 4x4 Strassen coefficients for block level
    # Block indices: I,J,K in {0,1}
    # A block (I,J), B block (J,K), C block (I,K)
    # Flatten 2x2 blocks: index 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    
    # beta[s][I][J] for A[I,J]
    # gamma[s][J][K] for B[J,K]
    # delta[s][I][K] for C[I,K]
    
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)
    
    # s=0: m1 = (A00+A11)*(B00+B11)
    beta[0][0][0] = 1; beta[0][1][1] = 1
    gamma[0][0][0] = 1; gamma[0][1][1] = 1
    delta[0][0][0] = 1; delta[0][1][1] = 1
    
    # s=1: m2 = (A10+A11)*B00
    beta[1][1][0] = 1; beta[1][1][1] = 1
    gamma[1][0][0] = 1
    delta[1][1][0] = 1; delta[1][1][1] = -1
    
    # s=2: m3 = A00*(B01-B11)
    beta[2][0][0] = 1
    gamma[2][0][1] = 1; gamma[2][1][1] = -1
    delta[2][0][1] = 1; delta[2][1][1] = 1
    
    # s=3: m4 = A11*(B10-B00)
    beta[3][1][1] = 1
    gamma[3][1][0] = 1; gamma[3][0][0] = -1
    delta[3][0][0] = 1; delta[3][1][0] = 1
    
    # s=4: m5 = (A00+A01)*B11
    beta[4][0][0] = 1; beta[4][0][1] = 1
    gamma[4][1][1] = 1
    delta[4][0][0] = -1; delta[4][0][1] = 1
    
    # s=5: m6 = (A10-A00)*(B00+B01)
    beta[5][1][0] = 1; beta[5][0][0] = -1
    gamma[5][0][0] = 1; gamma[5][0][1] = 1
    delta[5][1][1] = 1
    
    # s=6: m7 = (A01-A11)*(B10+B11)
    beta[6][0][1] = 1; beta[6][1][1] = -1
    gamma[6][1][0] = 1; gamma[6][1][1] = 1
    delta[6][0][0] = 1
    
    # Build U, V, W using Kronecker-like structure
    # U has shape (16, 49), V has shape (16, 49), W has shape (16, 49)
    # Row index for U: i*4 + j where i,j in 0..3
    # Block: I = i//2, local: i' = i%2
    # Within block: local index = i'*2 + j'
    # Column: s*7 + t where s in 0..6 (block level), t in 0..6 (scalar level)
    
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for s in range(7):
        for t in range(7):
            c = s * 7 + t
            
            # Fill U
            for i in range(4):
                for j in range(4):
                    I = i // 2
                    ip = i % 2
                    J = j // 2
                    jp = j % 2
                    row_u = i * 4 + j
                    local_u = ip * 2 + jp
                    coeff = beta[s][I][J] * U2[local_u][t]
                    if coeff != 0:
                        U[row_u][c] = coeff
            
            # Fill V
            for j in range(4):
                for k in range(4):
                    J = j // 2
                    jp = j % 2
                    K = k // 2
                    kp = k % 2
                    row_v = j * 4 + k
                    local_v = jp * 2 + kp
                    coeff = gamma[s][J][K] * V2[local_v][t]
                    if coeff != 0:
                        V[row_v][c] = coeff
            
            # Fill W (trace convention: W[k*n+i, r])
            for k in range(4):
                for i in range(4):
                    K = k // 2
                    kp = k % 2
                    I = i // 2
                    ip = i % 2
                    row_w = k * 4 + i  # Trace convention
                    local_w = kp * 2 + ip
                    coeff = delta[s][I][K] * W2[local_w][t]
                    if coeff != 0:
                        W[row_w][c] = coeff
            
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
