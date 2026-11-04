import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)
    
    # M1
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1
    
    # M2
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1
    
    # M3
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = -1
    
    # M4
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[1, 3] = -1
    W2[0, 3] = -1; W2[2, 3] = 1
    
    # M5
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = 1
    
    # M6
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1
    
    # M7
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = -1; W2[1, 6] = 1
    
    # 4x4x4
    U4 = np.zeros((16, 49), dtype=int)
    V4 = np.zeros((16, 49), dtype=int)
    W4 = np.zeros((16, 49), dtype=int)
    
    for i in range(2):
        for j in range(2):
            for k in range(2):
                for ip in range(2):
                    for jp in range(2):
                        for kp in range(2):
                            I = 2*i + ip
                            J = 2*j + jp
                            K = 2*k + kp
                            for r1 in range(7):
                                for r2 in range(7):
                                    r = r1 * 7 + r2
                                    U4[I*4 + J, r] = U2[i*2+j, r1] * U2[ip*2+jp, r2]
                                    V4[J*4 + K, r] = V2[j*2+k, r1] * V2[jp*2+kp, r2]
                                    W4[K*4 + I, r] = W2[k*2+i, r1] * W2[kp*2+ip, r2]
                                    
    # 4x4x8
    U = np.zeros((16, 98), dtype=int)
    V = np.zeros((32, 98), dtype=int)
    W = np.zeros((32, 98), dtype=int)
    
    U[:, 0:49] = U4
    U[:, 49:98] = U4
    
    for j in range(4):
        # First half (B_L)
        V[j*8 : j*8+4, 0:49] = V4[j*4 : j*4+4, 0:49]
        # Second half (B_R)
        V[j*8+4 : j*8+8, 49:98] = V4[j*4 : j*4+4, 0:49]
        
    W[0:16, 0:49] = W4
    W[16:32, 49:98] = W4
    
    return {"U": U, "V": V, "W": W}
