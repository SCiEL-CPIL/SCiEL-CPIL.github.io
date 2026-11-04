import numpy as np
import time

def construct():
    n, m, p = 4, 4, 7
    t0 = time.time()
    
    # Rank 49 for 4x4x4 (Strassen)
    U444, V444, W444 = _strassen_4x4x4()
    
    # Rank 44 for 4x4x3 (4x 2x2x3)
    U443, V443, W443 = _decomp_4x4x3()
    
    # Total rank 93
    rank = 49 + 44
    
    U = np.zeros((16, rank))
    V = np.zeros((28, rank))
    W = np.zeros((28, rank))
    
    # Fill 4x4x4 part
    U[:, 0:49] = U444
    # V444 is 16x49. We need to map to 28x49.
    # B is 4x7. First 4 cols are 4x4.
    # V444[j*4+k, r] corresponds to B[j,k] for k in 0..3.
    # In full V, B[j,k] is at V[j*7+k, r].
    for j in range(4):
        for k in range(4):
            V[j*7+k, 0:49] = V444[j*4+k, 0:49]
            
    # W444 is 16x49.
    # C is 4x7. First 4 cols are 4x4.
    # W444[k*4+i, r] corresponds to C[i,k] for k in 0..3.
    # In full W, C[i,k] is at W[k*4+i, r].
    for k in range(4):
        for i in range(4):
            W[k*4+i, 0:49] = W444[k*4+i, 0:49]
            
    # Fill 4x4x3 part
    U[:, 49:93] = U443
    # V443 is 12x44.
    # B is 4x3 (cols 4..6).
    # V443[j*3+k, r] corresponds to B[j,k] for k in 0..2.
    # In full V, B[j, 4+k] is at V[j*7+4+k, r].
    for j in range(4):
        for k in range(3):
            V[j*7+4+k, 49:93] = V443[j*3+k, 0:44]
            
    # W443 is 12x44.
    # C is 4x3 (cols 4..6).
    # W443[k*4+i, r] corresponds to C[i,k] for k in 0..2.
    # In full W, C[i, 4+k] is at W[(4+k)*4+i, r].
    for k in range(3):
        for i in range(4):
            W[(4+k)*4+i, 49:93] = W443[k*4+i, 0:44]
            
    if time.time() - t0 > 120:
        # Fallback
        rank = 112
        U = np.zeros((16, rank))
        V = np.zeros((28, rank))
        W = np.zeros((28, rank))
        r = 0
        for i in range(4):
            for j in range(4):
                for k in range(7):
                    U[i*4+j, r] = 1
                    V[j*7+k, r] = 1
                    W[k*4+i, r] = 1
                    r += 1
                    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

def _strassen_2x2x2():
    U = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V = np.array([
        [1, 1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 1, -1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ])
    W = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1, -1]
    ])
    return U, V, W

def _strassen_4x4x4():
    U_s, V_s, W_s = _strassen_2x2x2()
    U = np.zeros((16, 49))
    V = np.zeros((16, 49))
    W = np.zeros((16, 49))
    for r1 in range(7):
        for r2 in range(7):
            r = r1*7 + r2
            for i1 in range(2):
                for j1 in range(2):
                    for i2 in range(2):
                        for j2 in range(2):
                            i = i1*2 + i2
                            j = j1*2 + j2
                            U[i*4+j, r] = U_s[i1*2+j1, r1] * U_s[i2*2+j2, r2]
            for j1 in range(2):
                for k1 in range(2):
                    for j2 in range(2):
                        for k2 in range(2):
                            j = j1*2 + j2
                            k = k1*2 + k2
                            V[j*4+k, r] = V_s[j1*2+k1, r1] * V_s[j2*2+k2, r2]
            for k1 in range(2):
                for i1 in range(2):
                    for k2 in range(2):
                        for i2 in range(2):
                            k = k1*2 + k2
                            i = i1*2 + i2
                            W[k*4+i, r] = W_s[k1*2+i1, r1] * W_s[k2*2+i2, r2]
    return U, V, W

def _decomp_4x4x3():
    # 4x 2x2x3
    U_223, V_223, W_223 = _strassen_2x2x3()
    U = np.zeros((16, 44))
    V = np.zeros((12, 44))
    W = np.zeros((12, 44))
    
    for b in range(4):
        i1 = b // 2
        j1 = b % 2
        u_off = b * 11
        v_off = b * 11
        w_off = b * 11
        
        for i2 in range(2):
            for j2 in range(2):
                i = i1*2 + i2
                j = j1*2 + j2
                U[i*4+j, u_off:u_off+11] = U_223[i2*2+j2, :]
                
        for k in range(3):
            for j2 in range(2):
                j = j1*2 + j2
                V[j*3+k, v_off:v_off+11] = V_223[k*2+j2, :]
                
        for k in range(3):
            for i2 in range(2):
                i = i1*2 + i2
                W[k*4+i, w_off:w_off+11] = W_223[k*2+i2, :]
                
    return U, V, W

def _strassen_2x2x3():
    U_s, V_s, W_s = _strassen_2x2x2()
    # Schoolbook 2x2x1
    U_sc = np.zeros((4, 4))
    V_sc = np.zeros((2, 4))
    W_sc = np.zeros((2, 4))
    for i in range(2):
        for j in range(2):
            r = i*2 + j
            U_sc[i*2+j, r] = 1
            V_sc[j, r] = 1
            W_sc[i, r] = 1
            
    U = np.hstack([U_s, U_sc])
    V = np.zeros((6, 11))
    V[0:4, 0:7] = V_s
    V[4:6, 7:11] = V_sc
    W = np.zeros((6, 11))
    W[0:4, 0:7] = W_s
    W[4:6, 7:11] = W_sc
    return U, V, W
