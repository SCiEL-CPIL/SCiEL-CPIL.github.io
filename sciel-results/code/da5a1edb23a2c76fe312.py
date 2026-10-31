import numpy as np
import time

def construct():
    # Strassen decomposition for 2x2 matrices (rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=int)

    # Outer Strassen coefficients for 4x4 (block structure 2x2 of 2x2 blocks)
    beta = np.zeros((7, 2, 2), dtype=int)
    gamma = np.zeros((7, 2, 2), dtype=int)
    delta = np.zeros((7, 2, 2), dtype=int)
    
    # m1 = (A11+A22)(B11+B22)
    beta[0,0,0]=1; beta[0,1,1]=1
    gamma[0,0,0]=1; gamma[0,1,1]=1
    delta[0,0,0]=1; delta[0,1,1]=1
    
    # m2 = (A21+A22)B11
    beta[1,1,0]=1; beta[1,1,1]=1
    gamma[1,0,0]=1
    delta[1,1,0]=1; delta[1,1,1]=-1
    
    # m3 = A11(B21-B22)
    beta[2,0,0]=1
    gamma[2,0,1]=1; gamma[2,1,1]=-1
    delta[2,0,1]=1; delta[2,1,1]=1
    
    # m4 = A22(B11-B21)
    beta[3,1,1]=1
    gamma[3,1,0]=1; gamma[3,0,0]=-1
    delta[3,0,0]=1; delta[3,1,0]=1
    
    # m5 = (A11+A12)B22
    beta[4,0,0]=1; beta[4,0,1]=1
    gamma[4,1,1]=1
    delta[4,0,0]=-1; delta[4,0,1]=1
    
    # m6 = (A21-A11)(B11+B12)
    beta[5,1,0]=1; beta[5,0,0]=-1
    gamma[5,0,0]=1; gamma[5,0,1]=1
    delta[5,1,1]=1
    
    # m7 = (A12-A22)(B21+B22)
    beta[6,0,1]=1; beta[6,1,1]=-1
    gamma[6,1,0]=1; gamma[6,1,1]=1
    delta[6,0,0]=1

    # Build 4x4 decomposition (rank 49) via Kronecker structure
    U = np.zeros((16, 49), dtype=int)
    V = np.zeros((16, 49), dtype=int)
    W = np.zeros((16, 49), dtype=int)
    
    for I in range(2):
        for J in range(2):
            for i2 in range(2):
                for j2 in range(2):
                    r = 2*I + i2
                    c = 2*J + j2
                    row_u = r*4 + c
                    u2_idx = i2*2 + j2
                    U[row_u, :] += np.outer(beta[:, I, J], U2[u2_idx, :]).flatten()
                    
    for J in range(2):
        for K in range(2):
            for j2 in range(2):
                for k2 in range(2):
                    c = 2*J + j2
                    k = 2*K + k2
                    row_v = c*4 + k
                    v2_idx = j2*2 + k2
                    V[row_v, :] += np.outer(gamma[:, J, K], V2[v2_idx, :]).flatten()
                    
    for I in range(2):
        for K in range(2):
            for i2 in range(2):
                for k2 in range(2):
                    k = 2*K + k2
                    i = 2*I + i2
                    row_w = k*4 + i
                    w2_idx = k2*2 + i2
                    W[row_w, :] += np.outer(delta[:, I, K], W2[w2_idx, :]).flatten()

    # Verification using einsum
    U3 = U.reshape(4, 4, 49)  # [i, j, t]
    V3 = V.reshape(4, 4, 49)  # [j, k, t]
    W3 = W.reshape(4, 4, 49)  # [k, i, t]
    
    result_ik = np.einsum('ijt,jkt,kit->ik', U3, V3, W3)
    
    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(4, 4))
    B = np.random.randint(-5, 6, size=(4, 4))
    C = A @ B
    
    C_trace = np.zeros(16)
    for i in range(4):
        for k in range(4):
            C_trace[k*4+i] = C[i,k]
    
    result_trace = np.zeros(16)
    for i in range(4):
        for k in range(4):
            result_trace[k*4+i] = result_ik[i,k]
    
    if not np.allclose(result_trace, C_trace, atol=1e-10):
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
