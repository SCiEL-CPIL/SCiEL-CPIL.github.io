import numpy as np

def _strassen_2x2x2_factors():
    U_terms = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [1, 0, -1, 0], [0, 1, 0, -1]
    ]
    V_terms = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 0, 1, -1],
        [-1, 1, 0, 0], [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    W_terms = [
        [1, 0, 0, 1], [0, 0, 1, -1], [0, 1, 0, 1],
        [1, 0, 1, 0], [-1, 1, 0, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]
    return np.array(U_terms), np.array(V_terms), np.array(W_terms)

def _permute_indices(kron_vec, size=4):
    # kron_vec is length 16
    # K = i1*8 + j1*4 + i2*2 + j2
    # G = i1*8 + i2*4 + j1*2 + j2
    permuted = np.zeros(16)
    for K in range(16):
        i1 = K // 8
        rem = K % 8
        j1 = rem // 4
        rem = rem % 4
        i2 = rem // 2
        j2 = rem % 2
        G = i1 * 8 + i2 * 4 + j1 * 2 + j2
        permuted[G] = kron_vec[K]
    return permuted

def construct():
    n, m, p = 4, 4, 8
    U2, V2, W2 = _strassen_2x2x2_factors()
    
    R4 = 49
    U4 = np.zeros((16, R4))
    V4 = np.zeros((16, R4))
    W4 = np.zeros((16, R4))
    
    for r1 in range(7):
        for r2 in range(7):
            r = r1 * 7 + r2
            u_kron = np.kron(U2[:, r1], U2[:, r2])
            v_kron = np.kron(V2[:, r1], V2[:, r2])
            w_kron = np.kron(W2[:, r1], W2[:, r2])
            
            U4[:, r] = _permute_indices(u_kron)
            V4[:, r] = _permute_indices(v_kron)
            W4[:, r] = _permute_indices(w_kron)
            
    R = 2 * R4
    U = np.zeros((16, R))
    V = np.zeros((32, R))
    W = np.zeros((32, R))
    
    U[:, :R4] = U4
    U[:, R4:] = U4
    
    for j in range(4):
        for k in range(4):
            V[j*8 + k, :R4] = V4[j*4 + k, :]
            V[j*8 + 4 + k, R4:] = V4[j*4 + k, :]
            
    for k in range(4):
        for i in range(4):
            W[k*4 + i, :R4] = W4[k*4 + i, :]
            W[(4+k)*4 + i, R4:] = W4[k*4 + i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
