import numpy as np

def construct():
    # Strassen 2x2x2 decomposition vectors
    u1 = np.array([1, 0, 0, 1])
    v1 = np.array([1, 0, 0, 1])
    w1 = np.array([1, 0, 0, 1])
    
    u2 = np.array([0, 0, 1, 1])
    v2 = np.array([1, 0, 0, 0])
    w2 = np.array([0, 0, 1, 0])
    
    u3 = np.array([1, 0, 0, 0])
    v3 = np.array([0, 1, 0, -1])
    w3 = np.array([0, 1, 0, 0])
    
    u4 = np.array([0, 0, 0, 1])
    v4 = np.array([0, 0, 1, -1])
    w4 = np.array([1, 0, 1, 0])
    
    u5 = np.array([1, 1, 0, 0])
    v5 = np.array([0, 0, 0, 1])
    w5 = np.array([-1, 1, 0, 0])
    
    u6 = np.array([-1, 0, 1, 0])
    v6 = np.array([1, 1, 0, 0])
    w6 = np.array([0, 0, 0, 1])
    
    u7 = np.array([0, 1, 0, -1])
    v7 = np.array([0, 0, 1, 1])
    w7 = np.array([1, 0, 0, 1])
    
    u_vecs = [u1, u2, u3, u4, u5, u6, u7]
    v_vecs = [v1, v2, v3, v4, v5, v6, v7]
    w_vecs = [w1, w2, w3, w4, w5, w6, w7]
    
    # Construct 4x4x4 Strassen decomposition
    u_444 = []
    v_444 = []
    w_444 = []
    for i in range(7):
        for j in range(7):
            u_444.append(np.kron(u_vecs[i], u_vecs[j]))
            v_444.append(np.kron(v_vecs[i], v_vecs[j]))
            w_444.append(np.kron(w_vecs[i], w_vecs[j]))
            
    # Construct 4x4x1 schoolbook decomposition
    u_441 = []
    v_441 = []
    w_441 = []
    for i in range(4):
        for j in range(4):
            u = np.zeros(16)
            u[i*4+j] = 1
            v = np.zeros(4)
            v[j] = 1
            w = np.zeros(4)
            w[i] = 1
            u_441.append(u)
            v_441.append(v)
            w_441.append(w)
            
    # Combine for 4x4x9
    # First 4x4x4: cols 0-3 of B and C
    # Second 4x4x4: cols 4-7 of B and C
    # 4x4x1: col 8 of B and C
    
    U_final = np.zeros((16, 114))
    V_final = np.zeros((36, 114))
    W_final = np.zeros((36, 114))
    
    # First 4x4x4
    for idx in range(49):
        U_final[:, idx] = u_444[idx]
        # V indices: j*9 + k, where k is 0..3
        # v_444[idx] is length 16, corresponding to j*4+k for j=0..3, k=0..3
        # We need to map this to j*9+k
        v_temp = np.zeros(36)
        for j in range(4):
            for k in range(4):
                v_temp[j*9 + k] = v_444[idx][j*4 + k]
        V_final[:, idx] = v_temp
        # W indices: k*4 + i, where k is 0..3
        # w_444[idx] is length 16, corresponding to k*4+i for k=0..3, i=0..3
        # We need to map this to k*4+i
        w_temp = np.zeros(36)
        for k in range(4):
            for i in range(4):
                w_temp[k*4 + i] = w_444[idx][k*4 + i]
        W_final[:, idx] = w_temp
        
    # Second 4x4x4
    for idx in range(49):
        U_final[:, 49 + idx] = u_444[idx]
        # V indices: j*9 + (k+4), where k is 0..3
        v_temp = np.zeros(36)
        for j in range(4):
            for k in range(4):
                v_temp[j*9 + (k+4)] = v_444[idx][j*4 + k]
        V_final[:, 49 + idx] = v_temp
        # W indices: (k+4)*4 + i, where k is 0..3
        w_temp = np.zeros(36)
        for k in range(4):
            for i in range(4):
                w_temp[(k+4)*4 + i] = w_444[idx][k*4 + i]
        W_final[:, 49 + idx] = w_temp
        
    # 4x4x1
    for idx in range(16):
        U_final[:, 98 + idx] = u_441[idx]
        # V indices: j*9 + 8, where j is 0..3
        v_temp = np.zeros(36)
        for j in range(4):
            v_temp[j*9 + 8] = v_441[idx][j]
        V_final[:, 98 + idx] = v_temp
        # W indices: 8*4 + i, where i is 0..3
        w_temp = np.zeros(36)
        for i in range(4):
            w_temp[8*4 + i] = w_441[idx][i]
        W_final[:, 98 + idx] = w_temp
        
    return {"U": U_final, "V": V_final, "W": W_final}
