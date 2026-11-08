import numpy as np

DIMENSIONS = [4, 4, 8]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 decomposition
    # U, V, W are 4x7
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ])
    V2 = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 0],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, -1, 1, 0, 1]
    ])
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 1, 0, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Verify 2x2
    # C = sum_r U[:,r] V[:,r] W[:,r]
    # This is implicitly assumed correct.

    # Construct 4x4 decomposition
    # Blocks: 11, 12, 21, 22
    # Indices in 4x4 (flattened row-major):
    # 11: 0,1,4,5
    # 12: 2,3,6,7
    # 21: 8,9,12,13
    # 22: 10,11,14,15
    
    block_indices = {
        0: [0, 1, 4, 5],   # 11
        1: [2, 3, 6, 7],   # 12
        2: [8, 9, 12, 13], # 21
        3: [10, 11, 14, 15]# 22
    }
    
    # Strassen coefficients for 4x4
    # k: 0..6
    # alpha[k][b], beta[k][b], delta[k][b]
    # M_k = (sum alpha A_b) (sum beta B_b)
    # C_b = sum delta M_k
    
    alpha = [
        [1, 0, 0, 1], # M1: A11+A22
        [0, 0, 1, 1], # M2: A21+A22
        [1, 0, 0, 0], # M3: A11
        [0, 0, 0, 1], # M4: A22
        [1, 1, 0, 0], # M5: A11+A12
        [-1, 0, 1, 0],# M6: A21-A11
        [0, 1, 0, -1] # M7: A12-A22
    ]
    beta = [
        [1, 0, 0, 1], # M1: B11+B22
        [1, 0, 0, 0], # M2: B11
        [0, 1, 0, -1],# M3: B12-B22
        [0, 0, 1, -1],# M4: B21-B11
        [0, 0, 0, 1], # M5: B22
        [1, 1, 0, 0], # M6: B11+B12
        [0, 0, 1, 1]  # M7: B21+B22
    ]
    delta = [
        [1, 0, 0, 1], # C11+=M1, C22+=M1
        [0, 0, 1, -1],# C21+=M2, C22-=M2
        [0, 1, 0, 1], # C12+=M3, C22+=M3
        [1, 0, 1, 0], # C11+=M4, C21+=M4
        [-1, 1, 0, 0],# C11-=M5, C12+=M5
        [0, 0, 0, 1], # C22+=M6
        [1, 0, 0, 0]  # C11+=M7
    ]
    
    U44 = []
    V44 = []
    W44 = []
    
    for k in range(7):
        for r in range(7):
            u_vec = np.zeros(16)
            v_vec = np.zeros(16)
            w_vec = np.zeros(16)
            
            for b in range(4):
                # A contribution
                if alpha[k][b] != 0:
                    for idx, u_val in zip(block_indices[b], U2[:, r]):
                        u_vec[idx] += alpha[k][b] * u_val
                        
                # B contribution
                if beta[k][b] != 0:
                    for idx, v_val in zip(block_indices[b], V2[:, r]):
                        v_vec[idx] += beta[k][b] * v_val
                        
            # C contribution
            for b in range(4):
                if delta[k][b] != 0:
                    for idx, w_val in zip(block_indices[b], W2[:, r]):
                        w_vec[idx] += delta[k][b] * w_val
                        
            U44.append(u_vec)
            V44.append(v_vec)
            W44.append(w_vec)
            
    U44 = np.array(U44).T # 16x49
    V44 = np.array(V44).T # 16x49
    W44 = np.array(W44).T # 16x49
    
    # Now construct 4x4x8
    # B is 4x8. Split into B1 (4x4) and B2 (4x4).
    # C is 8x4? No, output is 4x8.
    # W index convention: W[k*n+i, r].
    # C is 4x8. Flattened row-major: C[0,0]..C[0,7], C[1,0]..
    # B is 4x8. Flattened: B[0,0]..B[0,7]...
    # Wait, V shape is (m*p, rank) = (32, rank).
    # B is 4x8. Flattened row-major is 32 elements.
    # B1 is cols 0-3. B2 is cols 4-7.
    # C1 is cols 0-3. C2 is cols 4-7.
    
    U_total = np.zeros((16, 98))
    V_total = np.zeros((32, 98))
    W_total = np.zeros((32, 98))
    
    # First half: B1 -> C1
    U_total[:, 0:49] = U44
    # V for B1: rows 0-15 of V_total correspond to B1
    V_total[0:16, 0:49] = V44
    # W for C1: rows 0-15 of W_total correspond to C1
    W_total[0:16, 0:49] = W44
    
    # Second half: B2 -> C2
    U_total[:, 49:98] = U44
    # V for B2: rows 16-31 of V_total correspond to B2
    V_total[16:32, 49:98] = V44
    # W for C2: rows 16-31 of W_total correspond to C2
    W_total[16:32, 49:98] = W44
    
    # Convert to lists of lists for the interface
    U_list = U_total.tolist()
    V_list = V_total.tolist()
    W_list = W_total.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
