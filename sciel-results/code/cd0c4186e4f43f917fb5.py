import numpy as np

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Part 1: 4x4x4 using Kronecker product of two 2x2x2 Strassen decompositions
    # 2x2x2 Strassen decomposition (Rank 7)
    # U, V, W are lists of 7 vectors of length 4
    U2 = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    V2 = [
        [1, 0, 0, 1],  # M1
        [1, 0, 0, 0],  # M2
        [0, 1, 0, -1], # M3
        [-1, 0, 1, 0], # M4
        [0, 0, 0, 1],  # M5
        [1, 1, 0, 0],  # M6
        [0, 0, 1, 1]   # M7
    ]
    W2 = [
        [1, 1, 1, 1],  # M1
        [0, 0, 1, -1], # M2
        [0, 1, -1, 0], # M3
        [-1, 0, 0, 1], # M4
        [0, 1, -1, 0], # M5
        [-1, 1, 0, 0], # M6
        [1, -1, 1, -1] # M7
    ]
    
    # Generate 49 terms for 4x4x4
    U4 = []
    V4 = []
    W4 = []
    
    for r in range(7):
        for s in range(7):
            u_kron = np.kron(U2[r], U2[s])
            v_kron = np.kron(V2[r], V2[s])
            w_kron = np.kron(W2[r], W2[s])
            U4.append(u_kron)
            V4.append(v_kron)
            W4.append(w_kron)
            
    # Part 2: 4x4x3 using Schoolbook
    # Columns 4, 5, 6 of B
    U3 = []
    V3 = []
    W3 = []
    for i in range(n):
        for j in range(m):
            for k in range(3):
                k_global = 4 + k
                u_vec = [0] * (n * m)
                v_vec = [0] * (m * p)
                w_vec = [0] * (p * n)
                
                u_vec[i * m + j] = 1
                v_vec[j * p + k_global] = 1
                w_vec[k_global * n + i] = 1
                
                U3.append(u_vec)
                V3.append(v_vec)
                W3.append(w_vec)
                
    # Combine
    U_all = U4 + U3
    V_all = V4 + V3
    W_all = W4 + W3
    
    rank = len(U_all)
    
    # Convert to lists of lists for output
    U_out = [list(map(float, u)) for u in U_all]
    V_out = [list(map(float, v)) for v in V_all]
    W_out = [list(map(float, w)) for w in W_all]
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
