import numpy as np

DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Define the rank-7 Strassen decomposition for 2x2 matrices
    U2 = np.array([
        [1, 0, 1, 0, 1, 0, 0], # a00
        [0, 0, 0, 0, 0, 1, 1], # a01
        [0, 1, 0, 0, 1, 0, 1], # a10
        [1, 1, 0, 1, 0, 1, 0]  # a11
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, -1, 0, 0, 0, 1], # b00
        [1, 0, 1, 0, 0, 0, 1],  # b01
        [0, 0, 0, -1, 0, 1, 0], # b10
        [0, 0, 0, 1, 1, 0, 0]   # b11
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1], # c00
        [0, 0, 1, 0, 1, 0, 0],  # c01
        [0, 1, 0, 1, 0, 0, 0],  # c10
        [1, -1, 1, 0, 0, 1, 0]  # c11
    ], dtype=float)
    
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    # Iterate over all valid (i, j, k) indices for 4x4 matrices
    # i: row of A (0..3)
    # j: col of A / row of B (0..3)
    # k: col of B (0..3)
    
    for i in range(4):
        for j in range(4):
            for k in range(4):
                I = i * 4 + j
                J = j * 4 + k
                K = k * 4 + i # W index convention: k*n + i
                
                for R1 in range(7):
                    for R2 in range(7):
                        R = R1 * 7 + R2
                        
                        # Kronecker product factors
                        u_val = U2[i, R1] * U2[j, R2]
                        v_val = V2[j, R1] * V2[k, R2]
                        w_val = W2[k, R1] * W2[i, R2]
                        
                        U4[I, R] = u_val
                        V4[J, R] = v_val
                        W4[K, R] = w_val
                        
    return {"U": U4.tolist(), "V": V4.tolist(), "W": W4.tolist()}
# EVOLVE-BLOCK-END
