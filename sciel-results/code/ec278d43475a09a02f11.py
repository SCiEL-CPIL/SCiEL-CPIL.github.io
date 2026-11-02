import numpy as np

def construct():
    # Dimensions
    n, m, p = 4, 4, 6
    
    # 1. Construct Strassen decomposition for 2x2x2
    # Rank 7
    # U1: 4x7, V1: 4x7, W1: 4x7
    U1 = np.zeros((4, 7), dtype=float)
    V1 = np.zeros((4, 7), dtype=float)
    W1 = np.zeros((4, 7), dtype=float)
    
    # Product 1
    U1[0, 0] = 1; U1[3, 0] = 1
    V1[0, 0] = 1; V1[3, 0] = 1
    W1[0, 0] = 1; W1[3, 0] = 1
    
    # Product 2
    U1[2, 1] = 1; U1[3, 1] = 1
    V1[0, 1] = 1
    W1[2, 1] = 1; W1[3, 1] = -1
    
    # Product 3
    U1[0, 2] = 1
    V1[1, 2] = 1; V1[3, 2] = -1
    W1[1, 2] = 1; W1[3, 2] = 1
    
    # Product 4
    U1[3, 3] = 1
    V1[0, 3] = -1; V1[2, 3] = 1
    W1[0, 3] = 1; W1[2, 3] = 1
    
    # Product 5
    U1[0, 4] = 1; U1[1, 4] = 1
    V1[3, 4] = 1
    W1[0, 4] = -1; W1[1, 4] = 1
    
    # Product 6
    U1[0, 5] = -1; U1[2, 5] = 1
    V1[0, 5] = 1; V1[1, 5] = 1
    W1[3, 5] = 1
    
    # Product 7
    U1[1, 6] = 1; U1[3, 6] = -1
    V1[2, 6] = 1; V1[3, 6] = 1
    W1[0, 6] = 1

    # 2. Construct Schoolbook decomposition for 2x2x3
    # Rank 12
    # U2: 4x12, V2: 6x12, W2: 6x12
    n2, m2, p2 = 2, 2, 3
    rank2 = n2 * m2 * p2
    U2 = np.zeros((n2 * m2, rank2), dtype=float)
    V2 = np.zeros((m2 * p2, rank2), dtype=float)
    W2 = np.zeros((p2 * n2, rank2), dtype=float)
    
    col = 0
    for i in range(n2):
        for j in range(m2):
            for k in range(p2):
                u_idx = i * m2 + j
                v_idx = j * p2 + k
                w_idx = k * n2 + i
                U2[u_idx, col] = 1
                V2[v_idx, col] = 1
                W2[w_idx, col] = 1
                col += 1

    # 3. Combine via Kronecker Product
    rank1 = 7
    rank_total = rank1 * rank2
    
    U = np.zeros((n * m, rank_total), dtype=float)
    V = np.zeros((m * p, rank_total), dtype=float)
    W = np.zeros((p * n, rank_total), dtype=float)
    
    # Dimensions of components
    u1_dim, u2_dim = 4, 4
    v1_dim, v2_dim = 4, 6
    w1_dim, w2_dim = 4, 6
    
    for c1 in range(rank1):
        for c2 in range(rank2):
            c_total = c1 * rank2 + c2
            
            # U
            for u1 in range(u1_dim):
                val1 = U1[u1, c1]
                if val1 != 0:
                    for u2 in range(u2_dim):
                        val2 = U2[u2, c2]
                        if val2 != 0:
                            U[u1 * u2_dim + u2, c_total] = val1 * val2
            
            # V
            for v1 in range(v1_dim):
                val1 = V1[v1, c1]
                if val1 != 0:
                    for v2 in range(v2_dim):
                        val2 = V2[v2, c2]
                        if val2 != 0:
                            V[v1 * v2_dim + v2, c_total] = val1 * val2
            
            # W
            for w1 in range(w1_dim):
                val1 = W1[w1, c1]
                if val1 != 0:
                    for w2 in range(w2_dim):
                        val2 = W2[w2, c2]
                        if val2 != 0:
                            W[w1 * w2_dim + w2, c_total] = val1 * val2

    # Convert to lists for output
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
