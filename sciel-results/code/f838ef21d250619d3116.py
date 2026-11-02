import time

DIMENSIONS = [2, 2, 16]

def construct():
    n, m, p = DIMENSIONS
    rank = 56
    
    # Strassen 2x2x2 factors
    # U: (4, 7), V: (4, 7), W: (4, 7)
    # Indices for U: A[00], A[01], A[10], A[11]
    # Indices for V: B[00], B[01], B[10], B[11]
    # Indices for W: C[00], C[10], C[01], C[11] (based on k*n+i)
    
    U_strassen = [
        [1, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1]
    ]
    
    V_strassen = [
        [1, 1, 0, -1, 0, 1, 0],
        [1, 0, 0, 1, 1, 0, 1],
        [1, 0, 1, 0, 0, 1, 1],
        [1, 0, -1, 0, 1, 0, 1]
    ]
    
    W_strassen = [
        [1, -1, 1, 0, 0, 0, 0], # C[00]
        [0, 0, 0, 0, 0, 1, 1], # C[10]
        [0, 0, 0, 0, 0, 1, 1], # C[01]
        [1, -1, -1, 1, 0, 0, 0] # C[11]
    ]
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # 8 blocks
    for b in range(8):
        # Block b corresponds to columns 2b, 2b+1 of B and C
        for s in range(7):
            col = b * 7 + s
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_strassen[i * 2 + j][s]
                    
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    k_global = 2 * b + k_local
                    V[j * p + k_global][col] = V_strassen[j * 2 + k_local][s]
                    
            # Fill W
            # W index is k_global * n + i
            for k_local in range(2):
                for i in range(2):
                    k_global = 2 * b + k_local
                    w_idx = k_global * n + i
                    # Map local W index (0..3) to global
                    # Local 0: C[0,0] -> k_local=0, i=0
                    # Local 1: C[1,0] -> k_local=0, i=1
                    # Local 2: C[0,1] -> k_local=1, i=0
                    # Local 3: C[1,1] -> k_local=1, i=1
                    # Our W_strassen is indexed by [C00, C10, C01, C11]
                    # So local_idx = k_local * 2 + i
                    local_w_idx = k_local * 2 + i
                    W[w_idx][col] = W_strassen[local_w_idx][s]
                    
    return {"U": U, "V": V, "W": W}
