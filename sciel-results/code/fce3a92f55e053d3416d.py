import numpy as np

DIMENSIONS = [2, 2, 10]

def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=10
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (20, rank)
    # W shape: (p*n, rank) = (20, rank)
    
    num_blocks = p // 2  # 5 blocks
    block_rank = 7       # Strassen rank for 2x2
    rank = num_blocks * block_rank  # 35
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix)
    # Rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    U_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Strassen coefficients for V (B matrix, local)
    # Local rows: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    V_coeffs = [
        [1, 0, 0, 1],  # M1: e, h
        [1, 0, 0, 0],  # M2: e
        [0, 1, 0, -1], # M3: f, -h
        [-1, 0, 1, 0], # M4: -e, g
        [0, 0, 0, 1],  # M5: h
        [1, 1, 0, 0],  # M6: e, f
        [0, 0, 1, 1]   # M7: g, h
    ]
    
    # Strassen coefficients for W (C matrix, local)
    # Local rows: 0:(k=0,i=0), 1:(k=0,i=1), 2:(k=1,i=0), 3:(k=1,i=1)
    W_coeffs = [
        [1, 0, 0, 1],  # M1: C11, C22
        [0, 1, 0, -1], # M2: C21, -C22
        [0, 0, 1, 1],  # M3: C12, C22
        [1, 1, 0, 0],  # M4: C11, C21
        [-1, 0, 1, 0], # M5: -C11, C12
        [0, 0, 0, 1],  # M6: C22
        [1, 0, -1, 0]  # M7: C11, -C12
    ]
    
    for q in range(num_blocks):
        for s in range(block_rank):
            col = q * block_rank + s
            
            # Fill U
            for i in range(n * m):
                U[i][col] = U_coeffs[s][i]
                
            # Fill V
            # Global row index for V: j * p + k_global
            # k_global = 2 * q + k_local
            for j in range(m):
                for k_local in range(2):
                    k_global = 2 * q + k_local
                    v_row = j * p + k_global
                    # Local index in V_coeffs for this j, k_local
                    # Local rows are ordered: (j=0,k=0), (j=0,k=1), (j=1,k=0), (j=1,k=1)
                    local_idx = j * 2 + k_local
                    V[v_row][col] = V_coeffs[s][local_idx]
                    
            # Fill W
            # Global row index for W: k_global * n + i
            # k_global = 2 * q + k_local
            for k_local in range(2):
                k_global = 2 * q + k_local
                for i in range(n):
                    w_row = k_global * n + i
                    # Local index in W_coeffs for this k_local, i
                    # Local rows are ordered: (k=0,i=0), (k=0,i=1), (k=1,i=0), (k=1,i=1)
                    local_idx = k_local * n + i
                    W[w_row][col] = W_coeffs[s][local_idx]
                    
    return {"U": U, "V": V, "W": W}
