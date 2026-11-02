import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 2, 2, 12
    num_blocks = p // m  # 6 blocks
    rank = num_blocks * 7
    R = rank
    
    # Coefficients for Strassen 2x2x2
    # A indices: 0:a, 1:b, 2:c, 3:d
    # B indices: 0:e, 1:f, 2:g, 3:h
    # C indices (local): 0:C00, 1:C10, 2:C01, 3:C11 (based on W[k*n+i])
    
    # U coeffs (for A)
    # M1: a+d -> [1, 0, 0, 1]
    # M2: c+d -> [0, 0, 1, 1]
    # M3: a   -> [1, 0, 0, 0]
    # M4: d   -> [0, 0, 0, 1]
    # M5: a+b -> [1, 1, 0, 0]
    # M6: c-a -> [-1, 0, 1, 0]
    # M7: b-d -> [0, 1, 0, -1]
    U_block = np.array([
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ])
    
    # V coeffs (for B)
    # M1: e+h -> [1, 0, 0, 1]
    # M2: e   -> [1, 0, 0, 0]
    # M3: f-h -> [0, 1, 0, -1]
    # M4: g-e -> [0, 0, 1, -1]
    # M5: h   -> [0, 0, 0, 1]
    # M6: e+f -> [1, 1, 0, 0]
    # M7: g+h -> [0, 0, 1, 1]
    V_block = np.array([
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, 0, 1, -1],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ])
    
    # W coeffs (for C)
    # M1: C00, C11 -> [1, 0, 0, 1]
    # M2: C10, -C11 -> [0, 1, 0, -1]
    # M3: C01, C11 -> [0, 0, 1, 1]
    # M4: C00, C10 -> [1, 1, 0, 0]
    # M5: -C00, C01 -> [-1, 0, 1, 0]
    # M6: C11 -> [0, 0, 0, 1]
    # M7: C00 -> [1, 0, 0, 0]
    W_block = np.array([
        [1, 0, 0, 1],
        [0, 1, 0, -1],
        [0, 0, 1, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ])
    
    # Construct full matrices
    U = np.zeros((n * m, R), dtype=np.int32)
    V = np.zeros((m * p, R), dtype=np.int32)
    W = np.zeros((p * n, R), dtype=np.int32)
    
    for t in range(num_blocks):
        col_start = t * 7
        col_end = col_start + 7
        
        # Fill U
        U[:, col_start:col_end] = U_block.T
        
        # Fill V
        # B is 2x12. Block t covers columns 2t and 2t+1.
        # B indices in V are flattened (2, 12).
        # Row index in V corresponds to (j, k) in B.
        # j is 0..1, k is 0..11.
        # In the block, local B indices 0..3 correspond to:
        # 0: (0, 2t)
        # 1: (0, 2t+1)
        # 2: (1, 2t)
        # 3: (1, 2t+1)
        # V_block is 7x4. We need to place these into the correct rows of V.
        # V rows are j*p + k? Or k*m + j?
        # Prompt: V shape (m*p, rank). 
        # In prompt code: 
        # local_b 0 -> global_b = 2*t
        # local_b 1 -> global_b = 2*t+1
        # local_b 2 -> global_b = m*p//2 + 2*t
        # local_b 3 -> global_b = m*p//2 + 2*t + 1
        # m*p = 24. m*p//2 = 12.
        # So local 0,1 are in first half (j=0), local 2,3 in second half (j=1).
        # This implies V index is j*p + k? 
        # If j=0, k=2t -> 0*12 + 2t = 2t. Matches.
        # If j=1, k=2t -> 1*12 + 2t = 12+2t. Matches.
        
        v_indices = [2*t, 2*t+1, 12+2*t, 12+2*t+1]
        for s in range(7):
            col = col_start + s
            V[v_indices[0], col] = V_block[s, 0]
            V[v_indices[1], col] = V_block[s, 1]
            V[v_indices[2], col] = V_block[s, 2]
            V[v_indices[3], col] = V_block[s, 3]
            
        # Fill W
        # W shape (p*n, rank). Index k*n + i.
        # Block t corresponds to k=2t, 2t+1.
        # Local C indices 0..3:
        # 0: C00 (i=0, k=2t) -> idx (2t)*2 + 0 = 4t
        # 1: C10 (i=1, k=2t) -> idx (2t)*2 + 1 = 4t+1
        # 2: C01 (i=0, k=2t+1) -> idx (2t+1)*2 + 0 = 4t+2
        # 3: C11 (i=1, k=2t+1) -> idx (2t+1)*2 + 1 = 4t+3
        
        w_indices = [4*t, 4*t+1, 4*t+2, 4*t+3]
        for s in range(7):
            col = col_start + s
            W[w_indices[0], col] = W_block[s, 0]
            W[w_indices[1], col] = W_block[s, 1]
            W[w_indices[2], col] = W_block[s, 2]
            W[w_indices[3], col] = W_block[s, 3]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
