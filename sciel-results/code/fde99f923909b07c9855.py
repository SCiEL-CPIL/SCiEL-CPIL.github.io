import numpy as np

DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 16
    num_blocks = p // 2  # 8 blocks
    core_rank = 7
    rank = core_rank * num_blocks  # 56
    
    # Define Strassen factors for 2x2x2 core
    # U_core: 4x7, V_core: 4x7, W_core: 4x7
    # Indices:
    # U: i*m+j -> 0:00, 1:01, 2:10, 3:11
    # V: j*p+k -> 0:00, 1:01, 2:10, 3:11
    # W: k*n+i -> 0:00, 1:01, 2:10, 3:11
    
    U_core = np.zeros((4, 7))
    V_core = np.zeros((4, 7))
    W_core = np.zeros((4, 7))
    
    # Strassen factors
    # M1 = (a00+a11)(b00+b11) -> c00+=1, c11+=1
    U_core[[0, 3], 0] = 1
    V_core[[0, 3], 0] = 1
    W_core[[0, 3], 0] = 1
    
    # M2 = (a10+a11)b00 -> c10+=1, c11+=-1
    U_core[[2, 3], 1] = 1
    V_core[[0], 1] = 1
    W_core[[2], 1] = 1
    W_core[[3], 1] = -1
    
    # M3 = a00(b01-b11) -> c01+=1, c11+=1
    U_core[[0], 2] = 1
    V_core[[1], 2] = 1
    V_core[[3], 2] = -1
    W_core[[1], 2] = 1
    W_core[[3], 2] = 1
    
    # M4 = a11(b10-b00) -> c00+=1, c10+=1
    U_core[[3], 3] = 1
    V_core[[2], 3] = 1
    V_core[[0], 3] = -1
    W_core[[0], 3] = 1
    W_core[[2], 3] = 1
    
    # M5 = (a00+a01)b11 -> c00+=-1, c01+=1
    U_core[[0, 1], 4] = 1
    V_core[[3], 4] = 1
    W_core[[0], 4] = -1
    W_core[[1], 4] = 1
    
    # M6 = (a10-a00)(b00+b01) -> c11+=1
    U_core[[2], 5] = 1
    U_core[[0], 5] = -1
    V_core[[0, 1], 5] = 1
    W_core[[3], 5] = 1
    
    # M7 = (a01-a11)(b10+b11) -> c00+=1
    U_core[[1], 6] = 1
    U_core[[3], 6] = -1
    V_core[[2, 3], 6] = 1
    W_core[[0], 6] = 1
    
    # Construct Full U: 4 x 56
    # U is just U_core repeated for each block
    U = np.tile(U_core, (1, num_blocks))
    
    # Construct Full V: 32 x 56
    V = np.zeros((m * p, rank))
    for t in range(num_blocks):
        # Columns for block t
        col_start = t * core_rank
        col_end = (t + 1) * core_rank
        
        # Rows for block t in V: j*p + (2t + k_local)
        # j in 0..1, k_local in 0..1
        # V_core index: j*2 + k_local
        for j in range(m):
            for k_local in range(2):
                v_row = j * p + (2 * t + k_local)
                vc_row = j * 2 + k_local
                V[v_row, col_start:col_end] = V_core[vc_row, :]
                
    # Construct Full W: 32 x 56
    W = np.zeros((p * n, rank))
    for t in range(num_blocks):
        # Columns for block t
        col_start = t * core_rank
        col_end = (t + 1) * core_rank
        
        # Rows for block t in W: k*n + i = (2t + k_local)*n + i
        # k_local in 0..1, i in 0..1
        # W_core index: k_local*n + i
        for k_local in range(2):
            for i in range(n):
                w_row = (2 * t + k_local) * n + i
                wc_row = k_local * n + i
                W[w_row, col_start:col_end] = W_core[wc_row, :]
                
    # Convert to lists as required by interface (or return np arrays if allowed? 
    # Interface says "factor matrices... with integer/half-integer entries".
    # Current program returns lists of lists. 
    # Evaluator likely accepts lists or arrays. 
    # To be safe, stick to lists as in seed, or check if np arrays are ok.
    # The seed returns lists. I will return lists to be safe.
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
