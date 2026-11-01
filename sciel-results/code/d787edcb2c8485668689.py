import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = 115.0 # 5 seconds safety margin before 120s
    
    n, m, p = 2, 2, 8
    rank = 28
    
    # Initialize U, V, W
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen factors for 2x2 x 2x2 -> 2x2
    # A is 2x2, indices: 00, 01, 10, 11
    # B is 2x2, indices: 00, 01, 10, 11
    # C is 2x2, indices: 00, 01, 10, 11
    
    # Strassen terms P1..P7
    # P1 = (A00+A11)*(B00+B11)
    # P2 = (A10+A11)*B00
    # P3 = A00*(B01-B11)
    # P4 = A11*(B10-B00)
    # P5 = (A00+A01)*B11
    # P6 = (A10-A00)*(B00+B01)
    # P7 = (A01-A11)*(B10+B11)
    
    # U vectors (coefficients for A)
    # Order: 00, 01, 10, 11
    U_strassen = np.array([
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, 0, -1] # P7
    ])
    
    # V vectors (coefficients for B)
    # Order: 00, 01, 10, 11
    V_strassen = np.array([
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ])
    
    # W vectors (coefficients for C)
    # Order: 00, 01, 10, 11
    W_strassen = np.array([
        [1, 0, 0, 1], # P1
        [0, 0, 1, -1],# P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ])
    
    num_blocks = p // 2 # 4 blocks of 2 columns
    
    # Vectorized construction
    # U is repeated 4 times
    U = np.tile(U_strassen.T, (1, num_blocks))
    
    # V and W need to be interleaved based on block indices
    # V has shape (16, 28). 
    # For each block t (0..3), we place V_strassen into rows corresponding to B's block t.
    # B indices for block t: 2t, 2t+1, 8+2t, 8+2t+1
    # W indices for block t: 4t, 4t+2, 4t+1, 4t+3
    
    for t in range(num_blocks):
        col_start = t * 7
        col_end = (t + 1) * 7
        
        v_indices = [2*t, 2*t+1, 8+2*t, 8+2*t+1]
        w_indices = [4*t, 4*t+2, 4*t+1, 4*t+3]
        
        # V_strassen is 7x4. We need to transpose it to 4x7 to fit into columns?
        # No, V_strassen[r][l] is the coefficient for B element l in term r.
        # In V matrix, column `col` corresponds to term `r`.
        # So V[index, col] = V_strassen[r, l].
        # We are filling columns col_start to col_end.
        # For a specific column `c` in this range, `r = c - col_start`.
        # V[v_indices[l], c] = V_strassen[r, l].
        
        # This is equivalent to:
        # V[v_indices, col_start:col_end] = V_strassen.T
        
        V[v_indices, col_start:col_end] = V_strassen.T
        W[w_indices, col_start:col_end] = W_strassen.T
        
    if time.time() > deadline:
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
