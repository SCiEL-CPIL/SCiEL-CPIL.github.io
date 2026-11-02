import numpy as np

DIMENSIONS = [2, 2, 2]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 7
    
    U = np.zeros((n*m, rank), dtype=np.float64)
    V = np.zeros((m*p, rank), dtype=np.float64)
    W = np.zeros((p*n, rank), dtype=np.float64)
    
    # Strassen's Algorithm factors
    # Format: (u_coords, u_vals, v_coords, v_vals, w_coords, w_vals)
    # u_coords: (i, j), v_coords: (j, k), w_coords: (k, i)
    factors = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0,0), (1,1)], [1, 1], 
         [(0,0), (1,1)], [1, 1], 
         [(0,0), (1,1)], [1, 1]),
        # M2 = (A10 + A11) * B00
        ([(1,0), (1,1)], [1, 1], 
         [(0,0)], [1], 
         [(0,1), (1,1)], [1, -1]),
        # M3 = A00 * (B01 - B11)
        ([(0,0)], [1], 
         [(0,1), (1,1)], [1, -1], 
         [(1,0), (1,1)], [1, 1]),
        # M4 = A11 * (B10 - B00)
        ([(1,1)], [1], 
         [(1,0), (0,0)], [1, -1], 
         [(0,0), (0,1)], [1, 1]),
        # M5 = (A00 + A01) * B11
        ([(0,0), (0,1)], [1, 1], 
         [(1,1)], [1], 
         [(0,0), (1,0)], [-1, 1]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(1,0), (0,0)], [1, -1], 
         [(0,0), (0,1)], [1, 1], 
         [(1,1)], [1]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(0,1), (1,1)], [1, -1], 
         [(1,0), (1,1)], [1, 1], 
         [(0,0)], [1]),
    ]
    
    for r, (u_c, u_v, v_c, v_v, w_c, w_v) in enumerate(factors):
        for (i, j), val in zip(u_c, u_v):
            U[i*m + j, r] = val
        for (j, k), val in zip(v_c, v_v):
            V[j*p + k, r] = val
        for (k, i), val in zip(w_c, w_v):
            W[k*n + i, r] = val
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
