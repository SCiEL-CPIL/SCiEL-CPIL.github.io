import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    # Rank 56 is derived from 8 blocks of 2x2x2, each with rank 7 (Strassen)
    rank = 56
    
    # Initialize U, V, W
    # U shape: (n*m, rank) = (4, 56)
    # V shape: (m*p, rank) = (32, 56)
    # W shape: (p*n, rank) = (32, 56)
    # Trace convention W[k*n+i, r] implies index is k*n + i.
    # For p=16, n=2, indices are 0..31.
    # C[i, k] -> W[k*n + i].
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen rank-7 decomposition vectors for 2x2 * 2x2
    # A indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # B indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # C indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    u_vecs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, 1],   # M2
        [1, 0, 0, 0],   # M3
        [0, 0, 0, 1],   # M4
        [1, 1, 0, 0],   # M5
        [-1, 0, 1, 0],  # M6
        [0, 1, 0, -1]   # M7
    ]
    
    v_vecs = [
        [1, 0, 0, 1],   # M1
        [1, 0, 0, 0],   # M2
        [0, 1, 0, -1],  # M3
        [-1, 0, 1, 0],  # M4
        [0, 0, 0, 1],   # M5
        [1, 1, 0, 0],   # M6
        [0, 0, 1, 1]    # M7
    ]
    
    w_vecs = [
        [1, 0, 0, 1],   # M1
        [0, 0, 1, -1],  # M2
        [0, 1, 0, 1],   # M3
        [1, 0, 1, 0],   # M4
        [-1, 1, 0, 0],  # M5
        [0, 0, 0, 1],   # M6
        [1, 0, 0, 0]    # M7
    ]
    
    # Fill U, V, W
    for r in range(8):
        base_col = 7 * r
        for s in range(7):
            col = base_col + s
            
            # U
            for li in range(4):
                if u_vecs[s][li] != 0:
                    U[li][col] = u_vecs[s][li]
                    
            # V
            # B indices for block r:
            # B[0, 2r] is V[2r]
            # B[0, 2r+1] is V[2r+1]
            # B[1, 2r] is V[16+2r]
            # B[1, 2r+1] is V[16+2r+1]
            v_idx = [2*r, 2*r+1, 16+2*r, 16+2*r+1]
            for li in range(4):
                if v_vecs[s][li] != 0:
                    V[v_idx[li]][col] = v_vecs[s][li]
                    
            # W
            # C indices for block r (trace convention W[k*n+i, r])
            # C[0, 2r] is W[2r*2 + 0] = W[4r]
            # C[0, 2r+1] is W[(2r+1)*2 + 0] = W[4r+2]
            # C[1, 2r] is W[2r*2 + 1] = W[4r+1]
            # C[1, 2r+1] is W[(2r+1)*2 + 1] = W[4r+3]
            w_idx = [4*r, 4*r+2, 4*r+1, 4*r+3]
            for li in range(4):
                if w_vecs[s][li] != 0:
                    W[w_idx[li]][col] = w_vecs[s][li]
                    
    # Convert to lists of lists as per interface
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
