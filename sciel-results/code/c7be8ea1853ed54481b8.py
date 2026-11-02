import numpy as np
import time

DIMENSIONS = [2, 2, 16]
_DEADLINE = 140.0

def construct():
    _t0 = time.time()
    n, m, p = DIMENSIONS
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen vectors for 2x2 matrix multiplication
    # A is flattened as [A00, A01, A10, A11]
    # B is flattened as [B00, B01, B10, B11]
    # C is calculated as C = A * B
    
    # The specific permutation of Strassen steps used in the original working code:
    # Step 0: M3 = (A00+A11)(B00+B01)
    # Step 1: M7 = (A10+A11)(B01-B00)  <- Note: Code had [0,0,1,1] for A, let's verify consistency
    # Actually, let's just copy the exact vectors from the working code to ensure validity.
    
    u_s = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v_s = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_s = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0]
    ]
    
    for b in range(num_blocks):
        col_offset = b * 7
        for s in range(7):
            c = col_offset + s
            
            # U: A indices (same for all blocks)
            U[0, c] = u_s[s][0]
            U[1, c] = u_s[s][1]
            U[2, c] = u_s[s][2]
            U[3, c] = u_s[s][3]
            
            # V: B indices for block b
            # B is 2x16. Block b is cols 2b, 2b+1.
            # V index: row * 16 + col
            V[2*b, c] = v_s[s][0]
            V[2*b+1, c] = v_s[s][1]
            V[16+2*b, c] = v_s[s][2]
            V[16+2*b+1, c] = v_s[s][3]
            
            # W: C indices
            # Convention: W[k*n+i, r]
            # C[i, k] maps to W[k*n + i, r]
            # n=2
            # Block b covers k=2b, 2b+1
            W[4*b, c] = w_s[s][0]      # C[0, 2b]
            W[4*b+2, c] = w_s[s][1]    # C[0, 2b+1]
            W[4*b+1, c] = w_s[s][2]    # C[1, 2b]
            W[4*b+3, c] = w_s[s][3]    # C[1, 2b+1]
            
    if time.time() - _t0 > _DEADLINE:
        pass
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
