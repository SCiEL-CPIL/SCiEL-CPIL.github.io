import time
import numpy as np

DIMENSIONS = [2, 2, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    n, m, p = DIMENSIONS
    rank = 21
    
    # Strassen coefficients for 2x2 matrix multiplication
    # U, V, W vectors for the 7 rank-1 terms
    strassen = [
        {'U': [1, 0, 0, 1], 'V': [1, 0, 0, 1], 'W': [1, 0, 0, 1]}, # P1
        {'U': [0, 0, 1, 1], 'V': [1, 0, 0, 0], 'W': [0, 1, 0, -1]}, # P2
        {'U': [1, 0, 0, 0], 'V': [0, 1, 0, -1], 'W': [0, 0, 1, 1]}, # P3
        {'U': [0, 0, 0, 1], 'V': [-1, 0, 1, 0], 'W': [1, 1, 0, 0]}, # P4
        {'U': [1, 1, 0, 0], 'V': [0, 0, 0, 1], 'W': [-1, 0, 1, 0]}, # P5
        {'U': [-1, 0, 1, 0], 'V': [1, 1, 0, 0], 'W': [0, 0, 0, 1]}, # P6
        {'U': [0, 1, 0, -1], 'V': [0, 0, 1, 1], 'W': [1, 0, 0, 0]}, # P7
    ]
    
    # U: 4 x 21
    # V: 12 x 21
    # W: 12 x 21
    U = np.zeros((4, rank), dtype=int)
    V = np.zeros((12, rank), dtype=int)
    W = np.zeros((12, rank), dtype=int)
    
    for b in range(3):
        for s in range(7):
            col = b * 7 + s
            sc = strassen[s]
            
            u_vals = sc['U']
            v_vals = sc['V']
            w_vals = sc['W']
            
            # U
            U[0, col] = u_vals[0]
            U[1, col] = u_vals[1]
            U[2, col] = u_vals[2]
            U[3, col] = u_vals[3]
            
            # V mapping
            v_idx = [2*b, 2*b+1, 6+2*b, 6+2*b+1]
            V[v_idx[0], col] = v_vals[0]
            V[v_idx[1], col] = v_vals[1]
            V[v_idx[2], col] = v_vals[2]
            V[v_idx[3], col] = v_vals[3]
            
            # W mapping
            w_idx = [4*b, 4*b+1, 4*b+2, 4*b+3]
            W[w_idx[0], col] = w_vals[0]
            W[w_idx[1], col] = w_vals[1]
            W[w_idx[2], col] = w_vals[2]
            W[w_idx[3], col] = w_vals[3]
            
    return {"U": U, "V": V, "W": W}
