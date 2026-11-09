import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen coefficients for 2x2x2
    # s=1..7
    # alpha, beta, gamma vectors of length 4
    # Indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    
    alphas = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 1], # P2
        [1, 0, 0, 0], # P3
        [0, 0, 0, 1], # P4
        [1, 1, 0, 0], # P5
        [-1, 0, 1, 0],# P6
        [0, 1, -1, 0] # P7
    ]
    betas = [
        [1, 0, 0, 1], # P1
        [1, 0, 0, 0], # P2
        [0, 1, 0, -1],# P3
        [-1, 0, 1, 0],# P4
        [0, 0, 0, 1], # P5
        [1, 1, 0, 0], # P6
        [0, 0, 1, 1]  # P7
    ]
    gammas = [
        [1, 0, 0, 1], # P1
        [0, 0, 1, 0], # P2
        [0, 1, 0, 1], # P3
        [1, 0, 1, 0], # P4
        [-1, 1, 0, 0],# P5
        [0, 0, 0, 1], # P6
        [1, 0, 0, 0]  # P7
    ]
    
    # Helper to fill one block
    def fill_block(u_map, v_map, w_map, col_offset):
        for t in range(3):
            for s in range(7):
                c = col_offset + t * 7 + s
                for local_idx in range(4):
                    U[u_map[local_idx], c] = alphas[s][local_idx]
                    V[v_map[local_idx], c] = betas[s][local_idx]
                    W[w_map[local_idx], c] = gammas[s][local_idx]
                    
    # Block 1: A1 B1
    # A1: cols 0,1. u: 0,1,4,5
    u_map_1 = [0, 1, 4, 5]
    # B1: rows 0,1.
    # v_map depends on t.
    # w_map depends on t.
    
    # We need to loop t outside to update v_map
    for t in range(3):
        v_map_1 = [2*t, 2*t+1, 6+2*t, 7+2*t]
        w_map_1 = [4*t, 4*t+1, 4*t+2, 4*t+3]
        # Fill for this t
        for s in range(7):
            c = t * 7 + s
            for local_idx in range(4):
                U[u_map_1[local_idx], c] = alphas[s][local_idx]
                V[v_map_1[local_idx], c] = betas[s][local_idx]
                W[w_map_1[local_idx], c] = gammas[s][local_idx]
                
    # Block 2: A2 B2
    # A2: cols 2,3. u: 2,3,6,7
    u_map_2 = [2, 3, 6, 7]
    col_offset = 21
    
    for t in range(3):
        v_map_2 = [12+2*t, 13+2*t, 18+2*t, 19+2*t]
        w_map_2 = [4*t, 4*t+1, 4*t+2, 4*t+3]
        for s in range(7):
            c = col_offset + t * 7 + s
            for local_idx in range(4):
                U[u_map_2[local_idx], c] = alphas[s][local_idx]
                V[v_map_2[local_idx], c] = betas[s][local_idx]
                W[w_map_2[local_idx], c] = gammas[s][local_idx]
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
