import numpy as np
import time

def construct():
    """
    Returns a rank-minimal exact tensor decomposition for 2x2 * 2x6 matrix multiplication.
    Uses Strassen's algorithm on three 2x2 blocks, achieving rank 21.
    """
    start_time = time.time()
    deadline = 110.0  # Safety margin before 120s limit
    
    n, m, p = 2, 2, 6
    nm, mp, pn = n * m, m * p, p * n  # 4, 12, 12
    
    rank = 21
    
    U = np.zeros((nm, rank))
    V = np.zeros((mp, rank))
    W = np.zeros((pn, rank))
    
    # Strassen steps for 2x2 multiplication
    # u, v, w are vectors of length 4 corresponding to indices 00, 01, 10, 11
    strassen_steps = [
        {'u': [1, 0, 0, 1], 'v': [1, 0, 0, 1], 'w': [1, 0, 0, 1]},
        {'u': [0, 0, 1, 1], 'v': [1, 0, 0, 0], 'w': [0, 0, 1, -1]},
        {'u': [1, 0, 0, 0], 'v': [0, 1, 0, -1], 'w': [0, 1, 0, 1]},
        {'u': [0, 0, 0, 1], 'v': [-1, 0, 1, 0], 'w': [1, 0, 1, 0]},
        {'u': [1, 1, 0, 0], 'v': [0, 0, 0, 1], 'w': [-1, 1, 0, 0]},
        {'u': [-1, 0, 1, 0], 'v': [1, 1, 0, 0], 'w': [0, 0, 0, 1]},
        {'u': [0, 1, 0, -1], 'v': [0, 0, 1, 1], 'w': [1, 0, 0, 0]},
    ]
    
    # Pre-compute indices
    # V index: j * 6 + k
    # W index: k * 2 + i
    # Local indices 0,1,2,3 correspond to (0,0), (0,1), (1,0), (1,1)
    
    v_indices = []
    w_indices = []
    for g in range(3):
        k0 = 2 * g
        k1 = 2 * g + 1
        # V indices for B_g: j=0,1 and k=k0,k1
        # Order: 00, 01, 10, 11 -> (0,k0), (0,k1), (1,k0), (1,k1)
        v_indices.append([
            0 * 6 + k0, 
            0 * 6 + k1, 
            1 * 6 + k0, 
            1 * 6 + k1
        ])
        # W indices for C_g: i=0,1 and k=k0,k1
        # Order: 00, 01, 10, 11 -> (i=0,k0), (i=0,k1), (i=1,k0), (i=1,k1)
        # Index formula: k * 2 + i
        w_indices.append([
            k0 * 2 + 0,  # 00: i=0, k=k0
            k1 * 2 + 0,  # 01: i=0, k=k1
            k0 * 2 + 1,  # 10: i=1, k=k0
            k1 * 2 + 1   # 11: i=1, k=k1
        ])

    # Fill U, V, W
    for g in range(3):
        for s in range(7):
            r = 7 * g + s
            step = strassen_steps[s]
            
            u_list = step['u']
            v_list = step['v']
            w_list = step['w']
            
            v_idx_list = v_indices[g]
            w_idx_list = w_indices[g]
            
            for local_idx in range(4):
                u_val = u_list[local_idx]
                if u_val != 0:
                    U[local_idx, r] = u_val
                    
                v_val = v_list[local_idx]
                if v_val != 0:
                    V[v_idx_list[local_idx], r] = v_val
                    
                w_val = w_list[local_idx]
                if w_val != 0:
                    W[w_idx_list[local_idx], r] = w_val

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
