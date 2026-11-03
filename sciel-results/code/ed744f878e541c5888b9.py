import time
import numpy as np

DIMENSIONS = [2, 4, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    n, m, p = DIMENSIONS
    rank = 36
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    r = 0
    
    strassen = [
        { 'U': [(0,0,1), (1,1,1)], 'V': [(0,0,1), (1,1,1)], 'W': [(0,0,1), (1,1,1)] },
        { 'U': [(1,0,1), (1,1,1)], 'V': [(0,0,1)], 'W': [(0,1,1), (1,1,-1)] },
        { 'U': [(0,0,1)], 'V': [(1,0,1), (1,1,-1)], 'W': [(1,0,1), (1,1,1)] },
        { 'U': [(1,1,1)], 'V': [(0,1,1), (1,0,-1)], 'W': [(0,0,1), (0,1,1)] },
        { 'U': [(0,0,1), (0,1,1)], 'V': [(1,1,1)], 'W': [(0,0,-1), (1,0,1)] },
        { 'U': [(1,0,1), (0,0,-1)], 'V': [(0,0,1), (1,0,1)], 'W': [(1,1,1)] },
        { 'U': [(0,1,1), (1,1,-1)], 'V': [(0,1,1), (1,1,1)], 'W': [(0,0,1)] }
    ]
    
    schoolbook = [
        { 'U': [(0,0,1)], 'V': [(0,0,1)], 'W': [(0,0,1)] },
        { 'U': [(0,1,1)], 'V': [(1,0,1)], 'W': [(0,0,1)] },
        { 'U': [(1,0,1)], 'V': [(0,0,1)], 'W': [(0,1,1)] },
        { 'U': [(1,1,1)], 'V': [(1,0,1)], 'W': [(0,1,1)] }
    ]

    for b in range(2):
        if time.time() > deadline:
            break
        for c in range(3):
            if time.time() > deadline:
                break
            if c < 2:
                patterns = strassen
                k_offset = c * 2
            else:
                patterns = schoolbook
                k_offset = 4
                
            for pat in patterns:
                for i_l, j_l, val in pat['U']:
                    i_g = i_l
                    j_g = b * 2 + j_l
                    U[i_g * m + j_g, r] += val
                
                for j_l, k_l, val in pat['V']:
                    j_g = b * 2 + j_l
                    k_g = k_offset + k_l
                    V[j_g * p + k_g, r] += val
                
                for k_l, i_l, val in pat['W']:
                    k_g = k_offset + k_l
                    i_g = i_l
                    W[k_g * n + i_g, r] += val
                
                r += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
