import numpy as np
import time
import sys

DIMENSIONS = [4, 4, 7]
_START_TIME = time.time()
TIME_LIMIT = 145

def _time_remaining():
    return TIME_LIMIT - (time.time() - _START_TIME)

def construct():
    n, m, p = DIMENSIONS
    total_rank = 93
    
    U = np.zeros((n * m, total_rank), dtype=float)
    V = np.zeros((m * p, total_rank), dtype=float)
    W = np.zeros((p * n, total_rank), dtype=float)
    
    current_rank = 0
    
    # Strassen 2x2x2 terms (local indices)
    s2 = [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}),
        ({(0,0): 1}, {(0,1): 1, (1,1): -1}),
        ({(1,1): 1}, {(1,0): 1, (0,0): -1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1})
    ]
    
    # Strassen 4x4x4 block structure
    s4 = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}),
        ({0: 1}, {1: 1, 3: -1}),
        ({3: 1}, {2: 1, 0: -1}),
        ({0: 1, 1: 1}, {3: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1})
    ]
    
    # C matrix coefficients for Strassen (local 2x2 or 4x4)
    c2 = [
        {(0,0): 1, (1,1): 1},
        {(1,0): 1, (1,1): -1},
        {(0,1): 1, (1,1): 1},
        {(0,0): 1, (1,0): 1},
        {(0,0): -1, (0,1): 1},
        {(1,1): 1},
        {(0,0): 1}
    ]
    
    # Part 1: 4x4x4 using Strassen (Rank 49)
    for m_idx in range(7):
        x_def, y_def = s4[m_idx]
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = s2[t_idx]
            
            # U coefficients
            for (li, lj), cx in x_sub.items():
                for b, cb in x_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    U[i * m + j, r] += cx * cb
            
            # V coefficients
            for (li, lj), cy in y_sub.items():
                for b, cb in y_def.items():
                    bi, bj = b // 2, b % 2
                    i = 2 * bi + li
                    j = 2 * bj + lj
                    V[i * p + j, r] += cy * cb
            
            # W coefficients
            c_block = c2[m_idx]
            for (li, lj), cc in c2[t_idx].items():
                if cc == 0: continue
                for (br, bc), cb in c_block.items():
                    if cb == 0: continue
                    i_f = 2 * br + li
                    k_f = 2 * bc + lj
                    W[k_f * n + i_f, r] += cc * cb
    
    # Part 2: 4x4x3 using 4 * (2x2x3) (Rank 44)
    def add_2x2x3(arb, acb, brb, crb):
        nonlocal current_rank
        
        # Strassen part (cols 4,5)
        for t_idx in range(7):
            r = current_rank
            current_rank += 1
            x_sub, y_sub = s2[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * arb + li
                j = 2 * acb + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * brb + li
                j = 4 + lj
                V[i * p + j, r] += cy
            
            for (li, lj), cc in c2[t_idx].items():
                if cc == 0: continue
                i_f = 2 * crb + li
                k_f = 4 + lj
                W[k_f * n + i_f, r] += cc
        
        # Schoolbook part (col 6)
        sb = [
            ({(0,0): 1}, {(0,0): 1}, (0,0)),
            ({(0,1): 1}, {(1,0): 1}, (0,0)),
            ({(1,0): 1}, {(0,0): 1}, (1,0)),
            ({(1,1): 1}, {(1,0): 1}, (1,0))
        ]
        
        for t_idx in range(4):
            r = current_rank
            current_rank += 1
            x_sub, y_sub, cp = sb[t_idx]
            
            for (li, lj), cx in x_sub.items():
                i = 2 * arb + li
                j = 2 * acb + lj
                U[i * m + j, r] += cx
            
            for (li, lj), cy in y_sub.items():
                i = 2 * brb + li
                V[i * p + 6, r] += cy
            
            i_f = 2 * crb + cp[0]
            W[6 * n + i_f, r] = 1.0
    
    add_2x2x3(0, 0, 0, 0)
    add_2x2x3(0, 1, 1, 0)
    add_2x2x3(1, 0, 0, 1)
    add_2x2x3(1, 1, 1, 1)
    
    U = U[:, :current_rank]
    V = V[:, :current_rank]
    W = W[:, :current_rank]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
