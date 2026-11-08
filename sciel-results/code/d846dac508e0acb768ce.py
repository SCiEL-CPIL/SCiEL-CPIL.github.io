import numpy as np
import time

def construct():
    start = time.time()
    deadline = start + 140

    n, m, p = 2, 4, 6
    rank = 42
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    STRASSEN_TENSORS = [
        {'u': {(0,0): 1, (1,1): 1}, 'v': {(0,0): 1, (1,1): 1}, 'w': {(0,0): 1, (1,1): 1}},
        {'u': {(1,0): 1, (1,1): 1}, 'v': {(0,0): 1}, 'w': {(1,0): 1, (1,1): -1}},
        {'u': {(0,0): 1}, 'v': {(0,1): 1, (1,1): -1}, 'w': {(0,1): 1, (1,1): 1}},
        {'u': {(1,1): 1}, 'v': {(1,0): 1, (0,0): -1}, 'w': {(0,0): 1, (1,0): 1}},
        {'u': {(0,0): 1, (0,1): 1}, 'v': {(1,1): 1}, 'w': {(0,0): -1, (0,1): 1}},
        {'u': {(1,0): 1, (0,0): -1}, 'v': {(0,0): 1, (0,1): 1}, 'w': {(1,1): 1}},
        {'u': {(0,1): 1, (1,1): -1}, 'v': {(1,0): 1, (1,1): 1}, 'w': {(0,0): 1}},
    ]
    
    col_offset = 0
    for half in range(2):
        a_col_base = half * 2
        b_row_base = half * 2
        for col_grp in range(3):
            b_col_base = col_grp * 2
            c_col_base = col_grp * 2
            for t in STRASSEN_TENSORS:
                for (di, dj), coeff_u in t['u'].items():
                    U[di * m + (a_col_base + dj), col_offset] += coeff_u
                for (di, dj), coeff_v in t['v'].items():
                    V[(b_row_base + di) * p + (b_col_base + dj), col_offset] += coeff_v
                for (di, dj), coeff_w in t['w'].items():
                    W[(c_col_base + dj) * n + di, col_offset] += coeff_w
                col_offset += 1
    
    assert col_offset == 42
    
    # Verify
    np.random.seed(42)
    A_test = np.random.randn(n, m)
    B_test = np.random.randn(m, p)
    C_expected = A_test @ B_test
    C_check = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            for j in range(m):
                for r in range(rank):
                    C_check[i, k] += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    max_err = np.max(np.abs(C_check - C_expected))
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
