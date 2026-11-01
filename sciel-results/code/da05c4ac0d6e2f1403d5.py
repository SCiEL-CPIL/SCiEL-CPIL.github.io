import numpy as np
import time

DIMENSIONS = [2, 4, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 150.0
    
    # We will construct the rank 28 decomposition using Strassen's algorithm on 2x2 blocks.
    # A (2x4) = [A_L, A_R], B (4x4) = [[B_TL, B_TR], [B_BL, B_BR]]
    # C_L = A_L B_TL + A_R B_BL
    # C_R = A_L B_TR + A_R B_BR
    
    # Strassen terms for X(2x2) * Y(2x2)
    # Local indices: 0:(0,0), 1:(0,1), 2:(1,0), 3:(1,1)
    # Term format: (u_coeffs, v_coeffs, w_coeffs) where coeffs are lists of (local_idx, val)
    
    def s_term1(u, v, w):
        return ([(u,0,1), (u,3,1)], [(v,0,1), (v,3,1)], [(w,0,1), (w,3,1)])
    # This approach is getting messy. Let's stick to the explicit list.

    strassen = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]
    
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Global index mappings
    # A (2x4):
    # A_L cols 0,1: 0,1,4,5
    # A_R cols 2,3: 2,3,6,7
    # B (4x4):
    # TL: 0,1,4,5
    # TR: 2,3,6,7
    # BL: 8,9,12,13
    # BR: 10,11,14,15
    # C (2x4): W[k*n+i] = W[k*2+i]
    # C_L cols 0,1: 0,1,2,3 (k=0,1; i=0,1) -> idx: 0,1,2,3
    # C_R cols 2,3: 4,5,6,7 (k=2,3; i=0,1) -> idx: 4,5,6,7
    
    # Mappings for U (A)
    u_map_L = [0, 1, 4, 5]
    u_map_R = [2, 3, 6, 7]
    
    # Mappings for V (B)
    v_map_TL = [0, 1, 4, 5]
    v_map_TR = [2, 3, 6, 7]
    v_map_BL = [8, 9, 12, 13]
    v_map_BR = [10, 11, 14, 15]
    
    # Mappings for W (C)
    # Local (r,c) where r in 0,1 and c in 0,1 for the 2x2 block
    # Global col index = base_col + c
    # W index = global_col * n + r = (base_col + c) * 2 + r
    def w_map(base_col):
        res = [0, 0, 0, 0]
        for loc in range(4):
            r = loc // 2
            c = loc % 2
            res[loc] = (base_col + c) * 2 + r
        return res
    
    w_map_L = w_map(0) # cols 0,1
    w_map_R = w_map(2) # cols 2,3
    
    col = 0
    
    # 1. A_L * B_TL -> C_L
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1
        
    # 2. A_R * B_BL -> C_L
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BL[idx], col] += val
        for idx, val in w_t:
            W[w_map_L[idx], col] += val
        col += 1
            
    # 3. A_L * B_TR -> C_R
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_L[idx], col] += val
        for idx, val in v_t:
            V[v_map_TR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1
            
    # 4. A_R * B_BR -> C_R
    for u_t, v_t, w_t in strassen:
        for idx, val in u_t:
            U[u_map_R[idx], col] += val
        for idx, val in v_t:
            V[v_map_BR[idx], col] += val
        for idx, val in w_t:
            W[w_map_R[idx], col] += val
        col += 1
        
    if time.time() - start_time > time_limit:
        raise TimeoutError("Construction took too long")
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

if __name__ == "__main__":
    res = construct()
    n, m, p = DIMENSIONS
    U = np.array(res["U"], dtype=float)
    V = np.array(res["V"], dtype=float)
    W = np.array(res["W"], dtype=float)
    A = np.arange(n * m).reshape(n, m) + 1.0
    B = np.arange(m * p).reshape(m, p) + 2.0
    C = A @ B
    Cf = np.zeros((n, p))
    for ii in range(n):
        for kk in range(p):
            acc = 0.0
            for cc in range(U.shape[1]):
                for jj in range(m):
                    acc += U[ii * m + jj, cc] * V[jj * p + kk, cc] * W[kk * n + ii, cc]
            Cf[ii, kk] = acc
    assert np.allclose(C, Cf), (C, Cf)
    print("OK rank", U.shape[1])
