import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Rank 35 construction
    R = 35
    U = np.zeros((dim_u, R))
    V = np.zeros((dim_v, R))
    W = np.zeros((dim_w, R))
    
    col = 0
    
    # Strassen 2x2x2 block
    # Local indices for u: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Global U index: i*3 + j
    # 0 -> 0*3+0 = 0
    # 1 -> 0*3+1 = 1
    # 2 -> 1*3+0 = 3
    # 3 -> 1*3+1 = 4
    u_map = [0, 1, 3, 4]
    
    # Local indices for v: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Global V index: j*4 + k
    # 0 -> 0*4+0 = 0
    # 1 -> 0*4+1 = 1
    # 2 -> 1*4+0 = 4
    # 3 -> 1*4+1 = 5
    v_map = [0, 1, 4, 5]
    
    # Local indices for w: (0,0)->0, (0,1)->1, (1,0)->2, (1,1)->3
    # Global W index: k*3 + i
    # 0 -> 0*3+0 = 0
    # 1 -> 0*3+1 = 1
    # 2 -> 1*3+0 = 3
    # 3 -> 1*3+1 = 4
    w_map = [0, 1, 3, 4]
    
    # Strassen vectors
    u_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, -1, 0]
    ]
    v_vecs = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, -1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w_vecs = [
        [1, 0, 0, 1],
        [0, 0, 1, -1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [-1, 1, 0, 0],
        [0, 0, 0, 0],
        [-1, 0, 0, 1]
    ]
    
    for r in range(7):
        for lu in range(4):
            if u_vecs[r][lu] != 0:
                U[u_map[lu], col] += u_vecs[r][lu]
        for lv in range(4):
            if v_vecs[r][lv] != 0:
                V[v_map[lv], col] += v_vecs[r][lv]
        for lw in range(4):
            if w_vecs[r][lw] != 0:
                W[w_map[lw], col] += w_vecs[r][lw]
        col += 1
        
    # Schoolbook for the rest
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if i < 2 and j < 2 and k < 2:
                    continue
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                U[u_idx, col] = 1
                V[v_idx, col] = 1
                W[w_idx, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
