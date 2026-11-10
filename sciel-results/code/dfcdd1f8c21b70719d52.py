import numpy as np
import time
import random

def nikodym_set() -> np.ndarray:
    p = 7
    k = 3 # alpha^2 = 3
    
    elems = []
    for a in range(p):
        for b in range(p):
            elems.append((a, b))
            
    idx_map = {e: i for i, e in enumerate(elems)}
    
    add_tbl = np.zeros((49, 49), dtype=np.int64)
    mul_tbl = np.zeros((49, 49), dtype=np.int64)
    
    for i in range(49):
        a1, b1 = elems[i]
        for j in range(49):
            a2, b2 = elems[j]
            res_a = (a1 + a2) % p
            res_b = (b1 + b2) % p
            add_tbl[i, j] = idx_map[(res_a, res_b)]
            
            res_a = (a1 * a2 + k * b1 * b2) % p
            res_b = (a1 * b2 + a2 * b1) % p
            mul_tbl[i, j] = idx_map[(res_a, res_b)]
            
    num_points = 49 * 49
    one_idx = idx_map[(1, 0)]
    zero_idx = idx_map[(0, 0)]
    
    neighbors = [[None for _ in range(50)] for _ in range(num_points)]
    
    for x_idx in range(num_points):
        x_coord = x_idx // 49
        y_coord = x_idx % 49
        
        for v_idx in range(50):
            if v_idx < 49:
                m_idx = v_idx
                dx_idx = one_idx
                dy_idx = m_idx
            else:
                dx_idx = zero_idx
                dy_idx = one_idx
            
            line_points = []
            for t_idx in range(49):
                tx = add_tbl[x_coord, mul_tbl[t_idx, dx_idx]]
                ty = add_tbl[y_coord, mul_tbl[t_idx, dy_idx]]
                pt_idx = tx * 49 + ty
                line_points.append(pt_idx)
            
            neigh = [pt for pt in line_points if pt != x_idx]
            neighbors[x_idx][v_idx] = neigh
            
    best_S_size = 0
    best_S = None
    start_time = time.time()
    deadline = start_time + 590
    
    # Use a local random instance to avoid interfering with global state?
    # Or just use random.
    
    while time.time() < deadline:
        in_S = np.ones(num_points, dtype=bool)
        C = np.full((num_points, 50), 48, dtype=np.int64)
        zero_count = np.zeros(num_points, dtype=np.int64)
        
        order = list(range(num_points))
        random.shuffle(order)
        
        changed = True
        while changed:
            changed = False
            for x in order:
                if not in_S[x]:
                    continue
                if zero_count[x] == 0:
                    in_S[x] = False
                    changed = True
                    for v in range(50):
                        for y in neighbors[x][v]:
                            C[y][v] -= 1
                            if C[y][v] == 0:
                                zero_count[y] += 1
        
        s_size = np.sum(in_S)
        if s_size > best_S_size:
            best_S_size = s_size
            best_S = in_S.copy()
            
    n_points = []
    for i in range(num_points):
        if not best_S[i]:
            x_coord = i // 49
            y_coord = i % 49
            xa, xb = elems[x_coord]
            ya, yb = elems[y_coord]
            n_points.append([[xa, xb], [ya, yb]])
            
    return np.array(n_points, dtype=np.int64)
