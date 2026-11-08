import numpy as np
import time

def construct():
    start_time = time.time()
    
    n, m, p = 4, 4, 7
    terms = []
    
    def add_term(u, v, w):
        terms.append((u, v, w))
    
    # Build the 49-term 4x4*4x4*4x4 head
    # Using recursive Strassen: 7 outer terms, each with 7 inner terms
    outer = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    inner = [
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1),(3,-1)]),
        ([(0,1)], [(1,1),(3,-1)], [(1,1),(3,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(1,1)]),
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]
    
    def get_global_idx(block_idx, local_idx, is_A):
        br = block_idx // 2
        bc = block_idx % 2
        lr = local_idx // 2
        lc = local_idx % 2
        row = br * 2 + lr
        col = bc * 2 + lc
        if is_A:
            return row * m + col
        else:
            return row * p + col
    
    def get_w_idx(block_idx_c, local_idx_c):
        br = block_idx_c // 2
        bc = block_idx_c % 2
        lr = local_idx_c // 2
        lc = local_idx_c % 2
        i = br * 2 + lr
        k = bc * 2 + lc
        return k * n + i
    
    # Head: cols 0-3
    for oa, ob, oc in outer:
        for ia, ib, ic in inner:
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            for a_blk, a_coeff in oa:
                for x_loc, x_coeff in ia:
                    idx = get_global_idx(a_blk, x_loc, True)
                    u[idx] += a_coeff * x_coeff
            for b_blk, b_coeff in ob:
                for y_loc, y_coeff in ib:
                    idx = get_global_idx(b_blk, y_loc, False)
                    v[idx] += b_coeff * y_coeff
            for c_blk, c_coeff in oc:
                for z_loc, z_coeff in ic:
                    idx = get_w_idx(c_blk, z_loc)
                    w[idx] += c_coeff * z_coeff
            add_term(u, v, w)
    
    # Mid: cols 4-5 using 4 outer Strassen terms
    outer_mid = [
        ([(0,1),(3,1)], [(0,1)], [(0,1)]),
        ([(2,1),(3,1)], [(0,1)], [(2,1)]),
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(2,1)]),
        ([(1,1),(3,-1)], [(2,1)], [(0,1)]),
    ]
    
    def get_global_idx_mid(block_idx, local_idx, is_A, k_off=4):
        br = block_idx // 2
        bc = block_idx % 2
        lr = local_idx // 2
        lc = local_idx % 2
        row = br * 2 + lr
        col = bc * 2 + lc
        if is_A:
            return row * m + col
        else:
            return row * p + (col + k_off)
    
    def get_w_idx_mid(block_idx_c, local_idx_c, k_off=4):
        br = block_idx_c // 2
        bc = block_idx_c % 2
        lr = local_idx_c // 2
        lc = local_idx_c % 2
        i = br * 2 + lr
        k = bc * 2 + lc
        return (k + k_off) * n + i
    
    for oa, ob, oc in outer_mid:
        for ia, ib, ic in inner:
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            for a_blk, a_coeff in oa:
                for x_loc, x_coeff in ia:
                    idx = get_global_idx_mid(a_blk, x_loc, True)
                    u[idx] += a_coeff * x_coeff
            for b_blk, b_coeff in ob:
                for y_loc, y_coeff in ib:
                    idx = get_global_idx_mid(b_blk, y_loc, False)
                    v[idx] += b_coeff * y_coeff
            for c_blk, c_coeff in oc:
                for z_loc, z_coeff in ic:
                    idx = get_w_idx_mid(c_blk, z_loc)
                    w[idx] += c_coeff * z_coeff
            add_term(u, v, w)
    
    # Tail: col 6 using schoolbook
    for i in range(n):
        for j in range(m):
            u = [0.0] * (n * m)
            v = [0.0] * (m * p)
            w = [0.0] * (p * n)
            u[i * m + j] = 1.0
            v[j * p + 6] = 1.0
            w[6 * n + i] = 1.0
            add_term(u, v, w)
    
    elapsed = time.time() - start_time
    if elapsed > 130:
        return {"U": [], "V": [], "W": []}
    
    rank = len(terms)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            U[r][c] = u[r]
        for r in range(m * p):
            V[r][c] = v[r]
        for r in range(p * n):
            W[r][c] = w[r]
    
    return {"U": U, "V": V, "W": W}
