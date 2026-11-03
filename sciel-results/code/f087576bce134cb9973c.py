import numpy as np
import time

DIMENSIONS = [3, 5, 5]

def _strassen_222_factors(i_offset, j_offset, k_offset, U, V, W, rank_offset):
    """
    Adds Strassen 2x2x2 factors to U, V, W.
    i_offset, j_offset, k_offset are the starting indices for i, j, k.
    U shape (3*5, rank), V shape (5*5, rank), W shape (5*3, rank).
    Note: W index is k*3 + i.
    """
    r = rank_offset
    # Strassen 7 multiplications
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # Define the 7 terms (u_vec, v_vec, w_vec)
    # u_vec is in R^4 (2x2), v_vec in R^4 (2x2), w_vec in R^4 (2x2)
    # We need to map these to the global indices.
    
    # Helper to add a term
    def add_term(u_local, v_local, w_local):
        nonlocal r
        for u_idx_local, u_val in enumerate(u_local):
            if u_val != 0:
                i_local = u_idx_local // 2
                j_local = u_idx_local % 2
                u_global = (i_offset + i_local) * 5 + (j_offset + j_local)
                U[u_global, r] += u_val
        for v_idx_local, v_val in enumerate(v_local):
            if v_val != 0:
                j_local = v_idx_local // 2
                k_local = v_idx_local % 2
                v_global = (j_offset + j_local) * 5 + (k_offset + k_local)
                V[v_global, r] += v_val
        for w_idx_local, w_val in enumerate(w_local):
            if w_val != 0:
                k_local = w_idx_local // 2
                i_local = w_idx_local % 2
                w_global = (k_offset + k_local) * 3 + (i_offset + i_local)
                W[w_global, r] += w_val
        r += 1

    # M1
    u1 = [1, 0, 0, 1]
    v1 = [1, 0, 0, 1]
    w1 = [1, 0, 0, 1]
    add_term(u1, v1, w1)
    
    # M2
    u2 = [0, 0, 1, 1]
    v2 = [1, 0, 0, 0]
    w2 = [1, 0, 0, 0]
    add_term(u2, v2, w2)
    
    # M3
    u3 = [1, 0, 0, 0]
    v3 = [0, 1, 0, -1]
    w3 = [0, 1, 0, 0]
    add_term(u3, v3, w3)
    
    # M4
    u4 = [0, 0, 0, 1]
    v4 = [0, 0, 1, -1]
    w4 = [0, 0, 1, 0]
    add_term(u4, v4, w4)
    
    # M5
    u5 = [1, 1, 0, 0]
    v5 = [0, 0, 0, 1]
    w5 = [0, 0, 0, 1]
    add_term(u5, v5, w5)
    
    # M6
    u6 = [0, 0, 1, -1]
    v6 = [1, 1, 0, 0]
    w6 = [1, 1, 0, 0]
    add_term(u6, v6, w6)
    
    # M7
    u7 = [0, 1, 0, -1]
    v7 = [0, 0, 1, 1]
    w7 = [0, 0, 1, 1]
    add_term(u7, v7, w7)
    
    return r

def _linear_122_factors(i_offset, j_offset, k_offset, U, V, W, rank_offset):
    """
    Adds factors for M_{1,2,2} (Rank 4).
    A is 1x2, B is 2x2, C is 1x2.
    i_offset is the single i index.
    """
    r = rank_offset
    def add_term(u_local, v_local, w_local):
        nonlocal r
        for u_idx_local, u_val in enumerate(u_local):
            if u_val != 0:
                j_local = u_idx_local
                u_global = i_offset * 5 + (j_offset + j_local)
                U[u_global, r] += u_val
        for v_idx_local, v_val in enumerate(v_local):
            if v_val != 0:
                j_local = v_idx_local // 2
                k_local = v_idx_local % 2
                v_global = (j_offset + j_local) * 5 + (k_offset + k_local)
                V[v_global, r] += v_val
        for w_idx_local, w_val in enumerate(w_local):
            if w_val != 0:
                k_local = w_idx_local
                w_global = (k_offset + k_local) * 3 + i_offset
                W[w_global, r] += w_val
        r += 1

    # C_0 = A_0 B_00 + A_1 B_10
    # C_1 = A_0 B_01 + A_1 B_11
    # Term 1: A_0 B_00 -> C_0
    add_term([1, 0], [1, 0, 0, 0], [1, 0])
    # Term 2: A_1 B_10 -> C_0
    add_term([0, 1], [0, 0, 1, 0], [1, 0])
    # Term 3: A_0 B_01 -> C_1
    add_term([1, 0], [0, 1, 0, 0], [0, 1])
    # Term 4: A_1 B_11 -> C_1
    add_term([0, 1], [0, 0, 0, 1], [0, 1])
    
    return r

def _linear_221_factors(i_offset, j_offset, k_offset, U, V, W, rank_offset):
    """
    Adds factors for M_{2,2,1} (Rank 4).
    A is 2x2, B is 2x1, C is 2x1.
    """
    r = rank_offset
    def add_term(u_local, v_local, w_local):
        nonlocal r
        for u_idx_local, u_val in enumerate(u_local):
            if u_val != 0:
                i_local = u_idx_local // 2
                j_local = u_idx_local % 2
                u_global = (i_offset + i_local) * 5 + (j_offset + j_local)
                U[u_global, r] += u_val
        for v_idx_local, v_val in enumerate(v_local):
            if v_val != 0:
                j_local = v_idx_local
                v_global = (j_offset + j_local) * 5 + k_offset
                V[v_global, r] += v_val
        for w_idx_local, w_val in enumerate(w_local):
            if w_val != 0:
                i_local = w_idx_local
                w_global = k_offset * 3 + (i_offset + i_local)
                W[w_global, r] += w_val
        r += 1

    # C_0 = A_00 B_0 + A_01 B_1
    # C_1 = A_10 B_0 + A_11 B_1
    add_term([1, 0, 0, 0], [1, 0], [1, 0])
    add_term([0, 1, 0, 0], [0, 1], [1, 0])
    add_term([0, 0, 1, 0], [1, 0], [0, 1])
    add_term([0, 0, 0, 1], [0, 1], [0, 1])
    
    return r

def _linear_121_factors(i_offset, j_offset, k_offset, U, V, W, rank_offset):
    """
    Adds factors for M_{1,2,1} (Rank 2).
    A is 1x2, B is 2x1, C is 1x1.
    """
    r = rank_offset
    def add_term(u_local, v_local, w_local):
        nonlocal r
        for u_idx_local, u_val in enumerate(u_local):
            if u_val != 0:
                j_local = u_idx_local
                u_global = i_offset * 5 + (j_offset + j_local)
                U[u_global, r] += u_val
        for v_idx_local, v_val in enumerate(v_local):
            if v_val != 0:
                j_local = v_idx_local
                v_global = (j_offset + j_local) * 5 + k_offset
                V[v_global, r] += v_val
        for w_idx_local, w_val in enumerate(w_local):
            if w_val != 0:
                w_global = k_offset * 3 + i_offset
                W[w_global, r] += w_val
        r += 1

    # C = A_0 B_0 + A_1 B_1
    add_term([1, 0], [1, 0], [1])
    add_term([0, 1], [0, 1], [1])
    
    return r

def _linear_331_factors(i_offset, j_offset, k_offset, U, V, W, rank_offset):
    """
    Adds factors for M_{3,3,1} (Rank 9).
    A is 3x3, B is 3x1, C is 3x1.
    """
    r = rank_offset
    def add_term(u_local, v_local, w_local):
        nonlocal r
        for u_idx_local, u_val in enumerate(u_local):
            if u_val != 0:
                i_local = u_idx_local // 3
                j_local = u_idx_local % 3
                u_global = (i_offset + i_local) * 5 + (j_offset + j_local)
                U[u_global, r] += u_val
        for v_idx_local, v_val in enumerate(v_local):
            if v_val != 0:
                j_local = v_idx_local
                v_global = (j_offset + j_local) * 5 + k_offset
                V[v_global, r] += v_val
        for w_idx_local, w_val in enumerate(w_local):
            if w_val != 0:
                i_local = w_idx_local
                w_global = k_offset * 3 + (i_offset + i_local)
                W[w_global, r] += w_val
        r += 1

    # C_i = sum_j A_ij B_j
    for i in range(3):
        for j in range(3):
            u = [0]*9
            u[i*3+j] = 1
            v = [0]*3
            v[j] = 1
            w = [0]*3
            w[i] = 1
            add_term(u, v, w)
            
    return r

def _schoolbook_333_factors(i_offset, j_offset, k_offset, U, V, W, rank_offset):
    """
    Adds factors for M_{3,3,3} (Rank 27).
    """
    r = rank_offset
    for i in range(3):
        for j in range(3):
            for k in range(3):
                u_global = (i_offset + i) * 5 + (j_offset + j)
                v_global = (j_offset + j) * 5 + (k_offset + k)
                w_global = (k_offset + k) * 3 + (i_offset + i)
                U[u_global, r] = 1
                V[v_global, r] = 1
                W[w_global, r] = 1
                r += 1
    return r

def construct():
    n, m, p = DIMENSIONS
    nm = n * m
    mp = m * p
    pn = p * n
    
    # Total rank 73
    total_rank = 73
    U = np.zeros((nm, total_rank))
    V = np.zeros((mp, total_rank))
    W = np.zeros((pn, total_rank))
    
    r = 0
    
    # 1. M_{3,2,2} for i[0:3], j[0:2], k[0:2]
    # Split into M_{2,2,2} (i[0:2]) and M_{1,2,2} (i[2:3])
    r = _strassen_222_factors(0, 0, 0, U, V, W, r)
    r = _linear_122_factors(2, 0, 0, U, V, W, r)
    
    # 2. M_{3,2,3} for i[0:3], j[0:2], k[2:5]
    # Split into M_{2,2,2} (i[0:2], k[2:4]), M_{2,2,1} (i[0:2], k[4:5]),
    #          M_{1,2,2} (i[2:3], k[2:4]), M_{1,2,1} (i[2:3], k[4:5])
    r = _strassen_222_factors(0, 0, 2, U, V, W, r)
    r = _linear_221_factors(0, 0, 4, U, V, W, r)
    r = _linear_122_factors(2, 0, 2, U, V, W, r)
    r = _linear_121_factors(2, 0, 4, U, V, W, r)
    
    # 3. M_{3,3,2} for i[0:3], j[2:5], k[0:2]
    # Split into M_{3,3,1} (k[0:1]) and M_{3,3,1} (k[1:2])
    r = _linear_331_factors(0, 2, 0, U, V, W, r)
    r = _linear_331_factors(0, 2, 1, U, V, W, r)
    
    # 4. M_{3,3,3} for i[0:3], j[2:5], k[2:5]
    r = _schoolbook_333_factors(0, 2, 2, U, V, W, r)
    
    # Verify r == 73
    # print(f"Final rank: {r}")
    
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

