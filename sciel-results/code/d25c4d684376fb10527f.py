import numpy as np

DIMENSIONS = [2, 2, 11]

def construct():
    n, m, p = DIMENSIONS
    
    num_blocks_2x2 = p // 2
    rem_cols = p % 2
    
    rank = num_blocks_2x2 * 7 + rem_cols * 4
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    current_rank = 0
    
    def add_term(u_vec, v_vec, w_vec):
        nonlocal current_rank
        U[:, current_rank] = u_vec
        V[:, current_rank] = v_vec
        W[:, current_rank] = w_vec
        current_rank += 1

    # Strassen 2x2
    # A indices: 00, 01, 10, 11 -> 0, 1, 2, 3
    # B indices (local): 00, 01, 10, 11
    # C indices (local): 00, 01, 10, 11
    
    u_coeffs = [
        [1, 0, 0, 1], # M1: a+d
        [0, 0, 1, 1], # M2: c+d
        [1, 0, 0, 0], # M3: a
        [0, 0, 0, 1], # M4: d
        [1, 1, 0, 0], # M5: a+b
        [0, 0, 1, -1], # M6: c-a
        [0, 1, 0, -1]  # M7: b-d
    ]
    
    # V coefficients for local B indices 00, 01, 10, 11
    # e=00, f=01, g=10, h=11
    v_coeffs = [
        [1, 0, 0, 1], # M1: e+h
        [1, 0, 0, 0], # M2: e
        [0, 1, 0, -1], # M3: f-h
        [0, 0, 1, -1], # M4: g-e
        [0, 0, 0, 1], # M5: h
        [1, 1, 0, 0], # M6: e+f
        [0, 0, 1, 1]  # M7: g+h
    ]
    
    # W coefficients for local C indices 00, 01, 10, 11
    w_coeffs = [
        [1, 0, 0, 1], # M1: C00 + C11
        [0, 0, 1, -1], # M2: C10 - C11
        [0, 1, 0, 1], # M3: C01 + C11
        [1, 0, 1, 0], # M4: C00 + C10
        [-1, 1, 0, 0], # M5: -C00 + C01
        [0, 0, 0, 1], # M6: C11
        [1, 0, 0, 0]  # M7: C00
    ]
    
    for block_idx in range(num_blocks_2x2):
        k0 = block_idx * 2
        k1 = block_idx * 2 + 1
        
        for m_idx in range(7):
            u_full = [0] * (n * m)
            u_full[0] = u_coeffs[m_idx][0]
            u_full[1] = u_coeffs[m_idx][1]
            u_full[2] = u_coeffs[m_idx][2]
            u_full[3] = u_coeffs[m_idx][3]
            
            v_full = [0] * (m * p)
            # Map local B indices to global V indices
            # local 00 -> j=0, k=k0 -> V[0*p + k0]
            # local 01 -> j=0, k=k1 -> V[0*p + k1]
            # local 10 -> j=1, k=k0 -> V[1*p + k0]
            # local 11 -> j=1, k=k1 -> V[1*p + k1]
            
            v_full[0 * p + k0] = v_coeffs[m_idx][0]
            v_full[0 * p + k1] = v_coeffs[m_idx][1]
            v_full[1 * p + k0] = v_coeffs[m_idx][2]
            v_full[1 * p + k1] = v_coeffs[m_idx][3]
            
            w_full = [0] * (p * n)
            # Map local C indices to global W indices
            # local 00 -> i=0, k=k0 -> W[k0*n + 0]
            # local 01 -> i=0, k=k1 -> W[k1*n + 0]
            # local 10 -> i=1, k=k0 -> W[k0*n + 1]
            # local 11 -> i=1, k=k1 -> W[k1*n + 1]
            
            w_full[k0 * n + 0] = w_coeffs[m_idx][0]
            w_full[k1 * n + 0] = w_coeffs[m_idx][1]
            w_full[k0 * n + 1] = w_coeffs[m_idx][2]
            w_full[k1 * n + 1] = w_coeffs[m_idx][3]
            
            add_term(u_full, v_full, w_full)

    if rem_cols > 0:
        k0 = num_blocks_2x2 * 2
        
        # 2x1 terms
        # M1: a00 * b0 -> C00
        u = [1, 0, 0, 0]
        v = [0] * (m * p)
        v[0 * p + k0] = 1
        w = [0] * (p * n)
        w[k0 * n + 0] = 1
        add_term(u, v, w)
        
        # M2: a01 * b1 -> C00
        u = [0, 1, 0, 0]
        v = [0] * (m * p)
        v[1 * p + k0] = 1
        w = [0] * (p * n)
        w[k0 * n + 0] = 1
        add_term(u, v, w)
        
        # M3: a10 * b0 -> C10
        u = [0, 0, 1, 0]
        v = [0] * (m * p)
        v[0 * p + k0] = 1
        w = [0] * (p * n)
        w[k0 * n + 1] = 1
        add_term(u, v, w)
        
        # M4: a11 * b1 -> C10
        u = [0, 0, 0, 1]
        v = [0] * (m * p)
        v[1 * p + k0] = 1
        w = [0] * (p * n)
        w[k0 * n + 1] = 1
        add_term(u, v, w)

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
