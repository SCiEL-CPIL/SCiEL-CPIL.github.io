import numpy as np
import time

DIMENSIONS = [3, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    total_rank = 78
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    current_rank = 0
    
    def add_schoolbook_term(u_idx, v_idx, w_idx, val):
        nonlocal current_rank
        if current_rank >= total_rank:
            return
        U[u_idx][current_rank] += val
        V[v_idx][current_rank] += val
        W[w_idx][current_rank] += val
        current_rank += 1
        
    def add_strassen_term(u_coeffs, v_coeffs, w_coeffs):
        # u_coeffs: dict {idx: val}
        # v_coeffs: dict {idx: val}
        # w_coeffs: dict {idx: val}
        nonlocal current_rank
        if current_rank >= total_rank:
            return
        for idx, val in u_coeffs.items():
            U[idx][current_rank] += val
        for idx, val in v_coeffs.items():
            V[idx][current_rank] += val
        for idx, val in w_coeffs.items():
            W[idx][current_rank] += val
        current_rank += 1

    def add_schoolbook_block(n_start, n_end, m_start, m_end, p_start, p_end):
        for i in range(n_start, n_end):
            for j in range(m_start, m_end):
                for k in range(p_start, p_end):
                    add_schoolbook_term(i * m + j, j * p + k, k * n + i, 1)

    def add_strassen_block(n_start, m_start, p_start):
        # Local indices 0,1
        # Global indices
        n0, n1 = n_start, n_start + 1
        m0, m1 = m_start, m_start + 1
        p0, p1 = p_start, p_start + 1
        
        u00 = n0 * m + m0
        u01 = n0 * m + m1
        u10 = n1 * m + m0
        u11 = n1 * m + m1
        
        v00 = m0 * p + p0
        v01 = m0 * p + p1
        v10 = m1 * p + p0
        v11 = m1 * p + p1
        
        w00 = p0 * n + n0
        w01 = p1 * n + n0
        w10 = p0 * n + n1
        w11 = p1 * n + n1
        
        # 1
        add_strassen_term({u00: 1, u11: 1}, {v00: 1, v11: 1}, {w00: 1, w11: 1})
        # 2
        add_strassen_term({u10: 1, u11: 1}, {v00: 1}, {w10: 1, w11: 1})
        # 3
        add_strassen_term({u00: 1}, {v10: 1, v11: -1}, {w01: 1, w11: 1})
        # 4
        add_strassen_term({u11: 1}, {v01: 1, v00: -1}, {w00: 1, w10: 1})
        # 5
        add_strassen_term({u00: 1, u01: 1}, {v11: 1}, {w00: -1, w01: 1})
        # 6
        add_strassen_term({u10: 1, u00: -1}, {v00: 1, v01: 1}, {w11: 1})
        # 7
        add_strassen_term({u01: 1, u11: -1}, {v10: 1, v11: 1}, {w00: 1, w01: -1})

    # Block 1: n=0, m in [0,2), p in [0,2)
    add_schoolbook_block(0, 1, 0, 2, 0, 2)
    # Block 2: n=0, m in [0,2), p in [2,6)
    add_schoolbook_block(0, 1, 0, 2, 2, 6)
    # Block 3: n in [1,3), m in [0,2), p in [0,2)
    add_strassen_block(1, 0, 0)
    # Block 4: n in [1,3), m in [0,2), p in [2,6)
    # Split p in [2,6) into [2,4) and [4,6)
    add_strassen_block(1, 0, 2)
    add_strassen_block(1, 0, 4)
    
    # Block 5: n=0, m in [2,5), p in [0,2)
    add_schoolbook_block(0, 1, 2, 5, 0, 2)
    # Block 6: n=0, m in [2,5), p in [2,6)
    add_schoolbook_block(0, 1, 2, 5, 2, 6)
    # Block 7: n in [1,3), m in [2,5), p in [0,2)
    # Split m in [2,5) into [2,3) and [3,5)
    # 7a: m in [2,3)
    add_schoolbook_block(1, 3, 2, 3, 0, 2)
    # 7b: m in [3,5)
    add_strassen_block(1, 3, 0)
    
    # Block 8: n in [1,3), m in [2,5), p in [2,6)
    # Split m in [2,5) into [2,3) and [3,5)
    # 8a: m in [2,3)
    add_schoolbook_block(1, 3, 2, 3, 2, 6)
    # 8b: m in [3,5)
    # Split p in [2,6) into [2,4) and [4,6)
    add_strassen_block(1, 3, 2)
    add_strassen_block(1, 3, 4)

    return {"U": U, "V": V, "W": W}
