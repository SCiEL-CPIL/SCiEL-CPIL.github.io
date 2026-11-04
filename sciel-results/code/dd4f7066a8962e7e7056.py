import numpy as np

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 77
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    def strassen_2x2_vecs():
        return [
            ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
            ([0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, -1]),
            ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 0]),
            ([0, 0, 0, 1], [0, 0, 1, -1], [1, 0, 1, 0]),
            ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 0, 0, 0]),
            ([1, 0, -1, 0], [1, 1, 0, 0], [0, 0, 0, 1]),
            ([0, 1, 0, 1], [0, 0, 1, -1], [1, 0, 0, 0]),
        ]

    s_vecs = strassen_2x2_vecs()
    
    def get_block_indices_A(i_off, j_off):
        idxs = []
        for di in range(2):
            for dj in range(2):
                i = i_off * 2 + di
                j = j_off * 2 + dj
                idxs.append(i * m + j)
        return idxs

    def get_B_left_indices(j_off, k_off):
        idxs = []
        for dj in range(2):
            for dk in range(2):
                j = j_off * 2 + dj
                k = k_off * 2 + dk
                idxs.append(j * p + k)
        return idxs

    def get_C_left_indices(i_off, k_off):
        idxs = []
        for di in range(2):
            for dk in range(2):
                i = i_off * 2 + di
                k = k_off * 2 + dk
                idxs.append(k * n + i)
        return idxs

    strassen_steps = [
        { 'U': [(0, 1), (3, 1)], 'V': [(0, 1), (3, 1)], 'W': [(0, 1), (3, 1)] },
        { 'U': [(2, 1), (3, 1)], 'V': [(0, 1)], 'W': [(2, 1), (3, -1)] },
        { 'U': [(0, 1)], 'V': [(1, 1), (3, -1)], 'W': [(1, 1)] },
        { 'U': [(3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1), (2, 1)] },
        { 'U': [(0, 1), (1, 1)], 'V': [(3, 1)], 'W': [(0, -1)] },
        { 'U': [(2, 1), (0, -1)], 'V': [(0, 1), (1, 1)], 'W': [(3, 1)] },
        { 'U': [(1, 1), (3, 1)], 'V': [(2, 1), (0, -1)], 'W': [(0, 1)] },
    ]
    
    col = 0
    for step in strassen_steps:
        for s in range(7):
            u_vec, v_vec, w_vec = s_vecs[s]
            for b_idx, coeff in step['U']:
                val = u_vec[b_idx] * coeff
                if val != 0:
                    for idx in get_block_indices_A(b_idx // 2, b_idx % 2):
                        U[idx, col] += val
            for b_idx, coeff in step['V']:
                val = v_vec[b_idx] * coeff
                if val != 0:
                    for idx in get_B_left_indices(b_idx // 2, b_idx % 2):
                        V[idx, col] += val
            for b_idx, coeff in step['W']:
                val = w_vec[b_idx] * coeff
                if val != 0:
                    for idx in get_C_left_indices(b_idx // 2, b_idx % 2):
                        W[idx, col] += val
            col += 1

    def get_B_right_indices(j_off):
        idxs = []
        for dj in range(2):
            for dk in range(2):
                j = j_off * 2 + dj
                k = 4 + dk
                idxs.append(j * p + k)
        return idxs

    def get_C_right_indices(i_off):
        idxs = []
        for di in range(2):
            for dk in range(2):
                i = i_off * 2 + di
                k = 4 + dk
                idxs.append(k * n + i)
        return idxs

    products = [
        (0, 0, 0),
        (1, 1, 0),
        (2, 0, 1),
        (3, 1, 1),
    ]
    
    for a_b, br_b, cr_b in products:
        for s in range(7):
            u_vec, v_vec, w_vec = s_vecs[s]
            for i_local in range(4):
                val = u_vec[i_local]
                if val != 0:
                    i_off = a_b // 2
                    j_off = a_b % 2
                    di = i_local // 2
                    dj = i_local % 2
                    i = i_off * 2 + di
                    j = j_off * 2 + dj
                    U[i * m + j, col] += val
            
            for i_local in range(4):
                val = v_vec[i_local]
                if val != 0:
                    j_off = br_b
                    dj = i_local // 2
                    dk = i_local % 2
                    j = j_off * 2 + dj
                    k = 4 + dk
                    V[j * p + k, col] += val
            
            for i_local in range(4):
                val = w_vec[i_local]
                if val != 0:
                    i_off = cr_b
                    di = i_local // 2
                    dk = i_local % 2
                    i = i_off * 2 + di
                    k = 4 + dk
                    W[k * n + i, col] += val
            
            col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
