import time
import numpy as np

DIMENSIONS = [4, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 4, 4, 7
    total_rank = 97
    
    U = np.zeros((n*m, total_rank))  # (16, 97)
    V = np.zeros((m*p, total_rank))  # (28, 97)
    W = np.zeros((p*n, total_rank))  # (28, 97)
    
    def a_idx(i, j):
        return i * 4 + j
    
    def b_idx(j, k):
        return j * 7 + k
    
    def c_idx(i, k):
        return k * 4 + i
    
    # 7 M terms for 4x4x4 Strassen
    m1_left = [
        {a_idx(0,0): 1, a_idx(2,2): 1},
        {a_idx(0,1): 1, a_idx(2,3): 1},
        {a_idx(1,0): 1, a_idx(3,2): 1},
        {a_idx(1,1): 1, a_idx(3,3): 1},
    ]
    m1_right = [
        {b_idx(0,0): 1, b_idx(2,2): 1},
        {b_idx(0,1): 1, b_idx(2,3): 1},
        {b_idx(1,0): 1, b_idx(3,2): 1},
        {b_idx(1,1): 1, b_idx(3,3): 1},
    ]
    m1_out = [
        [(c_idx(0,0), 1), (c_idx(2,2), 1)],
        [(c_idx(1,0), 1), (c_idx(3,2), 1)],
        [(c_idx(0,1), 1), (c_idx(2,3), 1)],
        [(c_idx(1,1), 1), (c_idx(3,3), 1)],
    ]
    
    m2_left = [
        {a_idx(2,0): 1, a_idx(2,2): 1},
        {a_idx(2,1): 1, a_idx(2,3): 1},
        {a_idx(3,0): 1, a_idx(3,2): 1},
        {a_idx(3,1): 1, a_idx(3,3): 1},
    ]
    m2_right = [
        {b_idx(0,0): 1},
        {b_idx(0,1): 1},
        {b_idx(1,0): 1},
        {b_idx(1,1): 1},
    ]
    m2_out = [
        [(c_idx(2,0), 1), (c_idx(2,2), -1)],
        [(c_idx(3,0), 1), (c_idx(3,2), -1)],
        [(c_idx(2,1), 1), (c_idx(2,3), -1)],
        [(c_idx(3,1), 1), (c_idx(3,3), -1)],
    ]
    
    m3_left = [
        {a_idx(0,0): 1},
        {a_idx(0,1): 1},
        {a_idx(1,0): 1},
        {a_idx(1,1): 1},
    ]
    m3_right = [
        {b_idx(0,2): 1, b_idx(2,2): -1},
        {b_idx(0,3): 1, b_idx(2,3): -1},
        {b_idx(1,2): 1, b_idx(3,2): -1},
        {b_idx(1,3): 1, b_idx(3,3): -1},
    ]
    m3_out = [
        [(c_idx(0,2), 1), (c_idx(2,2), 1)],
        [(c_idx(1,2), 1), (c_idx(3,2), 1)],
        [(c_idx(0,3), 1), (c_idx(2,3), 1)],
        [(c_idx(1,3), 1), (c_idx(3,3), 1)],
    ]
    
    m4_left = [
        {a_idx(2,2): 1},
        {a_idx(2,3): 1},
        {a_idx(3,2): 1},
        {a_idx(3,3): 1},
    ]
    m4_right = [
        {b_idx(2,0): 1, b_idx(0,0): -1},
        {b_idx(2,1): 1, b_idx(0,1): -1},
        {b_idx(3,0): 1, b_idx(1,0): -1},
        {b_idx(3,1): 1, b_idx(1,1): -1},
    ]
    m4_out = [
        [(c_idx(0,0), 1), (c_idx(2,0), 1)],
        [(c_idx(1,0), 1), (c_idx(3,0), 1)],
        [(c_idx(0,1), 1), (c_idx(2,1), 1)],
        [(c_idx(1,1), 1), (c_idx(3,1), 1)],
    ]
    
    m5_left = [
        {a_idx(0,0): 1, a_idx(0,2): 1},
        {a_idx(0,1): 1, a_idx(0,3): 1},
        {a_idx(1,0): 1, a_idx(1,2): 1},
        {a_idx(1,1): 1, a_idx(1,3): 1},
    ]
    m5_right = [
        {b_idx(2,2): 1},
        {b_idx(2,3): 1},
        {b_idx(3,2): 1},
        {b_idx(3,3): 1},
    ]
    m5_out = [
        [(c_idx(0,0), -1), (c_idx(0,2), 1)],
        [(c_idx(1,0), -1), (c_idx(1,2), 1)],
        [(c_idx(0,1), -1), (c_idx(0,3), 1)],
        [(c_idx(1,1), -1), (c_idx(1,3), 1)],
    ]
    
    m6_left = [
        {a_idx(2,0): 1, a_idx(0,0): -1},
        {a_idx(2,1): 1, a_idx(0,1): -1},
        {a_idx(3,0): 1, a_idx(1,0): -1},
        {a_idx(3,1): 1, a_idx(1,1): -1},
    ]
    m6_right = [
        {b_idx(0,0): 1, b_idx(0,2): 1},
        {b_idx(0,1): 1, b_idx(0,3): 1},
        {b_idx(1,0): 1, b_idx(1,2): 1},
        {b_idx(1,1): 1, b_idx(1,3): 1},
    ]
    m6_out = [
        [(c_idx(2,2), 1)],
        [(c_idx(3,2), 1)],
        [(c_idx(2,3), 1)],
        [(c_idx(3,3), 1)],
    ]
    
    m7_left = [
        {a_idx(0,2): 1, a_idx(2,2): -1},
        {a_idx(0,3): 1, a_idx(2,3): -1},
        {a_idx(1,2): 1, a_idx(3,2): -1},
        {a_idx(1,3): 1, a_idx(3,3): -1},
    ]
    m7_right = [
        {b_idx(2,0): 1, b_idx(2,2): 1},
        {b_idx(2,1): 1, b_idx(2,3): 1},
        {b_idx(3,0): 1, b_idx(3,2): 1},
        {b_idx(3,1): 1, b_idx(3,3): 1},
    ]
    m7_out = [
        [(c_idx(0,0), 1)],
        [(c_idx(1,0), 1)],
        [(c_idx(0,1), 1)],
        [(c_idx(1,1), 1)],
    ]
    
    # Strassen 2x2x2 terms
    strassen_terms = [
        ([1,0,0,1], [1,0,0,1], [(0,1),(3,1)]),
        ([0,0,1,1], [1,0,0,0], [(2,1),(3,-1)]),
        ([1,0,0,0], [0,1,0,-1], [(1,1),(3,1)]),
        ([0,0,0,1], [0,0,1,-1], [(0,1),(2,1)]),
        ([1,1,0,0], [0,0,0,1], [(0,-1),(1,1)]),
        ([0,0,1,-1], [1,1,0,0], [(3,1)]),
        ([0,1,0,-1], [0,0,1,1], [(0,1)]),
    ]
    
    col = 0
    for m_left, m_right, m_out in [(m1_left, m1_right, m1_out),
                                     (m2_left, m2_right, m2_out),
                                     (m3_left, m3_right, m3_out),
                                     (m4_left, m4_right, m4_out),
                                     (m5_left, m5_right, m5_out),
                                     (m6_left, m6_right, m6_out),
                                     (m7_left, m7_right, m7_out)]:
        for lc, rc, out_list in strassen_terms:
            for xi in range(4):
                if lc[xi] != 0:
                    for u_idx, coef in m_left[xi].items():
                        U[u_idx, col] += lc[xi] * coef
            for yj in range(4):
                if rc[yj] != 0:
                    for v_idx, coef in m_right[yj].items():
                        V[v_idx, col] += rc[yj] * coef
            for (zi, zcoef) in out_list:
                for (w_idx, wcoef) in m_out[zi]:
                    W[w_idx, col] += zcoef * wcoef
            col += 1
    
    # Schoolbook 4x4x3 part
    for i in range(4):
        for j in range(4):
            for k in range(4, 7):
                U[a_idx(i,j), col] = 1
                V[b_idx(j,k), col] = 1
                W[c_idx(i,k), col] = 1
                col += 1
    
    assert col == 97, f"Expected col=97, got {col}"
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
