import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    
    def a_vec(i, j):
        v = np.zeros(16)
        v[i*4+j] = 1.0
        return v
    
    def b_vec(j, k):
        v = np.zeros(16)
        v[j*4+k] = 1.0
        return v
    
    # Build 4x4 x 4x4 Strassen-based rank-49 decomposition
    A11 = np.array([[a_vec(0,0), a_vec(0,1)], [a_vec(1,0), a_vec(1,1)]])
    A12 = np.array([[a_vec(0,2), a_vec(0,3)], [a_vec(1,2), a_vec(1,3)]])
    A21 = np.array([[a_vec(2,0), a_vec(2,1)], [a_vec(3,0), a_vec(3,1)]])
    A22 = np.array([[a_vec(2,2), a_vec(2,3)], [a_vec(3,2), a_vec(3,3)]])
    
    B11 = np.array([[b_vec(0,0), b_vec(0,1)], [b_vec(1,0), b_vec(1,1)]])
    B12 = np.array([[b_vec(0,2), b_vec(0,3)], [b_vec(1,2), b_vec(1,3)]])
    B21 = np.array([[b_vec(2,0), b_vec(2,1)], [b_vec(3,0), b_vec(3,1)]])
    B22 = np.array([[b_vec(2,2), b_vec(2,3)], [b_vec(3,2), b_vec(3,3)]])
    
    top_products = [
        (A11 + A22, B11 + B22),
        (A21 + A22, B11),
        (A11, B12 - B22),
        (A22, B21 - B11),
        (A11 + A12, B22),
        (A21 - A11, B11 + B12),
        (A12 - A22, B21 + B22),
    ]
    
    sub_to_csub = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    
    m_to_block = [
        [(0,0,1), (1,1,1)],
        [(1,0,1), (1,1,-1)],
        [(0,1,1), (1,1,1)],
        [(0,0,1), (1,0,1)],
        [(0,0,-1), (0,1,1)],
        [(1,1,1)],
        [(0,0,1)],
    ]
    
    all_terms = []
    for m_idx, (P, Q) in enumerate(top_products):
        p00, p01 = P[0,0], P[0,1]
        p10, p11 = P[1,0], P[1,1]
        q00, q01 = Q[0,0], Q[0,1]
        q10, q11 = Q[1,0], Q[1,1]
        
        s = [
            (p00+p11, q00+q11),
            (p10+p11, q00),
            (p00, q01-q11),
            (p11, q10-q00),
            (p00+p01, q11),
            (p10-p00, q00+q01),
            (p01-p11, q10+q11),
        ]
        
        for s_idx, (a_v, b_v) in enumerate(s):
            for (br, bc, sign_block) in m_to_block[m_idx]:
                for (sr, sc, sign_sub) in sub_to_csub[s_idx]:
                    c_row = br*2 + sr
                    c_col = bc*2 + sc
                    c_coeff = sign_block * sign_sub
                    c_vec = np.zeros(16)
                    c_vec[c_row*4+c_col] = c_coeff
                    all_terms.append((a_v, b_v, c_vec))
    
    # Build 4x4 U, V, W
    U_4x4 = np.zeros((16, 49))
    V_4x4 = np.zeros((16, 49))
    W_4x4 = np.zeros((16, 49))
    
    for c_idx, (a_v, b_v, c_v) in enumerate(all_terms):
        U_4x4[:, c_idx] = a_v
        V_4x4[:, c_idx] = b_v
        W_4x4[:, c_idx] = c_v
    
    # Build 4x4 x 4x8 decomposition
    rank = 98
    U = np.zeros((16, rank))
    V = np.zeros((32, rank))
    W = np.zeros((32, rank))
    
    U[:, 0:49] = U_4x4
    for j in range(4):
        for k in range(4):
            V[j*8+k, 0:49] = V_4x4[j*4+k, 0:49]
    for i in range(4):
        for k in range(4):
            W[k*4+i, 0:49] = W_4x4[i*4+k, 0:49]
    
    U[:, 49:98] = U_4x4
    for j in range(4):
        for k in range(4):
            V[j*8+(k+4), 49:98] = V_4x4[j*4+k, 0:49]
    for i in range(4):
        for k in range(4):
            W[(k+4)*4+i, 49:98] = W_4x4[i*4+k, 0:49]
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
