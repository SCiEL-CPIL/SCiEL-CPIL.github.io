import numpy as np
import time

DIMENSIONS = [3, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # n=3, m=3, p=8
    # U size: 9, V size: 24, W size: 24
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_vec = [0.0] * (n * m)
        v_vec = [0.0] * (m * p)
        w_vec = [0.0] * (p * n)
        
        for idx, val in u_dict.items():
            u_vec[idx] = val
        for idx, val in v_dict.items():
            v_vec[idx] = val
        for idx, val in w_dict.items():
            w_vec[idx] = val
            
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)

    def u_idx(i, j):
        return i * m + j
        
    def v_idx(j, k):
        return j * p + k
        
    def w_idx(k, i):
        return k * n + i

    # Strassen 2x2x2 vectors
    # Indices: 00, 01, 10, 11
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, -1, 1, 0],
        [0, 1, -1, 0]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [0, -1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1],
        [0, 1, 0, 1]
    ]

    # Part 1: A[0:2, 0:2] * B[0:2, 0:8]
    # B is 2x8, split into 4 blocks of 2x2
    for b in range(4):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        for r in range(7):
            u_dict = {}
            v_dict = {}
            w_dict = {}
            
            # Map local 2x2 indices to global
            # Local i, j in {0,1}
            # u_idx(i, j)
            u_dict[u_idx(0, 0)] = strassen_u[r][0]
            u_dict[u_idx(0, 1)] = strassen_u[r][1]
            u_dict[u_idx(1, 0)] = strassen_u[r][2]
            u_dict[u_idx(1, 1)] = strassen_u[r][3]
            
            # v_idx(j, k)
            # Local j in {0,1}, local k in {0,1} -> global k0, k1
            v_dict[v_idx(0, k0)] = strassen_v[r][0]
            v_dict[v_idx(0, k1)] = strassen_v[r][1]
            v_dict[v_idx(1, k0)] = strassen_v[r][2]
            v_dict[v_idx(1, k1)] = strassen_v[r][3]
            
            # w_idx(k, i)
            # Local k in {0,1}, local i in {0,1}
            w_dict[w_idx(k0, 0)] = strassen_w[r][0]
            w_dict[w_idx(k1, 0)] = strassen_w[r][1]
            w_dict[w_idx(k0, 1)] = strassen_w[r][2]
            w_dict[w_idx(k1, 1)] = strassen_w[r][3]
            
            add_term(u_dict, v_dict, w_dict)

    # Part 2: A[2, 0:2] * B[0:2, 0:8]
    # A row 2, cols 0,1. B rows 0,1, cols 0..7.
    # This is 1x2 * 2x8.
    # Split B into 4 blocks of 1x2 * 2x2? No.
    # A[2, j] for j in 0,1. B[j, k] for j in 0,1.
    # Result C[2, k].
    # Trivial: sum over j.
    # For each k, C[2, k] = A[2,0]B[0,k] + A[2,1]B[1,k].
    # We can treat this as 4 blocks of 1x2 * 2x2?
    # No, just do it term by term.
    # For each k (0..7):
    # Term 1: A[2,0] B[0,k] C[2,k]
    # Term 2: A[2,1] B[1,k] C[2,k]
    for k in range(8):
        add_term({u_idx(2, 0): 1}, {v_idx(0, k): 1}, {w_idx(k, 2): 1})
        add_term({u_idx(2, 1): 1}, {v_idx(1, k): 1}, {w_idx(k, 2): 1})

    # Part 3: A[0:2, 2] * B[2, 0:8]
    # A rows 0,1, col 2. B row 2, cols 0..7.
    # Result C[i, k] for i in 0,1.
    # C[i, k] = A[i, 2] B[2, k].
    for i in range(2):
        for k in range(8):
            add_term({u_idx(i, 2): 1}, {v_idx(2, k): 1}, {w_idx(k, i): 1})

    # Part 4: A[2, 2] * B[2, 0:8]
    # A[2,2] B[2,k] C[2,k]
    for k in range(8):
        add_term({u_idx(2, 2): 1}, {v_idx(2, k): 1}, {w_idx(k, 2): 1})

    rank = len(U_cols)
    U = np.array(U_cols).T
    V = np.array(V_cols).T
    W = np.array(W_cols).T
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
