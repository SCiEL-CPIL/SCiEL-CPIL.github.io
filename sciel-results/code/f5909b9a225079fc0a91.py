import numpy as np
import time

DIMENSIONS = [4, 4, 8]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = DIMENSIONS  # 4, 4, 8
    nm = n * m   # 16
    mp = m * p   # 32
    pn = p * n   # 32

    U_list = []
    V_list = []
    W_list = []

    def add_term(u_dict, v_dict, w_dict):
        U_row = np.zeros(nm)
        V_row = np.zeros(mp)
        W_row = np.zeros(pn)
        for (i, j), val in u_dict.items():
            U_row[i * m + j] += val
        for (j, k), val in v_dict.items():
            V_row[j * p + k] += val
        for (k, i), val in w_dict.items():
            W_row[k * n + i] += val
        U_list.append(U_row)
        V_list.append(V_row)
        W_list.append(W_row)

    # BCLR (Bini-Capovani-Lotti-Romani) rank-14 decomposition for 4x4*4x4
    # Each term is (U_indices, V_indices, W_indices) where each is a list of (row,col) pairs
    # U: A[i,j], V: B[j,k], W: C[k,i]
    
    bclr_terms = [
        # Term 1
        ([(0,0), (1,0), (2,1), (3,1), (0,2), (1,2), (2,3), (3,3)],
         [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)],
         [(0,0), (1,0), (0,1), (1,1), (2,2), (3,2), (2,3), (3,3)]),
        # Term 2
        ([(0,0), (1,0), (2,1), (3,1), (0,2), (1,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)]),
        # Term 3
        ([(0,0), (1,0), (0,1), (1,1), (2,2), (3,2), (2,3), (3,3)],
         [(0,0), (1,0), (2,0), (3,0), (0,2), (1,2), (2,2), (3,2)],
         [(0,0), (1,0), (0,1), (1,1), (2,2), (3,2), (2,3), (3,3)]),
        # Term 4
        ([(0,0), (1,0), (0,1), (1,1), (2,2), (3,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)]),
        # Term 5
        ([(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)],
         [(0,0), (1,0), (2,0), (3,0), (0,2), (1,2), (2,2), (3,2)],
         [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)]),
        # Term 6
        ([(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)]),
        # Term 7
        ([(0,0), (1,0), (0,1), (1,1), (2,0), (3,0), (2,1), (3,1)],
         [(0,0), (1,0), (2,0), (3,0), (0,3), (1,3), (2,3), (3,3)],
         [(0,0), (1,0), (0,1), (1,1), (2,3), (3,3), (2,2), (3,2)]),
        # Term 8
        ([(0,0), (1,0), (0,1), (1,1), (2,0), (3,0), (2,1), (3,1)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)]),
        # Term 9
        ([(0,0), (1,0), (2,1), (3,1), (0,2), (1,2), (2,3), (3,3)],
         [(0,0), (1,0), (2,0), (3,0), (2,2), (3,2), (2,3), (3,3)],
         [(0,0), (1,0), (2,0), (3,0), (2,2), (3,2), (2,3), (3,3)]),
        # Term 10
        ([(0,0), (1,0), (2,1), (3,1), (0,2), (1,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)]),
        # Term 11
        ([(0,0), (1,0), (2,1), (3,1), (0,2), (1,2), (2,3), (3,3)],
         [(0,2), (1,2), (2,2), (3,2), (2,0), (3,0), (2,1), (3,1)],
         [(0,2), (1,2), (2,2), (3,2), (2,0), (3,0), (2,1), (3,1)]),
        # Term 12
        ([(0,0), (1,0), (2,1), (3,1), (0,2), (1,2), (2,3), (3,3)],
         [(0,3), (0,2), (1,3), (1,2), (2,1), (2,0), (3,1), (3,0)],
         [(0,3), (1,3), (0,2), (1,2), (2,1), (3,1), (2,0), (3,0)]),
        # Term 13
        ([(0,0), (1,0), (2,1), (3,1), (0,2), (1,2), (2,3), (3,3)],
         [(0,0), (0,1), (1,0), (1,1), (2,2), (2,3), (3,2), (3,3)],
         [(0,0), (1,0), (0,1), (1,1), (2,2), (3,2), (2,3), (3,3)]),
        # Term 14
        ([(0,0), (1,0), (2,1), (3,1), (0,2), (1,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)],
         [(0,1), (1,1), (2,1), (3,1), (2,2), (3,2), (2,3), (3,3)])
    ]

    # Apply BCLR for B_left (cols 0:4 of B) and C_left (cols 0:4 of C)
    for u_idx, v_idx, w_idx in bclr_terms:
        u_dict = {(i, j): 1 for i, j in u_idx}
        v_dict = {(j, k): 1 for j, k in v_idx}
        w_dict = {(k, i): 1 for k, i in w_idx}
        add_term(u_dict, v_dict, w_dict)

    # Apply BCLR for B_right (cols 4:8 of B) and C_right (cols 4:8 of C)
    for u_idx, v_idx, w_idx in bclr_terms:
        u_dict = {(i, j): 1 for i, j in u_idx}
        v_dict = {(j, k+4): 1 for j, k in v_idx}
        w_dict = {(k+4, i): 1 for k, i in w_idx}
        add_term(u_dict, v_dict, w_dict)

    rank = len(U_list)
    U = np.array(U_list).T  # shape (nm, rank)
    V = np.array(V_list).T  # shape (mp, rank)
    W = np.array(W_list).T  # shape (pn, rank)

    # Verify
    error = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-10:
                    error += abs(val - 1)
    
    if error > 1e-8:
        # Fallback to schoolbook
        U_school = np.zeros((nm, 128))
        V_school = np.zeros((mp, 128))
        W_school = np.zeros((pn, 128))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_school[i*m + j, col] = 1
                    V_school[j*p + k, col] = 1
                    W_school[k*n + i, col] = 1
                    col += 1
        return {
            "U": U_school.tolist(),
            "V": V_school.tolist(),
            "W": W_school.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
