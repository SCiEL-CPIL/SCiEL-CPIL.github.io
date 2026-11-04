import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    n, m, p = 3, 4, 5
    
    def add_term(U_dict, V_dict, W_dict):
        U_vec = np.zeros(n*m, dtype=int)
        V_vec = np.zeros(m*p, dtype=int)
        W_vec = np.zeros(p*n, dtype=int)
        for (i,j) in U_dict:
            U_vec[i*m + j] = U_dict[(i,j)]
        for (j,k) in V_dict:
            V_vec[j*p + k] = V_dict[(j,k)]
        for (i,k) in W_dict:
            W_vec[k*n + i] = W_dict[(i,k)]
        return U_vec, V_vec, W_vec

    def strassen_2x2x2(i_off, j_off, k_off):
        cols = []
        U1 = {(i_off, j_off): 1, (i_off+1, j_off+1): 1}
        V1 = {(j_off, k_off): 1, (j_off+1, k_off+1): 1}
        W1 = {(i_off, k_off): 1, (i_off+1, k_off+1): 1}
        cols.append(add_term(U1, V1, W1))
        U2 = {(i_off+1, j_off): 1, (i_off+1, j_off+1): 1}
        V2 = {(j_off, k_off): 1}
        W2 = {(i_off+1, k_off): 1, (i_off+1, k_off+1): -1}
        cols.append(add_term(U2, V2, W2))
        U3 = {(i_off, j_off): 1}
        V3 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): -1}
        W3 = {(i_off, k_off+1): 1, (i_off+1, k_off+1): 1}
        cols.append(add_term(U3, V3, W3))
        U4 = {(i_off+1, j_off+1): 1}
        V4 = {(j_off, k_off): 1, (j_off, k_off+1): -1}
        W4 = {(i_off, k_off): 1, (i_off+1, k_off): 1}
        cols.append(add_term(U4, V4, W4))
        U5 = {(i_off, j_off): 1, (i_off, j_off+1): 1}
        V5 = {(j_off+1, k_off+1): 1}
        W5 = {(i_off, k_off): -1, (i_off, k_off+1): 1}
        cols.append(add_term(U5, V5, W5))
        U6 = {(i_off+1, j_off): 1, (i_off, j_off): -1}
        V6 = {(j_off, k_off): 1, (j_off, k_off+1): 1}
        W6 = {(i_off+1, k_off+1): 1}
        cols.append(add_term(U6, V6, W6))
        U7 = {(i_off, j_off+1): 1, (i_off+1, j_off+1): -1}
        V7 = {(j_off+1, k_off): 1, (j_off+1, k_off+1): 1}
        W7 = {(i_off, k_off): 1, (i_off+1, k_off): -1}
        cols.append(add_term(U7, V7, W7))
        return cols

    def strassen_2x2x3(i_off, j_off, k_off):
        """2x2x3 decomposition: 6 schoolbook terms (no Strassen benefit for non-square)"""
        cols = []
        for i_local in range(2):
            for j_local in range(2):
                for k_local in range(3):
                    U_dict = {(i_off + i_local, j_off + j_local): 1}
                    V_dict = {(j_off + j_local, k_off + k_local): 1}
                    W_dict = {(i_off + i_local, k_off + k_local): 1}
                    cols.append(add_term(U_dict, V_dict, W_dict))
        return cols

    columns = []
    
    # Top 2 rows, cols 0-1: 2 Strassen 2x2x2 blocks (j split into {0,1} and {2,3})
    columns.extend(strassen_2x2x2(0, 0, 0))
    columns.extend(strassen_2x2x2(0, 2, 0))
    
    # Top 2 rows, cols 2-4: 2 strassen_2x2x3 blocks (j split into {0,1} and {2,3})
    columns.extend(strassen_2x2x3(0, 0, 2))
    columns.extend(strassen_2x2x3(0, 2, 2))
    
    # Bottom row, all 5 cols: 1x4x5 = 20 terms
    for j in range(4):
        for k in range(5):
            U_dict = {(2, j): 1}
            V_dict = {(j, k): 1}
            W_dict = {(2, k): 1}
            columns.append(add_term(U_dict, V_dict, W_dict))
    
    total_rank = len(columns)
    
    U_final = np.zeros((n*m, total_rank), dtype=int)
    V_final = np.zeros((m*p, total_rank), dtype=int)
    W_final = np.zeros((p*n, total_rank), dtype=int)
    for c, (U_c, V_c, W_c) in enumerate(columns):
        U_final[:, c] = U_c
        V_final[:, c] = V_c
        W_final[:, c] = W_c
    
    # Verify
    np.random.seed(42)
    for _ in range(3):
        A_test = np.random.randint(-10, 10, (n, m))
        B_test = np.random.randint(-10, 10, (m, p))
        C_expected = A_test @ B_test
        C_actual = np.zeros((n, p), dtype=int)
        for c in range(total_rank):
            fa = np.dot(U_final[:, c], A_test.reshape(-1))
            fb = np.dot(V_final[:, c], B_test.reshape(-1))
            Wc_reshaped = W_final[:, c].reshape(p, n).T
            C_actual += Wc_reshaped * (fa * fb)
        if not np.allclose(C_actual, C_expected):
            rank = n * m * p
            U = np.zeros((n*m, rank), dtype=int)
            V = np.zeros((m*p, rank), dtype=int)
            W = np.zeros((p*n, rank), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i*m + j, col] = 1
                        V[j*p + k, col] = 1
                        W[k*n + i, col] = 1
                        col += 1
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U_final.tolist(), "V": V_final.tolist(), "W": W_final.tolist()}
