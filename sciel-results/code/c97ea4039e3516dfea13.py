"""Rewritten program: algebraic rank reduction via Strassen-style block identities."""
import time

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    
    U_cols = []
    V_cols = []
    W_cols = []
    
    def add_term(u_dict, v_dict, w_dict):
        u_vec = [0] * (n * m)
        v_vec = [0] * (m * p)
        w_vec = [0] * (p * n)
        for (i, j), c in u_dict.items():
            u_vec[i * m + j] = c
        for (j, k), c in v_dict.items():
            v_vec[j * p + k] = c
        for (i, k), c in w_dict.items():
            w_vec[k * n + i] = c
        U_cols.append(u_vec)
        V_cols.append(v_vec)
        W_cols.append(w_vec)
    
    def add_strassen_block(i0, i1, j0, j1, k0, k1):
        # m1 = (A[i0,j0] + A[i1,j1]) * (B[j0,k0] + B[j1,k1])
        add_term(
            {(i0, j0): 1, (i1, j1): 1},
            {(j0, k0): 1, (j1, k1): 1},
            {(i0, k0): 1, (i1, k1): 1}
        )
        # m2 = (A[i1,j0] + A[i1,j1]) * B[j0,k0]
        add_term(
            {(i1, j0): 1, (i1, j1): 1},
            {(j0, k0): 1},
            {(i1, k0): 1, (i1, k1): -1}
        )
        # m3 = A[i0,j0] * (B[j0,k1] - B[j1,k1])
        add_term(
            {(i0, j0): 1},
            {(j0, k1): 1, (j1, k1): -1},
            {(i0, k1): 1, (i1, k1): 1}
        )
        # m4 = A[i1,j1] * (B[j1,k0] - B[j0,k0])
        add_term(
            {(i1, j1): 1},
            {(j1, k0): 1, (j0, k0): -1},
            {(i0, k0): 1, (i1, k0): 1}
        )
        # m5 = (A[i0,j0] + A[i0,j1]) * B[j1,k1]
        add_term(
            {(i0, j0): 1, (i0, j1): 1},
            {(j1, k1): 1},
            {(i0, k0): -1, (i0, k1): 1}
        )
        # m6 = (A[i1,j0] - A[i0,j0]) * (B[j0,k0] + B[j0,k1])
        add_term(
            {(i1, j0): 1, (i0, j0): -1},
            {(j0, k0): 1, (j0, k1): 1},
            {(i1, k1): 1}
        )
        # m7 = (A[i0,j1] - A[i1,j1]) * (B[j1,k0] + B[j1,k1])
        add_term(
            {(i0, j1): 1, (i1, j1): -1},
            {(j1, k0): 1, (j1, k1): 1},
            {(i0, k0): 1}
        )
    
    def add_schoolbook_block(i_list, j_list, k_list):
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    add_term(
                        {(i, j): 1},
                        {(j, k): 1},
                        {(i, k): 1}
                    )
    
    i_blocks = [[0, 1], [2]]
    j_blocks = [[0, 1], [2, 3]]
    k_blocks = [[0, 1], [2, 3], [4, 5]]
    
    for ib in i_blocks:
        for jb in j_blocks:
            for kb in k_blocks:
                if len(ib) == 2 and len(jb) == 2 and len(kb) == 2:
                    add_strassen_block(ib[0], ib[1], jb[0], jb[1], kb[0], kb[1])
                else:
                    add_schoolbook_block(ib, jb, kb)
    
    rank = len(U_cols)
    U = [[U_cols[c][a] for c in range(rank)] for a in range(n * m)]
    V = [[V_cols[c][b] for c in range(rank)] for b in range(m * p)]
    W = [[W_cols[c][d] for c in range(rank)] for d in range(p * n)]
    
    return {"U": U, "V": V, "W": W}
