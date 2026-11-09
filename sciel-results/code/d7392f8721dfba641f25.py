"""Rewritten program: Strassen-accelerated 4x5x5 matrix multiplication tensor."""
import time

DIMENSIONS = [4, 5, 5]

def _strassen_2x2x2():
    """Return the 7 rank-1 terms for 2x2x2 matrix multiplication (Strassen)."""
    U = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                  # M3
        {(1,1): 1},                  # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1},      # M7
    ]
    V = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                  # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                  # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1},       # M7
    ]
    W = [
        {(0,0): 1, (0,1): 1, (1,0): 1, (1,1): 1},  # M1
        {(0,1): 1, (1,1): -1},                      # M2
        {(0,1): 1, (1,1): 1},                       # M3
        {(0,0): 1, (1,0): 1},                       # M4
        {(0,0): -1, (0,1): 1},                      # M5
        {(1,1): 1},                                 # M6
        {(0,0): 1},                                 # M7
    ]
    return U, V, W

def _schoolbook_block(n, m, p):
    """Return schoolbook rank-1 terms for an n x m x p block."""
    U = []
    V = []
    W = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U.append({(i, j): 1})
                V.append({(j, k): 1})
                W.append({(k, i): 1})
    return U, V, W

def construct():
    start_time = time.time()
    deadline = 110.0
    
    n, m, p = DIMENSIONS  # 4, 5, 5
    
    n1, n2 = 2, 2
    m1, m2 = 2, 3
    p1, p2 = 2, 3
    
    all_U = []
    all_V = []
    all_W = []
    
    def add_block(i_off, j_off, k_off, bn, bm, bp, use_strassen):
        if use_strassen and bn == 2 and bm == 2 and bp == 2:
            U_loc, V_loc, W_loc = _strassen_2x2x2()
        else:
            U_loc, V_loc, W_loc = _schoolbook_block(bn, bm, bp)
        
        for u_loc, v_loc, w_loc in zip(U_loc, V_loc, W_loc):
            u_glob = {(ii + i_off, jj + j_off): c for (ii, jj), c in u_loc.items()}
            v_glob = {(jj + j_off, kk + k_off): c for (jj, kk), c in v_loc.items()}
            w_glob = {(kk + k_off, ii + i_off): c for (kk, ii), c in w_loc.items()}
            all_U.append(u_glob)
            all_V.append(v_glob)
            all_W.append(w_glob)
    
    add_block(0, 0, 0, n1, m1, p1, True)
    add_block(0, 0, p1, n1, m1, p2, False)
    add_block(0, m1, 0, n1, m2, p1, False)
    add_block(0, m1, p1, n1, m2, p2, False)
    add_block(n1, 0, 0, n2, m1, p1, True)
    add_block(n1, 0, p1, n2, m1, p2, False)
    add_block(n1, m1, 0, n2, m2, p1, False)
    add_block(n1, m1, p1, n2, m2, p2, False)
    
    rank = len(all_U)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r in range(rank):
        for (i, j), c in all_U[r].items():
            U[i * m + j][r] = c
        for (j, k), c in all_V[r].items():
            V[j * p + k][r] = c
        for (k, i), c in all_W[r].items():
            W[k * n + i][r] = c
    
    return {"U": U, "V": V, "W": W}
