"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
import time
_start = time.time()
DEADLINE = 150.0
SAFETY = 10.0

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}

def construct():
    n, m, p = DIMENSIONS  # 2, 3, 6
    
    rank_total = 33
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    # Strassen 2x2x2 rank-7 decomposition terms
    strassen_2x2x2 = [
        ([ (0,1), (3,1) ], [ (0,1), (3,1) ], [ (0,1), (3,1) ]),
        ([ (2,1), (3,1) ], [ (0,1) ], [ (1,1), (3,-1) ]),
        ([ (0,1) ], [ (2,1), (3,-1) ], [ (2,1), (3,1) ]),
        ([ (3,1) ], [ (0,1), (2,-1) ], [ (0,1), (1,1) ]),
        ([ (0,1), (1,1) ], [ (3,1) ], [ (0,-1), (2,1) ]),
        ([ (1,1), (2,-1) ], [ (0,1), (2,1) ], [ (3,1) ]),
        ([ (2,1), (3,-1) ], [ (1,1), (3,1) ], [ (0,1) ]),
    ]
    
    col = 0
    for k0 in [0, 2, 4]:
        for (u_entries, v_entries, w_entries) in strassen_2x2x2:
            r = col
            for (idx, val) in u_entries:
                i = idx // 2
                j = idx % 2
                U[i * m + j][r] += val
            for (idx, val) in v_entries:
                j = idx // 2
                k_local = idx % 2
                V[j * p + (k0 + k_local)][r] += val
            for (idx, val) in w_entries:
                k_local = idx // 2
                i = idx % 2
                W[(k0 + k_local) * n + i][r] += val
            col += 1
    
    for i in range(n):
        for j in [2]:
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    assert col == rank_total
    
    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0
                for r in range(rank_total):
                    s += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                if s != 1:
                    return _schoolbook(n, m, p)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
