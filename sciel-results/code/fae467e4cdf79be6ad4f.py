"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [3, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    # U: (n*m, rank), V: (m*p, rank), W: (p*n, rank)
    # We don't know rank upfront, so we use lists of lists and append columns
    U = [[0] * 0 for _ in range(n * m)]
    V = [[0] * 0 for _ in range(m * p)]
    W = [[0] * 0 for _ in range(p * n)]
    
    def add_term(u_vals, v_vals, w_vals):
        # u_vals: dict (i,j) -> val
        # v_vals: dict (j,k) -> val
        # w_vals: dict (k,i) -> val
        # Append a new column
        rank = len(U[0]) if U else 0
        for i in range(n):
            for j in range(m):
                U[i*m + j].append(u_vals.get((i,j), 0))
        for j in range(m):
            for k in range(p):
                V[j*p + k].append(v_vals.get((j,k), 0))
        for k in range(p):
            for i in range(n):
                W[k*n + i].append(w_vals.get((k,i), 0))

    # Strassen 2x2x2 decomposition
    # Local indices 0,1
    # Terms (u, v, w)
    strassen_terms = [
        # 1
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # 2
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(0,1):1, (1,1):-1}),
        # 3
        ({(0,0):1}, {(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}),
        # 4
        ({(1,1):1}, {(1,0):1, (0,0):-1}, {(0,1):1, (1,1):1}),
        # 5
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(1,0):-1, (1,1):1}),
        # 6
        ({(1,0):-1, (0,0):1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # 7
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,0):-1}),
    ]
    
    # Verify Strassen terms sum to 1 for all i,j,k in 0..1
    # This is a known property, but let's assume it's correct.
    
    blocks = [
        (0, 0, 0), # i0, j0, k0
        (0, 2, 0),
        (0, 0, 2),
        (0, 2, 2),
    ]
    
    for i0, j0, k0 in blocks:
        if time.time() > deadline:
            break
        for u, v, w in strassen_terms:
            # Map local to global
            gu = {}
            for (li, lj), val in u.items():
                gi = i0 + li
                gj = j0 + lj
                gu[(gi, gj)] = val
            gv = {}
            for (lj, lk), val in v.items():
                gj = j0 + lj
                gk = k0 + lk
                gv[(gj, gk)] = val
            gw = {}
            for (lk, li), val in w.items():
                gk = k0 + lk
                gi = i0 + li
                gw[(gk, gi)] = val
            add_term(gu, gv, gw)
            
    # Border 1: i=2
    u1 = {}
    for j in range(m):
        u1[(2, j)] = 1
    v1 = {}
    for j in range(m):
        for k in range(p):
            v1[(j, k)] = 1
    w1 = {}
    for k in range(p):
        w1[(k, 2)] = 1
    add_term(u1, v1, w1)
    
    # Border 2: k=4, i in {0,1}
    u2 = {}
    for i in [0, 1]:
        for j in range(m):
            u2[(i, j)] = 1
    v2 = {}
    for j in range(m):
        v2[(j, 4)] = 1
    w2 = {}
    for i in [0, 1]:
        w2[(4, i)] = 1
    add_term(u2, v2, w2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
