# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    u_dim = n * m  # 4
    v_dim = m * p  # 8
    w_dim = p * n  # 8
    
    rank = 14
    U = [[0] * rank for _ in range(u_dim)]
    V = [[0] * rank for _ in range(v_dim)]
    W = [[0] * rank for _ in range(w_dim)]
    
    # Strassen terms for 2x2x2
    # u: (i, j), v: (j, k), w: (k, i)
    # i in {0,1}, j in {0,1}, k in {0,1}
    
    terms = [
        # 1: (a00+a11)(b00+b11) -> c00, c11
        ([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)]),
        # 2: (a10+a11)b00 -> c10, c11
        ([(1,0,1), (1,1,1)], [(0,0,1)], [(0,1,1), (1,1,-1)]),
        # 3: a00(b01-b11) -> c01, c11
        ([(0,0,1)], [(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)]),
        # 4: a11(b10-b00) -> c00, c10
        ([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)]),
        # 5: (a00+a01)b11 -> c01, c00
        ([(0,0,1), (0,1,1)], [(1,1,1)], [(1,0,1), (0,0,-1)]),
        # 6: (a10-a00)(b00+b01) -> c11
        ([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)]),
        # 7: (a01-a11)(b10+b11) -> c00
        ([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)])
    ]
    
    # Fill matrices for Block 1 (k=0,1) and Block 2 (k=2,3)
    for r in range(7):
        u_entries, v_entries, w_entries = terms[r]
        
        # Block 1: k in {0, 1}
        col1 = r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col1] += val
        for j, k, val in v_entries:
            v_idx = j * p + k
            V[v_idx][col1] += val
        for k, i, val in w_entries:
            w_idx = k * n + i
            W[w_idx][col1] += val
            
        # Block 2: k in {2, 3}
        col2 = 7 + r
        for i, j, val in u_entries:
            u_idx = i * m + j
            U[u_idx][col2] += val
        for j, k, val in v_entries:
            k_new = k + 2
            v_idx = j * p + k_new
            V[v_idx][col2] += val
        for k, i, val in w_entries:
            k_new = k + 2
            w_idx = k_new * n + i
            W[w_idx][col2] += val
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
