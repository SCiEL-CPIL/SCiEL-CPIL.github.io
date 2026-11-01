"""Program for matrix multiplication 2,2,10 using direct-sum Strassen decomposition."""
DIMENSIONS = [2, 2, 10]

# EVOLVE-BLOCK-START
def construct():
    """
    Decompose 2x2 * 2x10 multiplication into five independent 2x2 * 2x2 blocks.
    Each block uses a rank-7 Strassen decomposition. Total rank = 35.

    Indexing:
    - U has shape (n*m, rank) = (4, 35), entry U[i*m+j, c]
    - V has shape (m*p, rank) = (20, 35), entry V[j*p+k, c]
    - W has shape (p*n, rank) = (20, 35), entry W[k*n+i, c]

    For block r (0..4), B columns are 2r, 2r+1.
    """
    n, m, p = DIMENSIONS
    nb = p // 2
    rank = nb * 7

    # Define the 7 Strassen products compactly as flat lists of (index, value)
    # U index = i*m+j (i,j in {0,1})
    # V index = j*p+k_local where k_local in {0,1} (local column within block)
    # W index = k_local*n+i (local)
    
    # Each term: (u_list, v_list, w_list)
    # u_list: [(i*m+j, val), ...]
    # v_list: [(j*2 + k_local, val), ...]  -- using p=2 for local indexing
    # w_list: [(k_local*n + i, val), ...]
    
    terms = [
        # M1 = (a+d)(e+h) -> C[0,0]+, C[1,1]+
        ([(0,1),(3,1)], [(0,1),(3,1)], [(0,1),(3,1)]),
        # M2 = (c+d)e -> C[1,0]+, C[1,1]-
        ([(2,1),(3,1)], [(0,1)], [(1,1),(3,-1)]),
        # M3 = a(f-h) -> C[0,1]+, C[1,1]+
        ([(0,1)], [(1,1),(3,-1)], [(2,1),(3,1)]),
        # M4 = d(g-e) -> C[0,0]+, C[1,0]+
        ([(3,1)], [(2,1),(0,-1)], [(0,1),(1,1)]),
        # M5 = (a+b)h -> C[0,0]-, C[0,1]+
        ([(0,1),(1,1)], [(3,1)], [(0,-1),(2,1)]),
        # M6 = (c-a)(e+f) -> C[1,1]+
        ([(2,1),(0,-1)], [(0,1),(1,1)], [(3,1)]),
        # M7 = (b-d)(g+h) -> C[0,0]+
        ([(1,1),(3,-1)], [(2,1),(3,1)], [(0,1)]),
    ]

    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]

    for r in range(nb):
        for t in range(7):
            c = r * 7 + t
            u_list, v_list, w_list = terms[t]
            for idx, val in u_list:
                U[idx][c] = float(val)
            for idx, val in v_list:
                j, kl = divmod(idx, 2)
                V[j * p + (2 * r + kl)][c] = float(val)
            for idx, val in w_list:
                kl, i = divmod(idx, n)
                W[(2 * r + kl) * n + i][c] = float(val)

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
