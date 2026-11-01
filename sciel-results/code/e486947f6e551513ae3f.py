"""Program for matrix multiplication 2,2,10 using block-diagonal Strassen decomposition."""
import time

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
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill

    n, m, p = DIMENSIONS
    nb = p // 2  # 5 blocks
    rank = nb * 7  # 35

    # Strassen rank-7 decomposition for 2x2 matrices
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h), M2 = (c+d)e, M3 = a(f-h), M4 = d(g-e)
    # M5 = (a+b)h, M6 = (c-a)(e+f), M7 = (b-d)(g+h)
    #
    # C[0,0] = M1 + M4 - M5 + M7
    # C[0,1] = M3 + M5
    # C[1,0] = M2 + M4
    # C[1,1] = M1 - M2 + M3 + M6
    #
    # Each term stored as (u_list, v_list, w_list) where:
    #   u_list: [(i*m+j, val), ...]  -- factor of A
    #   v_list: [(j*2+kl, val), ...] -- factor of B (local kl in {0,1})
    #   w_list: [(kl*n+i, val), ...] -- factor of C (local kl in {0,1})

    terms = [
        # M1 = (a+d)(e+h) -> C[0,0]+=1, C[1,1]+=1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2 = (c+d)e -> C[1,0]+=1, C[1,1]-=1
        ([(2, 1), (3, 1)], [(0, 1)], [(1, 1), (3, -1)]),
        # M3 = a(f-h) -> C[0,1]+=1, C[1,1]+=1
        ([(0, 1)], [(1, 1), (3, -1)], [(2, 1), (3, 1)]),
        # M4 = d(g-e) -> C[0,0]+=1, C[1,0]+=1
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (1, 1)]),
        # M5 = (a+b)h -> C[0,0]-=1, C[0,1]+=1
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (2, 1)]),
        # M6 = (c-a)(e+f) -> C[1,1]+=1
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7 = (b-d)(g+h) -> C[0,0]+=1
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    U = [[0.0] * rank for _ in range(n * m)]   # 4 x 35
    V = [[0.0] * rank for _ in range(m * p)]   # 20 x 35
    W = [[0.0] * rank for _ in range(p * n)]   # 20 x 35

    for r in range(nb):
        if time.time() > deadline:
            break
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
