import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = 4, 4, 7

    # Build the full multiplication tensor T of shape (n*m, m*p, p*n)
    # T[i*m + j, j*p + k, k*n + i] = 1
    T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    # Reshape T to a 2D matrix M of shape (n*m * m*p, p*n)
    # M[a, c] = T[a // (m*p), a % (m*p), c]
    # Actually, let's reshape as (n*m, m*p * p*n) or (n*m * m*p, p*n)
    # We'll use (n*m * m*p, p*n) to get a matrix with rank <= p*n = 28
    M = T.reshape(n * m * m * p, p * n)

    # Compute the rank of M
    # Use SVD to find the numerical rank
    U_svd, s, Vt = np.linalg.svd(M, full_matrices=False)
    # The rank is the number of singular values above a threshold
    tol = 1e-10
    rank_lower = np.sum(s > tol)

    # The tensor rank is at least rank_lower
    # Now we need to find an exact decomposition with rank as low as possible
    # Start with a higher rank and try to reduce

    # For now, use the known Strassen-based decomposition as a starting point
    # and verify it's correct, then try to optimize

    # Actually, let's just build the decomposition directly using the tensor structure
    # The tensor T is sparse and structured. We can find a decomposition by
    # expressing T as a sum of rank-1 tensors.

    # Use a greedy approach: start with all possible rank-1 tensors that match
    # some entries of T, and select a minimal set.

    # Given the complexity, let's use the known optimal decomposition for 4x4x4
    # and extend it to 4x4x7.

    # For 4x4x4, the Strassen decomposition has 49 terms.
    # For the remaining 3 columns (k=4,5,6), we need additional terms.

    # Let's build the 4x4x4 Strassen decomposition first.
    terms = []

    def add_term(u, v, w):
        terms.append((u, v, w))

    def zero_u(): return [0] * (n * m)
    def zero_v(): return [0] * (m * p)
    def zero_w(): return [0] * (p * n)

    # Strassen 4x4x4 decomposition
    outer_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    inner_terms = [
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        ([(3, 1)], [(2, 1), (0, -1)], [(0, 1), (2, 1)]),
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    def get_idx_A(block, local):
        br, bc = block // 2, block % 2
        lr, lc = local // 2, local % 2
        r, c = br * 2 + lr, bc * 2 + lc
        return r * m + c

    def get_idx_B(block, local):
        br, bc = block // 2, block % 2
        lr, lc = local // 2, local % 2
        r, c = br * 2 + lr, bc * 2 + lc
        return r * p + c

    def get_idx_C(block, local):
        br, bc = block // 2, block % 2
        lr, lc = local // 2, local % 2
        r, c = br * 2 + lr, bc * 2 + lc
        return c * n + r

    for oa, ob, oc in outer_terms:
        for ia, ib, ic in inner_terms:
            u = zero_u()
            v = zero_v()
            w = zero_w()

            for (la, da) in ia:
                for (ba, ca) in oa:
                    idx = get_idx_A(ba, la)
                    u[idx] += ca * da

            for (lb, db) in ib:
                for (bb, cb) in ob:
                    idx = get_idx_B(bb, lb)
                    v[idx] += cb * db

            for (lc, dc) in ic:
                for (bc, cc) in oc:
                    idx = get_idx_C(bc, lc)
                    w[idx] += cc * dc

            add_term(u, v, w)

    # For k=4,5: use Strassen 2x2 blocks
    def strassen_2x2_block(i_off, j_off, k_off):
        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off * m + j_off] = 1; u[(i_off + 1) * m + (j_off + 1)] = 1
        v[j_off * p + k_off] = 1; v[(j_off + 1) * p + (k_off + 1)] = 1
        w[k_off * n + i_off] = 1; w[(k_off + 1) * n + (i_off + 1)] = 1
        add_term(u, v, w)

        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off + 1) * m + j_off] = 1; u[(i_off + 1) * m + (j_off + 1)] = 1
        v[j_off * p + k_off] = 1
        w[k_off * n + (i_off + 1)] = 1; w[(k_off + 1) * n + (i_off + 1)] = -1
        add_term(u, v, w)

        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off * m + j_off] = 1
        v[j_off * p + (k_off + 1)] = 1; v[(j_off + 1) * p + (k_off + 1)] = -1
        w[(k_off + 1) * n + i_off] = 1; w[(k_off + 1) * n + (i_off + 1)] = 1
        add_term(u, v, w)

        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off + 1) * m + (j_off + 1)] = 1
        v[(j_off + 1) * p + k_off] = 1; v[j_off * p + k_off] = -1
        w[k_off * n + i_off] = 1; w[k_off * n + (i_off + 1)] = 1
        add_term(u, v, w)

        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off * m + j_off] = 1; u[i_off * m + (j_off + 1)] = 1
        v[(j_off + 1) * p + (k_off + 1)] = 1
        w[(k_off + 1) * n + i_off] = 1; w[k_off * n + i_off] = -1
        add_term(u, v, w)

        u = zero_u(); v = zero_v(); w = zero_w()
        u[(i_off + 1) * m + j_off] = 1; u[i_off * m + j_off] = -1
        v[j_off * p + k_off] = 1; v[j_off * p + (k_off + 1)] = 1
        w[(k_off + 1) * n + (i_off + 1)] = 1
        add_term(u, v, w)

        u = zero_u(); v = zero_v(); w = zero_w()
        u[i_off * m + (j_off + 1)] = 1; u[(i_off + 1) * m + (j_off + 1)] = -1
        v[(j_off + 1) * p + k_off] = 1; v[(j_off + 1) * p + (k_off + 1)] = 1
        w[k_off * n + i_off] = 1
        add_term(u, v, w)

    for i_b in range(2):
        for j_b in range(2):
            strassen_2x2_block(i_b * 2, j_b * 2, 4)

    # For k=6: schoolbook
    for i in range(n):
        for j in range(m):
            u = zero_u(); v = zero_v(); w = zero_w()
            u[i * m + j] = 1
            v[j * p + 6] = 1
            w[6 * n + i] = 1
            add_term(u, v, w)

    if time.time() > deadline:
        return {"U": [], "V": [], "W": []}

    rank = len(terms)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for c, (u, v, w) in enumerate(terms):
        for r in range(n * m):
            if u[r] != 0:
                U[r][c] = u[r]
        for r in range(m * p):
            if v[r] != 0:
                V[r][c] = v[r]
        for r in range(p * n):
            if w[r] != 0:
                W[r][c] = w[r]

    return {"U": U, "V": V, "W": W}
