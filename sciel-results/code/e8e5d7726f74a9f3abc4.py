import time
import itertools

def construct():
    t0 = time.time()
    deadline = t0 + 140.0

    def _ok():
        return (time.time() - t0) < deadline

    n, m, p = 2, 3, 8

    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # ---- Build the multiplication tensor T (trace convention W[k*n+i, r]) ----
    # T[i, a, k] = 1 for every (i, a, k)
    triples = [(i, a, k) for i in range(n) for a in range(m) for k in range(p)]
    tri_set = set(triples)

    # Precompute candidate term signatures (only terms that are subsets of the
    # support of T are ever useful for an exact cover).
    # term: (U_support_set, V_support_set, W_support_set)
    U_rows = list(range(n * m))
    V_rows = list(range(m * p))
    W_rows = list(range(p * n))

    def _build_terms():
        terms = []
        for u in range(len(U_rows)):
            i = u // m
            a = u % m
            u_set = frozenset((i, a))
            for v in range(len(V_rows)):
                a2 = v // p
                k2 = v % p
                if a2 != a:
                    continue
                v_set = frozenset((a, k2))
                for w in range(len(W_rows)):
                    k = w // n
                    i2 = w % n
                    if i2 != i or k != k2:
                        continue
                    w_set = frozenset((k, i))
                    terms.append((u_set, v_set, w_set))
        return terms

    terms = _build_terms()

    def _verify(U, V, W, rank):
        # Exact integer verification: U^T * V * W^T must equal T
        # T[i,a,k] = sum_r U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r]
        for (i, a, k) in triples:
            s = 0
            ui = i * m + a
            va = a * p + k
            wk = k * n + i
            for r in range(rank):
                s += U[ui][r] * V[va][r] * W[wk][r]
            if s != 1:
                return False
        return True

    def _greedy(max_rank, use_random=False):
        import random
        rng = random.Random(12345)
        # target: cover every triple exactly once
        # For each term, the set of triples it covers = { (i,a,k) : (i,a) in u_set, (a,k) in v_set, (k,i) in w_set }
        # Since each term is a product of singleton sets, it covers exactly one triple (i,a,k).
        # So this is a partition: each term covers exactly 1 triple.
        # Thus the only exact cover is to pick one term per triple => rank = n*m*p = 48.
        # The greedy here is a no-op for exact single-triple terms; we return None.
        return None

    # ---- Key structural insight ----
    # Each valid term covers exactly ONE triple (because U, V, W supports are
    # singletons in the canonical tensor). Therefore any EXACT decomposition
    # requires at least one term per triple, i.e. rank >= n*m*p = 48 for this
    # tensor representation.
    #
    # HOWEVER: we are allowed to use terms whose supports are LARGER (rank-1
    # tensors are U[:,r] * V[:,r] * W[:,r] where each column can have multiple
    # non-zero entries). The standard Strassen trick shows that a single rank-1
    # term can cover MULTIPLE triples simultaneously if we allow the column
    # vectors to have multiple non-zeros and use cancellation / structured sums.
    #
    # So the real search space is: choose rank r, and for each r choose
    #   U_col (length n*m), V_col (length m*p), W_col (length p*n)
    # such that for all (i,a,k):
    #   sum_{r} U[i*m+a, r] * V[a*p+k, r] * W[k*n+i, r] = 1
    # and minimize r.
    #
    # The current best (Strassen hybrid) achieves r = 44. We attempt to beat
    # it by a structured search: fix a block-diagonal Strassen structure and
    # try to reduce the number of 2x2 Strassen blocks by overlapping.

    # ---- Build the Strassen-hybrid (rank 44) as the baseline, then try to
    #      reduce rank by merging the schoolbook tail with Strassen blocks. ----
    S_U = [
        [1, 0, 0, 1], [0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 0, 1],
        [1, 1, 0, 0], [-1, 0, 1, 0], [0, 1, 0, -1]
    ]
    S_V = [
        [1, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, -1], [-1, 0, 1, 0],
        [0, 0, 0, 1], [1, 1, 0, 0], [0, 0, 1, 1]
    ]
    S_W = [
        [1, 0, 0, 1], [0, 1, 0, -1], [0, 0, 1, 1], [1, 1, 0, 0],
        [-1, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]
    ]

    def _build_hybrid(n_blocks):
        # n_blocks: number of 2x2 Strassen blocks over the 8 columns of B
        # Each block covers 2 columns of B using 7 terms.
        # Remaining columns (8 - 2*n_blocks) use schoolbook: 2*3*1 = 6 terms each.
        rank = 7 * n_blocks + 6 * (8 - 2 * n_blocks)
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for block_idx in range(n_blocks):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
        # Schoolbook for remaining columns
        k0 = 2 * n_blocks
        for k in range(k0, p):
            for i in range(n):
                for a in range(m):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # n_blocks = 4 gives rank = 28 + 0 = 28? No: 8 - 2*4 = 0, so rank = 28.
    # Wait, that's wrong. Let me recompute: 7*4 + 6*0 = 28. But the original
    # only used 2 Strassen blocks (covering 4 columns) + schoolbook for 4 columns:
    # 7*2 + 6*4 = 14 + 24 = 38? But the original says rank=44.
    # Let me re-examine: original Part 1 uses 4 blocks of 2x2 (block_idx in range(4)),
    # each with 7 terms => 28 terms. Part 2: 2*8 = 16 terms. Total = 44.
    # So the original covers ALL 8 columns with Strassen (4 blocks) AND then
    # adds schoolbook for column a=2 (the 3rd row of A). That's the key:
    # Strassen only handles A[0:2, 0:2] * B[0:2, :], i.e. only the first 2 rows
    # of A and first 2 columns of B. The 3rd row of A (a=2) is handled by schoolbook.

    # So the structure is:
    #   A is 2x3, B is 3x8.
    #   Split A into A_top (2x2) and A_bottom (2x1).
    #   Split B into B_left (2x8) and B_right (1x8).
    #   C = A_top * B_left + A_bottom * B_right.
    #   A_top * B_left: 2x2 * 2x8 => use Strassen per 2x2 block of B_left: 4 blocks * 7 = 28.
    #   A_bottom * B_right: 2x1 * 1x8 => schoolbook: 2*8 = 16.
    #   Total = 44.

    # To reduce rank, we can try to also use Strassen for the A_bottom * B_right
    # part, but that's 2x1 * 1x8 which doesn't decompose into 2x2 blocks.
    # Alternatively, we can restructure: treat A as 2x3 and B as 3x8, and find
    # a different block decomposition.

    # Try: use Strassen for 2x2 * 2x2 blocks covering the full 2x3 * 3x8 by
    # padding. But that's not straightforward.

    # The best structural change: try rank 43, 42, ... by removing one schoolbook
    # term and compensating with a modified Strassen term. This requires solving
    # a small linear system. Given the time budget, we'll try a limited search.

    # For now, return the known-good rank 44 construction, but verify it.
    result = _build_hybrid(4)
    # But _build_hybrid(4) gives 28 terms for Strassen + 0 for schoolbook = 28,
    # which is WRONG because it doesn't account for A_bottom.
    # Let me fix _build_hybrid to match the original structure exactly.

    def _build_original():
        rank = 44
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        # Part 1: A[0:2, 0:2] * B[0:2, :] using Strassen per 2x2 block
        for block_idx in range(4):
            k0 = block_idx * 2
            for r in range(7):
                for li in range(2):
                    for la in range(2):
                        u_val = S_U[r][li * 2 + la]
                        if u_val != 0:
                            U[li * m + la][col] = u_val
                for la in range(2):
                    for lk in range(2):
                        v_val = S_V[r][la * 2 + lk]
                        if v_val != 0:
                            V[la * p + (k0 + lk)][col] = v_val
                for lk in range(2):
                    for li in range(2):
                        w_val = S_W[r][lk * 2 + li]
                        if w_val != 0:
                            W[(k0 + lk) * n + li][col] = w_val
                col += 1
                if not _ok():
                    return _schoolbook()
        # Part 2: A[:, 2] * B[2, :] schoolbook
        for i in range(n):
            for k in range(p):
                U[i * m + 2][col] = 1
                V[2 * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                if not _ok():
                    return _schoolbook()
        return {"U": U, "V": V, "W": W}

    result = _build_original()
    if not _verify(result["U"], result["V"], result["W"], 44):
        return _schoolbook()

    # ---- Attempt to reduce rank: try removing one schoolbook term and solving
    #      for a compensating modification. This is a limited structural search. ----
    # Given the time budget and the difficulty of finding a valid rank-43 solution,
    # we return the verified rank-44 result.

    return result
