<<<<<<< SEARCH
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x6."""
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill
    n, m, p = N_DIM, M_DIM, P_DIM

    # guaranteed baseline
    base = _schoolbook(n, m, p)
    base_rank = n * m * p

    on, a2bc, b2ac, c2ab, ai, aj, bj, bk, ck, ci = _build_support(n, m, p)
    off = _build_offcheck(n, m, p)
    off_a, off_b, off_c = off[:, 0], off[:, 1], off[:, 2]

    rng = np.random.default_rng(12345)
    random.seed(12345)

    best = base
    best_rank = base_rank

    # Theoretical lower bound ~ n*p*m = 54 for this structure; but try lower.
    # Lower bound: max(nm, mp, pn) = max(9,18,18)=18, and also >= number of
    # distinct slices. We sweep upward from a small value.
    # Budget remaining after baseline:
    time_per_rank = 8  # seconds
    min_r = 18
    max_r = base_rank - 1  # we only care to beat schoolbook
    if time.time() > deadline - 10:
        return base

    # sweep ranks; for each, allow a few restarts within the time budget
    for r in range(min_r, base_rank):
        if time.time() > deadline - 6:
            break
        remaining = deadline - 6 - time.time()
        # how many restarts can we afford? each ~ time_per_rank/ n_restarts
        n_restarts = max(1, int(remaining / time_per_rank))
        n_restarts = min(n_restarts, 6)
        res = _try_rank(n, m, p, r, on, a2bc, b2ac, c2ab, n_restarts, deadline - 6, rng)
        if res is not None:
            best = res
            best_rank = r
            break  # found lowest-rank; stop (we sweep upward)

    # final safety: ensure best still verifies
    if time.time() > deadline - 3:
        return best
    try:
        if not _verify(best["U"], best["V"], best["W"], n, m, p):
            return base
    except Exception:
        return base
    return best
# EVOLVE-BLOCK-END
=======
def construct():
    """Find a rank-minimal exact tensor decomposition for 3x3 * 3x6."""
    start = time.time()
    deadline = start + 140  # safety margin before 150s kill
    n, m, p = N_DIM, M_DIM, P_DIM

    # guaranteed baseline
    base = _schoolbook(n, m, p)
    base_rank = n * m * p

    # Try explicit Strassen-based construction first
    strassen = _strassen_based_decomp(n, m, p)
    best = base
    best_rank = base_rank
    if strassen is not None:
        if _verify(strassen["U"], strassen["V"], strassen["W"], n, m, p):
            best = strassen
            best_rank = strassen["rank"]

    on, a_groups, b_groups, c_groups = _build_support(n, m, p)

    rng = np.random.default_rng(12345)
    random.seed(12345)

    # Sweep ranks upward from a lower bound
    # Lower bound: max(nm, mp, pn) = 18, but we know Strassen gives ~42
    # Try to find something better than the Strassen construction
    min_r = 18
    max_r = best_rank - 1  # only try to beat current best

    if time.time() > deadline - 10:
        return best

    # sweep ranks; for each, allow restarts within the time budget
    for r in range(min_r, max_r + 1):
        if time.time() > deadline - 6:
            break
        remaining = deadline - 6 - time.time()
        if remaining < 3:
            break
        # How many restarts can we afford?
        n_restarts = max(1, int(remaining / 10))
        n_restarts = min(n_restarts, 8)
        res = _try_rank(n, m, p, r, on, a_groups, b_groups, c_groups, n_restarts, deadline - 6, rng)
        if res is not None:
            best = res
            best_rank = r
            break  # found lowest-rank; stop (we sweep upward)

    # final safety: ensure best still verifies
    if time.time() > deadline - 3:
        return best
    try:
        if not _verify(best["U"], best["V"], best["W"], n, m, p):
            return base
    except Exception:
        return base
    return best
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
