def _try_improve(pts, max_time=10.0, seed=42):
    """Local search to improve edge count via single-point swaps (Greedy)."""
    n = len(pts)
    best = list(pts)
    best_e = _count_edges(best)
    in_set = set(best)
    
    # Build candidate pool: all points within hex distance 12 of any current point
    pool_set = set()
    for (a,b) in best:
        for a2 in range(a-12, a+13):
            for b2 in range(b-12, b+13):
                if _hexdist(a2,b2) <= 14 and abs(a2) <= 64 and abs(b2) <= 64:
                    pool_set.add((a2,b2))
    pool = list(pool_set)
    
    start = time.time()
    improved = True
    while improved and time.time() - start < max_time:
        improved = False
        for i in range(n):
            cur = best[i]
            # Try swapping cur with each candidate in pool
            # Optimization: Only check candidates that might improve
            # We check all, but break early if found best?
            # Greedy: take first improvement found.
            for cand in pool:
                if cand == cur or cand in in_set:
                    continue
                delta = _swap_delta(cur, cand, in_set)
                if delta > 0:
                    best_e += delta
                    best[i] = cand
                    in_set.discard(cur)
                    in_set.add(cand)
                    improved = True
                    if best_e == 76:
                        return best, best_e
                    break # Restart loop for this i? Or continue?
                    # Usually, restarting the scan for this point is safer.
    
    return best, best_e
