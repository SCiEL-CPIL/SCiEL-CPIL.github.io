def _local_search(pos0, proxy, budget, rng):
    pos = pos0.copy()
    V = proxy(pos)
    best = pos.copy(); best_V = V
    t0 = time.time()
    step = 0.03
    n = 128
    it = 0
    while time.time()-t0 < budget:
        it += 1
        # pick random coordinate
        d = rng.integers(n)
        # line search: try +/- step and adapt
        old = pos[d]
        # try a few candidates
        best_local = old; best_local_V = V
        for s in (step, -step, 2*step, -2*step, 0.5*step, -0.5*step):
            pos[d] = old + s
            v = proxy(pos)
            if v < best_local_V:
                best_local_V = v; best_local = old + s
        pos[d] = best_local
        V = best_local_V
        if V < best_V:
            best_V = V; best = pos.copy()
        # occasional shake
        if it % 50 == 0:
            # perturb a random subset
            idx = rng.choice(n, size=16, replace=False)
            pos[idx] += rng.normal(0, 0.05, size=16)
            V = proxy(pos)
            if V < best_V:
                best_V = V; best = pos.copy()
    return best, best_V
