def construct():
    t0 = time.time()
    best = None
    best_val = np.inf

    def consider(h):
        nonlocal best, best_val
        h = np.clip(h, 0.0, 1.0)
        # fix sum
        h = _project(h)
        if not np.all(np.isfinite(h)):
            return
        v = _value(h)
        if v < best_val:
            best_val = v
            best = h.copy()

    rng = np.random.default_rng(0)
    # seed 1: uniform
    consider(np.full(N, 0.5))

    # continuous optimization from several random balanced seeds
    seeds = [np.full(N, 0.5)]
    for _ in range(8):
        h = rng.random(N)
        h = _project(h)
        seeds.append(h)

    for seed in seeds:
        if time.time() - t0 > DEADLINE - 20:
            break
        T = 20.0
        lam = 100.0
        res = minimize(
            _objective_smooth, seed, args=(T, lam),
            jac=_grad_smooth, method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        consider(res.x)
        # refine with higher T
        if time.time() - t0 > DEADLINE - 15:
            break
        T = 100.0
        res2 = minimize(
            _objective_smooth, res.x, args=(T, lam),
            jac=_grad_smooth, method='L-BFGS-B',
            bounds=[(0.0, 1.0)] * N,
            options={'maxiter': 400, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        consider(res2.x)

    # binary local search (simulated annealing) on best
    if best is not None and time.time() - t0 < DEADLINE - 10:
        h = best.copy()
        # snap to binary-ish? keep continuous but do local pair flips
        # do local search: try small perturbations
        cur = h.copy()
        cur_val = _value(cur)
        temp = 0.01
        ones = np.where(cur > 0.5)[0]
        zeros = np.where(cur <= 0.5)[0]
        # Actually do continuous local search: random small moves
        for it in range(20000):
            if time.time() - t0 > DEADLINE - 5:
                break
            j = rng.integers(N)
            delta = rng.uniform(-0.05, 0.05)
            cand = cur.copy()
            cand[j] = np.clip(cand[j] + delta, 0.0, 1.0)
            cand = _project(cand)
            v = _value(cand)
            if v < cur_val or rng.random() < np.exp((cur_val - v) / temp):
                cur = cand
                cur_val = v
            temp *= 0.9999
        consider(cur)

    if best is None:
        best = np.full(N, 0.5)
    return best
