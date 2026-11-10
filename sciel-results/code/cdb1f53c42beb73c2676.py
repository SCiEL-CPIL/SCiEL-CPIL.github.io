def rudin_problem10() -> np.ndarray:
    t0 = time.time()
    DEADLINE = 595.0
    Y, W = _build()
    rng = np.random.default_rng(2024)
    
    def run(c0, maxiter, ftol, gtol):
        res = minimize(_obj, c0, jac=True, method='L-BFGS-B',
                       options={'maxiter': maxiter, 'ftol': ftol, 'gtol': gtol, 'maxls': 50})
        c = res.x
        r = _ratio(c, Y, W)
        return c, r
    
    best_c = None
    best_r = np.inf
    candidates = []  # (r, c) for top candidates to polish
    
    def consider(c, r):
        nonlocal best_c, best_r
        if r < best_r:
            best_r = r
            best_c = c.copy()
        candidates.append((r, c.copy()))
        candidates.sort(key=lambda x: x[0])
        if len(candidates) > 12:
            candidates = candidates[:12]
    
    # Phase 1
    n_inits = 0
    while time.time() - t0 < DEADLINE - 100:
        style = rng.integers(0, 5)
        if n_inits == 0:
            c0 = np.ones(_NM)
        elif style == 0:
            c0 = rng.standard_normal(_NM)
        elif style == 1:
            c0 = rng.standard_normal(_NM) * np.exp(-np.abs(np.arange(_NM)-10)*0.4)
        elif style == 2:
            c0 = rng.standard_normal(_NM)
            mask = rng.random(_NM) < 0.25
            c0[~mask] = 0
        elif style == 3:
            c0 = np.zeros(_NM)
            k = int(rng.integers(2, 9))
            idxs = rng.choice(_NM, size=k, replace=False)
            c0[idxs] = rng.standard_normal(k)
        else:
            c0 = rng.standard_normal(_NM)
            c0 = c0 * rng.uniform(0.5, 2.0)
        nrm = np.linalg.norm(c0)
        if nrm < 1e-9:
            continue
        c0 = c0/nrm
        n_inits += 1
        c, r = run(c0, maxiter=300, ftol=1e-11, gtol=1e-11)
        consider(c, r)
    
    # Phase 2: polish top candidates
    for r, c in list(candidates):
        if time.time() - t0 > DEADLINE - 3:
            break
        c2, r2 = run(c, maxiter=4000, ftol=1e-14, gtol=1e-14)
        consider(c2, r2)
    
    out = best_c/np.linalg.norm(best_c)
    return out
