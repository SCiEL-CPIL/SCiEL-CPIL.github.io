<<<<<<< SEARCH
def construct():
    t0 = time.time()
    def left():
        return DEADLINE_FRAC - (time.time() - t0)

    starts = _build_structured_starts()
    scored = []
    for s in starts:
        if left() <= 2.0:
            break
        sn = _normalize(s)
        scored.append((_score(sn), sn))
    scored.sort(key=lambda x: -x[0])

    best_score, best_pts = 0.0, _normalize(_regular_polygon(N, 1.0, 0.0))
    if scored:
        s0, p0 = scored[0]
        best_score, best_pts = s0, p0.copy()

    # Top candidates for intensive optimization
    # We take top 5 to allow exploration of different local basins
    top_candidates = [pts for _, pts in scored[:5]]

    def try_opt(x0, method, maxiter, fatol, xatol, use_smooth=False, tau=0.01, use_grad=False):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            if use_smooth and use_grad:
                obj_func = lambda f: _smooth_neg_score_and_grad(f, tau)
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, jac=True, options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 10000, 'maxls': 50})
            elif use_smooth:
                obj_func = lambda f: _smooth_neg_score(f, tau)
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 10000, 'maxls': 50})
            else:
                obj_func = _neg_score
                opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                        'adaptive': True, 'maxfev': 25000}
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    # Intensive optimization on top candidates
    # Strategy: 
    # 1. Nelder-Mead on top candidates to find local minima.
    # 2. L-BFGS-B with smooth surrogate (corrected soft-min) on best.
    # 3. Powell on the best found so far.
    # 4. L-BFGS-B with tighter tau for precision.
    # 5. Final tight Nelder-Mead polish.
    
    # Phase 1: Nelder-Mead on top candidates
    for pts in top_candidates:
        if left() <= 12.0:
            break
        try_opt(pts, 'Nelder-Mead', 15000, 1e-13, 1e-10)

    # Phase 2: L-BFGS-B with smooth surrogate + gradient (coarse tau first)
    if left() > 25.0:
        for tau in [0.2, 0.1, 0.05, 0.02]:
            if left() <= 8.0:
                break
            try_opt(best_pts, 'L-BFGS-B', 500, 1e-12, 1e-12, use_smooth=True, tau=tau, use_grad=True)
            # Quick Nelder-Mead polish after each L-BFGS-B step
            if left() > 10.0:
                try_opt(best_pts, 'Nelder-Mead', 3000, 1e-13, 1e-10)

    # Phase 3: Powell on best
    if left() > 15.0:
        try_opt(best_pts, 'Powell', 5000, 1e-13, 1e-11)

    # Phase 4: L-BFGS-B with tight tau for precision (with gradient)
    if left() > 12.0:
        for tau in [0.01, 0.005, 0.002]:
            if left() <= 5.0:
                break
            try_opt(best_pts, 'L-BFGS-B', 400, 1e-14, 1e-14, use_smooth=True, tau=tau, use_grad=True)

    # Phase 5: Final Polish with tight Nelder-Mead
    if left() > 8.0:
        try_opt(best_pts, 'Nelder-Mead', 10000, 1e-15, 1e-13)
    
    # Phase 6: Ultra-tight final polish
    if left() > 4.0:
        try_opt(best_pts, 'Nelder-Mead', 5000, 1e-16, 1e-14)

    return _normalize(best_pts)
=======
def construct():
    t0 = time.time()
    def left():
        return DEADLINE_FRAC - (time.time() - t0)

    best_score = 0.0
    best_pts = _normalize(_regular_polygon(N, 1.0, 0.0))

    # === PHASE A: 7+7 Symmetric Parameterization (9D search) ===
    # This is the structural change: reduce from 28D to 9D by exploiting symmetry
    if left() > 30.0:
        # Generate diverse starting points in the 9D parameter space
        hept_starts = []
        # Regular 14-gon equivalent: R_out=1, R_in=1, rot=0, all s=0
        hept_starts.append(np.array([1.0, 1.0, 0.0, 0, 0, 0, 0, 0, 0], dtype=float))
        # Various configurations
        for R_out in [0.9, 1.0, 1.1]:
            for R_in in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                for rot in [0.0, np.pi/7, np.pi/14]:
                    p = np.array([R_out, R_in, rot, 0, 0, 0, 0, 0, 0], dtype=float)
                    hept_starts.append(p)
        # With some symmetry-breaking offsets
        for R_out in [0.95, 1.0, 1.05]:
            for R_in in [0.4, 0.5, 0.6]:
                for rot in [0.0, np.pi/7]:
                    for s_pattern in [
                        [0.05, -0.05, 0.05, -0.05, 0.05, -0.05],
                        [0.1, 0, -0.1, 0, 0.1, 0],
                        [0.03, 0.03, -0.03, -0.03, 0.03, 0.03],
                    ]:
                        p = np.array([R_out, R_in, rot] + s_pattern, dtype=float)
                        hept_starts.append(p)
        
        # Score and sort
        hept_scored = []
        for p in hept_starts:
            if left() <= 2.0:
                break
            sc = _hept77_score(p)
            hept_scored.append((sc, p))
        hept_scored.sort(key=lambda x: -x[0])
        
        if hept_scored:
            hs, hp = hept_scored[0]
            if hs > best_score:
                best_score, best_pts = hs, _hept77_to_points(hp).copy()
        
        # Optimize top hept candidates
        top_hept = [p for _, p in hept_scored[:4]]
        for p0 in top_hept:
            if left() <= 15.0:
                break
            try:
                res = scipy.optimize.minimize(
                    _hept77_neg_score, p0, method='Nelder-Mead',
                    options={'maxiter': 5000, 'fatol': 1e-14, 'xatol': 1e-12, 'adaptive': True, 'maxfev': 10000}
                )
                sc = _hept77_score(res.x)
                if sc > best_score:
                    best_score = sc
                    best_pts = _hept77_to_points(res.x).copy()
            except Exception:
                pass
        
        # L-BFGS-B with numerical gradient on best hept params
        if left() > 20.0 and hept_scored:
            p_best = hept_scored[0][1].copy()
            # Update p_best from current best if it came from hept optimization
            for sc, p in hept_scored:
                if abs(_hept77_score(p) - best_score) < 1e-10:
                    p_best = p.copy()
                    break
            try:
                res = scipy.optimize.minimize(
                    _hept77_neg_score, p_best, method='L-BFGS-B',
                    jac=_hept77_grad,
                    options={'maxiter': 200, 'ftol': 1e-15, 'gtol': 1e-14, 'maxfev': 5000}
                )
                sc = _hept77_score(res.x)
                if sc > best_score:
                    best_score = sc
                    best_pts = _hept77_to_points(res.x).copy()
            except Exception:
                pass
            # Final Nelder-Mead polish on hept params
            if left() > 10.0:
                try:
                    res = scipy.optimize.minimize(
                        _hept77_neg_score, p_best, method='Nelder-Mead',
                        options={'maxiter': 3000, 'fatol': 1e-15, 'xatol': 1e-14, 'adaptive': True, 'maxfev': 5000}
                    )
                    sc = _hept77_score(res.x)
                    if sc > best_score:
                        best_score = sc
                        best_pts = _hept77_to_points(res.x).copy()
                except Exception:
                    pass

    # === PHASE B: Full 28D optimization (fallback and refinement) ===
    starts = _build_structured_starts()
    scored = []
    for s in starts:
        if left() <= 2.0:
            break
        sn = _normalize(s)
        scored.append((_score(sn), sn))
    scored.sort(key=lambda x: -x[0])

    if scored:
        s0, p0 = scored[0]
        if s0 > best_score:
            best_score, best_pts = s0, p0.copy()

    top_candidates = [pts for _, pts in scored[:5]]

    def try_opt(x0, method, maxiter, fatol, xatol, use_smooth=False, tau=0.01, use_grad=False):
        nonlocal best_score, best_pts
        if left() <= 4.0:
            return
        try:
            if use_smooth and use_grad:
                obj_func = lambda f: _smooth_neg_score_and_grad(f, tau)
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, jac=True, options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 10000, 'maxls': 50})
            elif use_smooth:
                obj_func = lambda f: _smooth_neg_score(f, tau)
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options={'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol, 'maxfev': 10000, 'maxls': 50})
            else:
                obj_func = _neg_score
                opts = {'maxiter': maxiter, 'fatol': fatol, 'xatol': xatol,
                        'adaptive': True, 'maxfev': 25000}
                res = scipy.optimize.minimize(obj_func, x0.flatten(), method=method, options=opts)
            opt = _normalize(res.x.reshape(-1, 2))
            sc = _score(opt)
            if sc > best_score:
                best_score, best_pts = sc, opt.copy()
        except Exception:
            pass

    # Nelder-Mead on top full-space candidates
    for pts in top_candidates:
        if left() <= 10.0:
            break
        try_opt(pts, 'Nelder-Mead', 8000, 1e-13, 1e-10)

    # L-BFGS-B with smooth surrogate
    if left() > 15.0:
        for tau in [0.1, 0.05, 0.02]:
            if left() <= 6.0:
                break
            try_opt(best_pts, 'L-BFGS-B', 300, 1e-12, 1e-12, use_smooth=True, tau=tau, use_grad=True)
            if left() > 8.0:
                try_opt(best_pts, 'Nelder-Mead', 2000, 1e-13, 1e-10)

    # Final polish
    if left() > 8.0:
        try_opt(best_pts, 'Nelder-Mead', 8000, 1e-15, 1e-13)
    if left() > 4.0:
        try_opt(best_pts, 'Nelder-Mead', 4000, 1e-16, 1e-14)

    return _normalize(best_pts)
>>>>>>> REPLACE
