def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    # Phase 1: Multiple hexagonal initializations with force relaxation
    best_pts = None
    best_r = 0.0

    for seed in [42, 123, 7, 999, 555, 1234, 4321]:
        if time.time() > deadline - 40.0:
            break

        pts = _hex_init(N, seed=seed)
        r = _compute_radius(pts)

        # Run force relaxation
        # Use a deadline for the relaxation phase to ensure time for polishing
        relax_deadline = deadline - 30.0
        rel_pts, rel_r = _force_relax(
            pts,
            dt=0.005,
            n_steps=800,
            wall_strength=10.0,
            pair_strength=1.0,
            deadline=relax_deadline
        )

        if rel_r > r:
            pts = rel_pts
            r = rel_r

        if r > best_r:
            best_r = r
            best_pts = pts.copy()

    # Phase 2: Gradient-based refinement with soft-min
    if time.time() < deadline - 10.0 and best_pts is not None:
        temperatures = [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
        x0 = best_pts.ravel()

        for T in temperatures:
            if time.time() > deadline - 8.0:
                break
            try:
                result = minimize(
                    lambda x: _objective_and_grad(x, T)[0],
                    x0,
                    jac=lambda x: _objective_and_grad(x, T)[1],
                    method='L-BFGS-B',
                    options={
                        'maxiter': 150,
                        'ftol': 1e-12,
                        'gtol': 1e-10,
                        'maxls': 50,
                        'maxfun': 400,
                    }
                )
                new_pts = result.x.reshape(-1, 2)
                norms = np.linalg.norm(new_pts, axis=1)
                if np.any(norms >= 1.0) or not np.all(np.isfinite(new_pts)):
                    continue
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
                    x0 = new_pts.ravel()
            except Exception:
                continue

    # Phase 3: Final high-precision polish
    if time.time() < deadline - 3.0:
        try:
            T_final = 0.00005
            result = minimize(
                lambda x: _objective_and_grad(x, T_final)[0],
                x0,
                jac=lambda x: _objective_and_grad(x, T_final)[1],
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-14,
                    'gtol': 1e-12,
                    'maxls': 100,
                    'maxfun': 800,
                }
            )
            new_pts = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_pts, axis=1)
            if np.all(norms < 1.0) and np.all(np.isfinite(new_pts)):
                new_r = _compute_radius(new_pts)
                if new_r > best_r:
                    best_r = new_r
                    best_pts = new_pts.copy()
        except Exception:
            pass

    # Ensure validity
    if best_pts is None:
        best_pts = _hex_init(N, seed=0)
    norms = np.linalg.norm(best_pts, axis=1)
    mask = norms < 1.0
    if not np.all(mask) or not np.all(np.isfinite(best_pts)):
        best_pts = _hex_init(N, seed=0)

    return best_pts
