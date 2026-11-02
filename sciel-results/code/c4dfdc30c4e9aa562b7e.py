<<<<<<< SEARCH
def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Symmetric initial configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = v * S3H
        return np.array([x, y])

    def _start_from_base(base, scale):
        pts = base + rng.normal(0, scale, size=(N, 2))
        return _project_to_triangle(pts)

    # Phase A: Differential Evolution (global)
    if _elapsed_ok(50):
        try:
            def de_obj(flat):
                pts = flat.reshape(-1, 2)
                return -_min_triangle_area(_project_to_triangle(pts))

            bounds = [(0, 1), (0, S3H)] * N
            de_res = differential_evolution(
                de_obj, bounds, maxiter=50, popsize=20,
                seed=42, tol=1e-10, polish=False,
                init='latinhypercube', mutation=(0.5, 1.5), recombination=0.7
            )
            de_pts = _project_to_triangle(de_res.x.reshape(-1, 2))
            _update_best(de_pts)
        except Exception:
            pass

    # Phase B: Sobol-grid + local nudge strategy
    # Initialize on quasi-random low-discrepancy points for broad coverage
    from scipy.stats import qmc

    def _sobol_points_in_triangle(n):
        """Generate n quasi-random points in the equilateral triangle."""
        sampler = qmc.Sobol(d=2, scramble=True)
        raw = sampler.random(n)
        # Convert to barycentric: use (b, c) with b>=0, c>=0, b+c<=1
        # Use rejection-free mapping: u,v in [0,1], if u+v>1 reflect
        b = raw[:, 0].copy()
        c = raw[:, 1].copy()
        mask = b + c > 1.0
        b[mask] = 1.0 - b[mask]
        c[mask] = 1.0 - c[mask]
        # Map to cartesian
        x = b + 0.5 * c
        y = c * S3H
        return np.column_stack([x, y])

    # Generate multiple Sobol-based initializations
    n_starts = 30
    bases = [init.copy()]
    for k in range(20):
        sobol_pts = _sobol_points_in_triangle(N)
        # Add small jitter to break symmetry
        jitter = rng.normal(0, 0.03, size=(N, 2))
        bases.append(_project_to_triangle(sobol_pts + jitter))
    for _ in range(5):
        rand_pts = np.array([_random_point_in_triangle() for _ in range(N)])
        bases.append(rand_pts)
    if best_pts is not None:
        bases.append(best_pts.copy())
        bases.append(_push_apart(best_pts, 0.01).copy())

    # Local nudge refinement: move points in smallest triangles
    def _local_nudge(pts, n_rounds=50, step_scale=0.02):
        """Move points involved in currently smallest triangles."""
        pts = pts.copy()
        for _ in range(n_rounds):
            if not _elapsed_ok(5):
                break
            areas = _all_triangle_areas(pts)
            # Find the 10 smallest triangles
            idx = np.argsort(areas)[:10]
            for tri_idx in idx:
                i, j, k = _TRIPLETS[tri_idx]
                # Try small moves for each of the 3 points
                for p_idx in [i, j, k]:
                    best_pos = pts[p_idx].copy()
                    best_area = areas[tri_idx]
                    for _trial in range(12):
                        # Random direction
                        angle = rng.uniform(0, 2 * np.pi)
                        step = step_scale * rng.uniform(0.3, 1.0)
                        dx = step * np.cos(angle)
                        dy = step * np.sin(angle)
                        new_pos = pts[p_idx] + np.array([dx, dy])
                        # Project
                        y_new = np.clip(new_pos[1], 0.0, S3H)
                        x_min = y_new / S3
                        x_max = 1.0 - y_new / S3
                        new_pos = np.array([np.clip(new_pos[0], x_min, x_max), y_new])
                        pts[p_idx] = new_pos
                        new_areas = _all_triangle_areas(pts)
                        new_min = np.min(new_areas)
                        if new_min > best_area:
                            best_area = new_min
                            best_pos = new_pos.copy()
                        else:
                            pts[p_idx] = best_pos.copy()
                    pts[p_idx] = best_pos
        return _project_to_triangle(pts)

    for s in range(n_starts):
        if not _elapsed_ok(25):
            break
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.03)
        _update_best(pts)

        # Apply local nudge refinement
        pts_nudged = _local_nudge(pts, n_rounds=30, step_scale=0.015)
        _update_best(pts_nudged)

        # Then do a few L-BFGS-B passes with moderate beta
        for beta in [100.0, 500.0, 2000.0]:
            if not _elapsed_ok(20):
                break
            try:
                bc = _cartesian_to_barycentric_correct(pts_nudged)
                flat_bc = bc.flatten()
                res = minimize(
                    lambda x, b=beta: _soft_min_bary_grad(x, b)[0],
                    flat_bc,
                    jac=lambda x, b=beta: _soft_min_bary_grad(x, b)[1],
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-12, 'maxls': 20}
                )
                bc_opt = res.x.reshape(-1, 2)
                b = np.maximum(bc_opt[:, 0], 0.0)
                c = np.maximum(bc_opt[:, 1], 0.0)
                s_val = b + c
                mask = s_val > 1.0
                b[mask] /= s_val[mask]
                c[mask] /= s_val[mask]
                pts_opt = _barycentric_to_cartesian(np.column_stack([b, c]))
                _update_best(pts_opt)
                pts_nudged = pts_opt
            except Exception:
                pass

    # Phase C: Additional local nudge passes with smaller steps
    if best_pts is not None and _elapsed_ok(15):
        for step_scale in [0.01, 0.005, 0.002]:
            if not _elapsed_ok(10):
                break
            pts_nudged = _local_nudge(best_pts, n_rounds=40, step_scale=step_scale)
            _update_best(pts_nudged)
            best_pts = pts_nudged

    # Phase D: Nelder-Mead polish in barycentric coords (fixed projection)
    if best_pts is not None and _elapsed_ok(12):
        bc_best = _cartesian_to_barycentric_correct(best_pts)
        for scale in [0.03, 0.015, 0.008]:
            if not _elapsed_ok(8):
                break
            for _ in range(5):
                if not _elapsed_ok(6):
                    break
                pert_bc = bc_best + rng.normal(0, scale, size=(N, 2))
                # Project to valid barycentric
                b = np.maximum(pert_bc[:, 0], 0.0)
                c = np.maximum(pert_bc[:, 1], 0.0)
                s_val = b + c
                mask = s_val > 1.0
                b[mask] /= s_val[mask]
                c[mask] /= s_val[mask]
                flat = np.column_stack([b, c]).flatten()

                def nm_obj(x):
                    bc_x = x.reshape(-1, 2)
                    bx = np.maximum(bc_x[:, 0], 0.0)
                    cx = np.maximum(bc_x[:, 1], 0.0)
                    s_val2 = bx + cx
                    mask2 = s_val2 > 1.0
                    bx[mask2] /= s_val2[mask2]
                    cx[mask2] /= s_val2[mask2]
                    px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
                    return -_soft_min_area(px, 5000.0)

                try:
                    res = minimize(nm_obj, flat, method='Nelder-Mead',
                                   options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14})
                    bc_res = res.x.reshape(-1, 2)
                    b = np.maximum(bc_res[:, 0], 0.0)
                    c = np.maximum(bc_res[:, 1], 0.0)
                    s_val = b + c
                    mask = s_val > 1.0
                    b[mask] /= s_val[mask]
                    c[mask] /= s_val[mask]
                    p = _barycentric_to_cartesian(np.column_stack([b, c]))
                    _update_best(p)
                except Exception:
                    pass

    # Phase E: Final fine polish with exact min (fixed projection)
    if best_pts is not None and _elapsed_ok(6):
        bc_best = _cartesian_to_barycentric_correct(best_pts)
        flat = bc_best.flatten()

        def final_nm(x):
            bc_x = x.reshape(-1, 2)
            bx = np.maximum(bc_x[:, 0], 0.0)
            cx = np.maximum(bc_x[:, 1], 0.0)
            s_val = bx + cx
            mask = s_val > 1.0
            bx[mask] /= s_val[mask]
            cx[mask] /= s_val[mask]
            px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
            return -_min_triangle_area(px)

        try:
            res = minimize(final_nm, flat, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-16})
            bc_res = res.x.reshape(-1, 2)
            b = np.maximum(bc_res[:, 0], 0.0)
            c = np.maximum(bc_res[:, 1], 0.0)
            s_val = b + c
            mask = s_val > 1.0
            b[mask] /= s_val[mask]
            c[mask] /= s_val[mask]
            p = _barycentric_to_cartesian(np.column_stack([b, c]))
            _update_best(p)
        except Exception:
            pass

    # Phase F: Final local nudge with very small steps
    if best_pts is not None and _elapsed_ok(3):
        pts_final = _local_nudge(best_pts, n_rounds=20, step_scale=0.001)
        _update_best(pts_final)

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return _project_to_triangle(pts)

    # Final safety: project and verify all points are inside
    result = _project_to_triangle(best_pts.copy())
    # Double-check: verify each point is inside the triangle
    for i in range(N):
        x, y = result[i]
        if y < -1e-10 or y > S3H + 1e-10:
            result[i] = [0.5, 0.0]
        x_min = y / S3
        x_max = 1.0 - y / S3
        if x < x_min - 1e-10 or x > x_max + 1e-10:
            result[i] = [(x_min + x_max) / 2, y]
    return result
=======
def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)

    best_pts = None
    best_area = 0.0

    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)

    def _update_best(pts):
        nonlocal best_pts, best_area
        pts = _project_to_triangle(pts.copy())
        area = _min_triangle_area(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Symmetric initial configuration (vertices + midpoints + centroid-ish)
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    _update_best(init)

    def _random_point_in_triangle():
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u + 0.5 * v
        y = v * S3H
        return np.array([x, y])

    def _start_from_base(base, scale):
        pts = base + rng.normal(0, scale, size=(N, 2))
        return _project_to_triangle(pts)

    from scipy.stats import qmc

    def _sobol_points_in_triangle(n):
        sampler = qmc.Sobol(d=2, scramble=True)
        raw = sampler.random(n)
        b = raw[:, 0].copy()
        c = raw[:, 1].copy()
        mask = b + c > 1.0
        b[mask] = 1.0 - b[mask]
        c[mask] = 1.0 - c[mask]
        x = b + 0.5 * c
        y = c * S3H
        return np.column_stack([x, y])

    # Generate diverse initializations
    bases = [init.copy()]
    for k in range(25):
        sobol_pts = _sobol_points_in_triangle(N)
        jitter = rng.normal(0, 0.04, size=(N, 2))
        bases.append(_project_to_triangle(sobol_pts + jitter))
    for _ in range(8):
        rand_pts = np.array([_random_point_in_triangle() for _ in range(N)])
        bases.append(rand_pts)
    # Grid-based init: spread points on a triangular grid
    grid_pts = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, S3H],
        [0.25, S3H/4], [0.75, S3H/4], [0.5, S3H/2],
        [0.125, S3H/8], [0.375, S3H/8], [0.625, S3H/8], [0.875, S3H/8],
        [0.5, 0.0],
    ])
    bases.append(_project_to_triangle(grid_pts))
    if best_pts is not None:
        bases.append(best_pts.copy())

    # Phase A: Quick DE for global exploration
    if _elapsed_ok(40):
        try:
            def de_obj(flat):
                pts = flat.reshape(-1, 2)
                return -_min_triangle_area(_project_to_triangle(pts))

            bounds = [(0, 1), (0, S3H)] * N
            de_res = differential_evolution(
                de_obj, bounds, maxiter=40, popsize=15,
                seed=42, tol=1e-10, polish=False,
                init='latinhypercube', mutation=(0.5, 1.5), recombination=0.7
            )
            de_pts = _project_to_triangle(de_res.x.reshape(-1, 2))
            _update_best(de_pts)
        except Exception:
            pass

    # Phase B: Multi-start with targeted push + soft-min Nelder-Mead
    n_starts = 35
    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        base = bases[s % len(bases)]
        pts = _start_from_base(base, 0.03)
        _update_best(pts)

        # Targeted push: directly attack smallest triangles
        pts_pushed = _targeted_push(pts, n_rounds=150, step_scale=0.02)
        _update_best(pts_pushed)

        # Soft-min Nelder-Mead in barycentric coords
        if _elapsed_ok(25):
            bc = _cartesian_to_barycentric_correct(pts_pushed)
            b = np.maximum(bc[:, 0], 0.0)
            c = np.maximum(bc[:, 1], 0.0)
            s_val = b + c
            mask = s_val > 1.0
            b[mask] /= s_val[mask]
            c[mask] /= s_val[mask]
            flat = np.column_stack([b, c]).flatten()

            def nm_soft(x, beta=3000.0):
                bc_x = x.reshape(-1, 2)
                bx = np.maximum(bc_x[:, 0], 0.0)
                cx = np.maximum(bc_x[:, 1], 0.0)
                sv = bx + cx
                m = sv > 1.0
                bx[m] /= sv[m]
                cx[m] /= sv[m]
                px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
                return -_soft_min_area(px, beta)

            try:
                res = minimize(nm_soft, flat, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13})
                bc_res = res.x.reshape(-1, 2)
                b2 = np.maximum(bc_res[:, 0], 0.0)
                c2 = np.maximum(bc_res[:, 1], 0.0)
                sv2 = b2 + c2
                m2 = sv2 > 1.0
                b2[m2] /= sv2[m2]
                c2[m2] /= sv2[m2]
                p2 = _barycentric_to_cartesian(np.column_stack([b2, c2]))
                _update_best(p2)
                pts_pushed = p2
            except Exception:
                pass

        # Additional targeted push after NM
        if _elapsed_ok(20):
            pts_pushed2 = _targeted_push(pts_pushed, n_rounds=100, step_scale=0.01)
            _update_best(pts_pushed2)
            pts_pushed = pts_pushed2

    # Phase C: Intensive refinement on best
    if best_pts is not None:
        # Multiple rounds of targeted push with decreasing step
        for step_scale in [0.015, 0.008, 0.004, 0.002, 0.001]:
            if not _elapsed_ok(15):
                break
            pts_refined = _targeted_push(best_pts, n_rounds=200, step_scale=step_scale)
            _update_best(pts_refined)

        # L-BFGS-B with soft min for smooth gradient-based refinement
        if _elapsed_ok(12):
            for beta in [500.0, 2000.0, 8000.0]:
                if not _elapsed_ok(10):
                    break
                try:
                    bc = _cartesian_to_barycentric_correct(best_pts)
                    b = np.maximum(bc[:, 0], 0.0)
                    c = np.maximum(bc[:, 1], 0.0)
                    sv = b + c
                    m = sv > 1.0
                    b[m] /= sv[m]
                    c[m] /= sv[m]
                    flat_bc = np.column_stack([b, c]).flatten()

                    def obj_grad(x, bta=beta):
                        bc_x = x.reshape(-1, 2)
                        bx = np.maximum(bc_x[:, 0], 0.0)
                        cx = np.maximum(bc_x[:, 1], 0.0)
                        sv2 = bx + cx
                        m2 = sv2 > 1.0
                        bx[m2] /= sv2[m2]
                        cx[m2] /= sv2[m2]
                        px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
                        val = -_soft_min_area(px, bta)
                        grad_pts = _soft_min_grad(px, bta)
                        gb = -grad_pts[:, 0]
                        gc = -(0.5 * grad_pts[:, 0] + S3H * grad_pts[:, 1])
                        return val, np.column_stack([gb, gc]).flatten()

                    res = minimize(
                        lambda x, bta=beta: obj_grad(x, bta)[0],
                        flat_bc,
                        jac=lambda x, bta=beta: obj_grad(x, bta)[1],
                        method='L-BFGS-B',
                        options={'maxiter': 500, 'ftol': 1e-15, 'gtol': 1e-13, 'maxls': 30}
                    )
                    bc_opt = res.x.reshape(-1, 2)
                    b3 = np.maximum(bc_opt[:, 0], 0.0)
                    c3 = np.maximum(bc_opt[:, 1], 0.0)
                    sv3 = b3 + c3
                    m3 = sv3 > 1.0
                    b3[m3] /= sv3[m3]
                    c3[m3] /= sv3[m3]
                    p3 = _barycentric_to_cartesian(np.column_stack([b3, c3]))
                    _update_best(p3)
                except Exception:
                    pass

        # Final Nelder-Mead with exact min
        if _elapsed_ok(8):
            bc_best = _cartesian_to_barycentric_correct(best_pts)
            b4 = np.maximum(bc_best[:, 0], 0.0)
            c4 = np.maximum(bc_best[:, 1], 0.0)
            sv4 = b4 + c4
            m4 = sv4 > 1.0
            b4[m4] /= sv4[m4]
            c4[m4] /= sv4[m4]
            flat_final = np.column_stack([b4, c4]).flatten()

            def final_nm(x):
                bc_x = x.reshape(-1, 2)
                bx = np.maximum(bc_x[:, 0], 0.0)
                cx = np.maximum(bc_x[:, 1], 0.0)
                sv = bx + cx
                m = sv > 1.0
                bx[m] /= sv[m]
                cx[m] /= sv[m]
                px = _barycentric_to_cartesian(np.column_stack([bx, cx]))
                return -_min_triangle_area(px)

            try:
                res = minimize(final_nm, flat_final, method='Nelder-Mead',
                               options={'maxiter': 4000, 'xatol': 1e-13, 'fatol': 1e-15})
                bc_res = res.x.reshape(-1, 2)
                b5 = np.maximum(bc_res[:, 0], 0.0)
                c5 = np.maximum(bc_res[:, 1], 0.0)
                sv5 = b5 + c5
                m5 = sv5 > 1.0
                b5[m5] /= sv5[m5]
                c5[m5] /= sv5[m5]
                p5 = _barycentric_to_cartesian(np.column_stack([b5, c5]))
                _update_best(p5)
            except Exception:
                pass

        # Very fine targeted push at the end
        if _elapsed_ok(4):
            pts_final = _targeted_push(best_pts, n_rounds=300, step_scale=0.0005)
            _update_best(pts_final)

    if best_pts is None:
        pts = np.array([_random_point_in_triangle() for _ in range(N)])
        return _project_to_triangle(pts)

    result = _project_to_triangle(best_pts.copy())
    for i in range(N):
        x, y = result[i]
        if y < -1e-10 or y > S3H + 1e-10:
            result[i] = [0.5, 0.0]
        x_min = y / S3
        x_max = 1.0 - y / S3
        if x < x_min - 1e-10 or x > x_max + 1e-10:
            result[i] = [(x_min + x_max) / 2, y]
    return result
>>>>>>> REPLACE
