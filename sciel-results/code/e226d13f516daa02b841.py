import numpy as np
import time
from scipy.optimize import minimize
from scipy.interpolate import CubicSpline

def moving_sofa_2d() -> np.ndarray:
    # --- Grid (fixed by contract) ---
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    x_grid, y_grid = np.meshgrid(xs, ys)
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    n_points = len(x_flat)

    start_time = time.time()
    time_limit = 600.0  # safety margin before 630s kill

    # --- Spline parameterization ---
    # t in [0,1]; x(t), y(t) cubic B-splines through control points;
    # angle(t) = 90 * s(t) where s is a monotone cubic spline through 0..1.
    # The evaluator's densified path is the linear interp of (x,y,a) at the
    # waypoints; a smooth spline makes those waypoints lie on a smooth curve,
    # which is geometrically better for the sofa (fewer kinks).
    N_CTRL = 9  # number of control points

    def _build_spline(cx, cy, ca):
        """cx,cy,ca: arrays of length N_CTRL. Returns function t->(x,y,a_deg)."""
        t_ctrl = np.linspace(0.0, 1.0, N_CTRL)
        sx = CubicSpline(t_ctrl, cx)
        sy = CubicSpline(t_ctrl, cy)
        # monotone-ish angle spline from 0 to 90
        sa = CubicSpline(t_ctrl, ca)
        return sx, sy, sa

    def _spline_waypoints(sx, sy, sa, n_wp):
        """Sample n_wp waypoints (incl endpoints) along t in [0,1]."""
        ts = np.linspace(0.0, 1.0, n_wp)
        xs_ = sx(ts)
        ys_ = sy(ts)
        as_ = sa(ts)
        # enforce exact endpoints 0 and 90
        xs_[0] = 0.0; ys_[0] = 0.0; as_[0] = 0.0
        xs_[-1] = -2.0; ys_[-1] = 0.0; as_[-1] = 90.0
        return list(zip(xs_, ys_, as_))

    # --- Fast evaluator ---
    def _make_densified(path):
        full = [(0.0, 0.0, 0.0)]
        for p in path:
            full.append((float(p[0]), float(p[1]), float(p[2])))
        full.append((-2.0, 0.0, 90.0))
        dens = [full[0]]
        for i in range(len(full) - 1):
            p1 = full[i]; p2 = full[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            for k in range(m):
                t = (k + 1) / m
                dens.append((
                    p1[0] + (p2[0] - p1[0]) * t,
                    p1[1] + (p2[1] - p1[1]) * t,
                    p1[2] + (p2[2] - p1[2]) * t,
                ))
        dens.append(full[-1])
        return dens

    def evaluate_path_fast(path):
        dens = _make_densified(path)
        n_dens = len(dens)
        angles_rad = np.array([np.deg2rad(d[2]) for d in dens])
        cos_arr = np.cos(angles_rad)
        sin_arr = np.sin(angles_rad)
        cx_arr = np.array([d[0] for d in dens])
        cy_arr = np.array([d[1] for d in dens])

        mask = np.ones(n_points, dtype=bool)
        for i in range(n_dens):
            cos_a = cos_arr[i]; sin_a = sin_arr[i]
            dx = x_flat - cx_arr[i]
            dy = y_flat - cy_arr[i]
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a
            in_arm1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            in_arm2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            mask &= (in_arm1 | in_arm2)
            if not np.any(mask):
                return 0.0
        return float(np.sum(mask)) * 0.0001

    def evaluate_path(path):
        return evaluate_path_fast(path)

    # --- Parameter <-> path conversion ---
    def params_to_path(params, n_wp):
        cx = params[0:N_CTRL]
        cy = params[N_CTRL:2*N_CTRL]
        ca = params[2*N_CTRL:3*N_CTRL]
        sx, sy, sa = _build_spline(cx, cy, ca)
        return _spline_waypoints(sx, sy, sa, n_wp)

    def path_to_params(path):
        # path includes endpoints; take interior control points
        # We fit control points to the interior waypoints of the spline.
        n_wp = len(path)
        ts = np.linspace(0.0, 1.0, n_wp)
        # endpoints fixed; fit interior
        cx = np.array([p[0] for p in path])
        cy = np.array([p[1] for p in path])
        ca = np.array([p[2] for p in path])
        # downsample to N_CTRL control points (keep endpoints)
        if n_wp >= N_CTRL:
            idx = np.linspace(0, n_wp - 1, N_CTRL).astype(int)
        else:
            idx = np.arange(n_wp)
            while len(idx) < N_CTRL:
                idx = np.insert(idx, len(idx) - 1, n_wp - 1)
        return np.concatenate([cx[idx], cy[idx], ca[idx]])

    # --- Initial spline control points (interior only; endpoints fixed) ---
    # Build a reasonable interior shape.
    def make_initial_params(variant=0):
        t = np.linspace(0.0, 1.0, N_CTRL)
        # x goes 0 -> -2, y small bulge, angle 0->90
        base_x = np.linspace(0.0, -2.0, N_CTRL)
        if variant == 0:
            base_y = 0.25 * np.sin(np.pi * t)
        elif variant == 1:
            base_y = 0.35 * np.sin(0.8 * np.pi * t)
        else:
            base_y = 0.2 * np.sin(np.pi * t) + 0.1 * t
        base_a = np.linspace(0.0, 90.0, N_CTRL)
        params = np.concatenate([base_x, base_y, base_a])
        # fix endpoints
        params[0] = 0.0; params[N_CTRL + 0] = 0.0; params[2*N_CTRL + 0] = 0.0
        params[-1] = -2.0; params[N_CTRL + N_CTRL - 1] = 0.0; params[2*N_CTRL + N_CTRL - 1] = 90.0
        return params

    N_WP = 24  # waypoints sampled from spline for evaluation

    # Evaluate a few initial variants
    best_score = 0.0
    best_params = None
    for v in range(3):
        params = make_initial_params(v)
        path = params_to_path(params, N_WP)
        s = evaluate_path(path)
        if s > best_score:
            best_score = s
            best_params = params.copy()

    current_params = best_params.copy()

    # --- Phase 1: random local search on spline control points ---
    np.random.seed(42)
    eps_xy = 0.12
    eps_a = 3.0
    no_improve = 0
    max_no_improve = 20
    phase1_deadline = min(time_limit, start_time + 380.0)
    restart_count = 0
    max_restarts = 4

    while time.time() < phase1_deadline:
        # pick an interior control point (not endpoints)
        idx = np.random.randint(1, N_CTRL - 1)
        dim = np.random.randint(0, 3)  # x, y, or a
        if dim == 0:
            new_val = current_params[idx] + np.random.normal(0, eps_xy)
        elif dim == 1:
            new_val = current_params[N_CTRL + idx] + np.random.normal(0, eps_xy)
        else:
            new_val = np.clip(current_params[2*N_CTRL + idx] + np.random.normal(0, eps_a), 0.0, 90.0)
        test = current_params.copy()
        if dim == 0:
            test[idx] = new_val
        elif dim == 1:
            test[N_CTRL + idx] = new_val
        else:
            test[2*N_CTRL + idx] = new_val
        path = params_to_path(test, N_WP)
        s = evaluate_path(path)
        if s > best_score:
            best_score = s
            best_params = test.copy()
            current_params = test.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= max_no_improve:
                eps_xy *= 0.85
                eps_a *= 0.85
                no_improve = 0
                if eps_xy < 0.01 and eps_a < 0.5:
                    current_params = best_params.copy()
                    eps_xy = 0.25
                    eps_a = 5.0
                    restart_count += 1
                    if restart_count >= max_restarts:
                        # try a fresh variant
                        v = np.random.randint(3)
                        current_params = make_initial_params(v)
                        restart_count = 0

    # --- Phase 2: multi-scale coordinate descent on control points ---
    polish_deadline = time_limit - 20.0
    scales = [
        (0.04, 0.8),
        (0.02, 0.4),
        (0.01, 0.2),
        (0.005, 0.1),
        (0.002, 0.05),
        (0.001, 0.02),
    ]
    for polish_xy, polish_a in scales:
        if time.time() > polish_deadline:
            break
        for pass_iter in range(2):
            if time.time() > polish_deadline:
                break
            improved = False
            for idx in range(1, N_CTRL - 1):
                if time.time() > polish_deadline:
                    break
                best_local = best_score
                best_local_p = current_params.copy()
                for dim, base, step in [(0, idx, polish_xy), (1, N_CTRL + idx, polish_xy), (2, 2*N_CTRL + idx, polish_a)]:
                    for f in (-1.0, -0.5, 0.5, 1.0):
                        val = current_params[base] + f * step
                        if dim == 2:
                            val = np.clip(val, 0.0, 90.0)
                        test = current_params.copy()
                        test[base] = val
                        path = params_to_path(test, N_WP)
                        s = evaluate_path(path)
                        if s > best_local:
                            best_local = s
                            best_local_p = test.copy()
                if best_local > best_score:
                    current_params = best_local_p
                    best_score = best_local
                    best_params = best_local_p.copy()
                    improved = True
            if not improved:
                break

    # --- Phase 3: Nelder-Mead on interior control points ---
    if time.time() < polish_deadline:
        def nm_objective(params):
            try:
                path = params_to_path(params, N_WP)
                return -evaluate_path(path)
            except Exception:
                return 1e10

        # Only optimize interior points; fix endpoints via wrapper.
        def nm_objective_interior(interior):
            try:
                full = best_params.copy()
                full[1:N_CTRL-1] = interior[0:N_CTRL-2]
                full[N_CTRL+1:N_CTRL+N_CTRL-2] = interior[N_CTRL-2:2*(N_CTRL-2)]
                full[2*N_CTRL+1:2*N_CTRL+N_CTRL-2] = np.clip(interior[2*(N_CTRL-2):3*(N_CTRL-2)], 0.0, 90.0)
                path = params_to_path(full, N_WP)
                return -evaluate_path(path)
            except Exception:
                return 1e10

        interior = best_params[1:N_CTRL-1].copy()
        try:
            result = minimize(
                nm_objective_interior,
                interior,
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 0.003, 'fatol': 1e-7, 'adaptive': True}
            )
            new_interior = result.x
            full = best_params.copy()
            full[1:N_CTRL-1] = new_interior[0:N_CTRL-2]
            full[N_CTRL+1:N_CTRL+N_CTRL-2] = new_interior[N_CTRL-2:2*(N_CTRL-2)]
            full[2*N_CTRL+1:2*N_CTRL+N_CTRL-2] = np.clip(new_interior[2*(N_CTRL-2):3*(N_CTRL-2)], 0.0, 90.0)
            path = params_to_path(full, N_WP)
            s = evaluate_path(path)
            if s > best_score:
                best_score = s
                best_params = full.copy()
                current_params = full.copy()
        except Exception:
            pass

    # --- Phase 4: final fine polish ---
    if time.time() < polish_deadline:
        fine_xy = 0.001
        fine_a = 0.02
        for pass_iter in range(3):
            if time.time() > polish_deadline:
                break
            improved = False
            for idx in range(1, N_CTRL - 1):
                if time.time() > polish_deadline:
                    break
                best_local = best_score
                best_local_p = current_params.copy()
                for dim, base, step in [(0, idx, fine_xy), (1, N_CTRL + idx, fine_xy), (2, 2*N_CTRL + idx, fine_a)]:
                    for f in (-1.0, 1.0):
                        val = current_params[base] + f * step
                        if dim == 2:
                            val = np.clip(val, 0.0, 90.0)
                        test = current_params.copy()
                        test[base] = val
                        path = params_to_path(test, N_WP)
                        s = evaluate_path(path)
                        if s > best_local:
                            best_local = s
                            best_local_p = test.copy()
                if best_local > best_score:
                    current_params = best_local_p
                    best_score = best_local
                    best_params = best_local_p.copy()
                    improved = True
            if not improved:
                break

    # --- Final: build output path from best spline ---
    # Use a denser waypoint sampling for the final output to capture the curve.
    final_path = params_to_path(best_params, 30)
    # Ensure endpoints exact and all finite
    final_path[0] = (0.0, 0.0, 0.0)
    final_path[-1] = (-2.0, 0.0, 90.0)
    out = np.array([[float(p[0]), float(p[1]), float(p[2])] for p in final_path])
    if not np.all(np.isfinite(out)):
        # Fallback to a safe known-good path
        out = np.array([
            [0.0, 0.0, 0.0],
            [-0.5, 0.2, 30.0],
            [-1.0, 0.25, 60.0],
            [-1.5, 0.2, 78.0],
            [-2.0, 0.0, 90.0],
        ])
    return out
