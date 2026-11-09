import numpy as np
import time
from scipy.optimize import minimize

def moving_sofa_2d() -> np.ndarray:
    t_start = time.time()
    time_limit = 600.0
    margin = 15.0

    # Pre-compute the sofa grid
    x_g = np.arange(-3.0, 1.5, 0.01)
    y_g = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(x_g, y_g)
    x_flat = X.flatten()
    y_flat = Y.flatten()
    n_points = x_flat.size

    def check_path(path_pts):
        placements = [(0.0, 0.0, 0.0)]
        prev_x, prev_y, prev_a = 0.0, 0.0, 0.0

        def add_interpolations(x1, y1, a1, x2, y2, a2):
            m = max(1, int(np.ceil(abs(a2 - a1))))
            for k in range(1, m + 1):
                t = k / m
                x = x1 + t * (x2 - x1)
                y = y1 + t * (y2 - y1)
                a = a1 + t * (a2 - a1)
                placements.append((x, y, a))

        for p in path_pts:
            px, py, pa = p
            add_interpolations(prev_x, prev_y, prev_a, px, py, pa)
            prev_x, prev_y, prev_a = px, py, pa

        add_interpolations(prev_x, prev_y, prev_a, -2.0, 0.0, 90.0)

        mask = np.ones(n_points, dtype=bool)

        for (cx, cy, ca) in placements:
            if not np.any(mask):
                return 0.0

            rad = np.deg2rad(ca)
            cos_a = np.cos(rad)
            sin_a = np.sin(rad)

            idx = np.where(mask)[0]
            if len(idx) == 0:
                return 0.0

            dx = x_flat[idx] - cx
            dy = y_flat[idx] - cy
            u = dx * cos_a + dy * sin_a
            v = -dx * sin_a + dy * cos_a

            cond1 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            cond2 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            valid = cond1 | cond2

            if not np.all(valid):
                new_mask = np.zeros(n_points, dtype=bool)
                new_mask[idx[valid]] = True
                mask = new_mask
                if not np.any(mask):
                    return 0.0
        count = np.sum(mask)
        return count * 0.0001

    def make_arc_path(params, n_pts=20):
        # params: [r, h, s]
        # r: radius of arc
        # h: vertical offset of arc center relative to start
        # s: scaling factor for horizontal displacement
        r, h, s = params
        # Arc from (0,0) angle 0 to (-2, 0) angle 90
        # We model the path as a circular arc
        # The arc must be tangent to horizontal at start and vertical at end
        # For a circular arc with radius r, center at (cx, cy)
        # At start (0,0), tangent is horizontal, so center is at (0, r) or (0, -r)
        # At end (-2, 0), tangent is vertical, so center is at (-2, cy)
        # For the arc to connect smoothly, the center must be equidistant
        # Let's parameterize differently: use the angle swept and radius
        # Actually, let's use a simpler parameterization:
        # The path is defined by x(t), y(t) where t in [0, 1]
        # x(t) = -2 * t
        # y(t) = h * sin(pi * t) + s * t * (1 - t)
        # But the directive suggests circular arc. Let's try:
        # Center at (cx, cy), radius r
        # Start: (0,0), angle 0 (tangent horizontal) -> center at (0, r)
        # End: (-2, 0), angle 90 (tangent vertical) -> center at (-2, r)
        # This doesn't work directly. Let's use a different approach.
        # Use a Bezier-like curve with 3 control points
        # P0 = (0, 0), P1 = (cx, cy), P2 = (-2, 0)
        # The angle at t is determined by the derivative
        # For simplicity, let's use quadratic Bezier
        cx, cy, s = params
        t_vals = np.linspace(0, 1, n_pts + 2)[1:-1]
        # Quadratic Bezier: B(t) = (1-t)^2 P0 + 2t(1-t) P1 + t^2 P2
        xs = 2 * t_vals * (1 - t_vals) * cx + t_vals**2 * (-2.0)
        ys = 2 * t_vals * (1 - t_vals) * cy
        # Angle: derivative is B'(t) = 2(1-t)(P1-P0) + 2t(P2-P1)
        # B'(t) = 2(1-t)(cx, cy) + 2t(-2-cx, -cy)
        dx_dt = 2 * (1 - t_vals) * cx + 2 * t_vals * (-2.0 - cx)
        dy_dt = 2 * (1 - t_vals) * cy + 2 * t_vals * (-cy)
        # The angle of the corridor is the angle of the tangent
        # But we need the angle such that the corridor orientation matches
        # The corridor's local frame is rotated by 'a' degrees
        # At start a=0, at end a=90
        # The tangent angle should go from 0 to 90
        # atan2(dy_dt, dx_dt) gives the tangent angle in radians
        # But we need to map this to the corridor angle 'a'
        # Actually, the corridor angle 'a' is the rotation of the corridor frame
        # The tangent of the path should align with the corridor's main axis
        # Let's just use a linear interpolation of angle for now, and optimize x, y
        as_ = 90.0 * t_vals
        return np.column_stack((xs, ys, as_))

    def make_cubic_path(params, n_pts=20):
        # params: [cx2, cx3, cy2, cy3]
        cx2, cx3, cy2, cy3 = params
        c1 = -2 - cx2 - cx3
        d1 = -cy2 - cy3

        t_vals = np.linspace(0, 1, n_pts + 2)[1:-1]
        xs = c1 * t_vals + cx2 * t_vals**2 + cx3 * t_vals**3
        ys = d1 * t_vals + cy2 * t_vals**2 + cy3 * t_vals**3
        as_ = 90.0 * t_vals
        return np.column_stack((xs, ys, as_))

    def objective(params, path_func):
        path = path_func(params)
        return -check_path(path)

    best_path = None
    best_score = -1.0

    # Phase 1: Grid search over quadratic Bezier control points
    cx_vals = np.linspace(-3, 1, 8)
    cy_vals = np.linspace(-2, 2, 8)

    for cx in cx_vals:
        for cy in cy_vals:
            if time.time() - t_start > time_limit - margin:
                break
            params = np.array([cx, cy, 0.0])
            path = make_arc_path(params)
            score = check_path(path)
            if score > best_score:
                best_score = score
                best_path = path
        if time.time() - t_start > time_limit - margin:
            break

    # Phase 2: Grid search over cubic paths
    c2_vals = np.linspace(-2, 2, 6)
    c3_vals = np.linspace(-1, 1, 4)
    d2_vals = np.linspace(-2, 2, 6)
    d3_vals = np.linspace(-1, 1, 4)

    for c2 in c2_vals:
        for d2 in d2_vals:
            for c3 in c3_vals:
                for d3 in d3_vals:
                    if time.time() - t_start > time_limit - margin - 30:
                        break
                    params = np.array([c2, c3, d2, d3])
                    path = make_cubic_path(params)
                    score = check_path(path)
                    if score > best_score:
                        best_score = score
                        best_path = path
                if time.time() - t_start > time_limit - margin - 30:
                    break
            if time.time() - t_start > time_limit - margin - 30:
                break
        if time.time() - t_start > time_limit - margin - 30:
            break

    # Phase 3: Local optimization
    if best_path is not None and best_score > 0:
        # Try to find which params gave the best path
        # Re-run grid search tracking best params
        best_params = None
        best_path_type = None

        for cx in cx_vals:
            for cy in cy_vals:
                if time.time() - t_start > time_limit - margin - 60:
                    break
                params = np.array([cx, cy, 0.0])
                path = make_arc_path(params)
                score = check_path(path)
                if score > best_score - 1e-10:
                    best_score = score
                    best_params = params
                    best_path_type = 'arc'
            if time.time() - t_start > time_limit - margin - 60:
                break

        for c2 in c2_vals:
            for d2 in d2_vals:
                for c3 in c3_vals:
                    for d3 in d3_vals:
                        if time.time() - t_start > time_limit - margin - 60:
                            break
                        params = np.array([c2, c3, d2, d3])
                        path = make_cubic_path(params)
                        score = check_path(path)
                        if score > best_score - 1e-10:
                            best_score = score
                            best_params = params
                            best_path_type = 'cubic'
                    if time.time() - t_start > time_limit - margin - 60:
                        break
                if time.time() - t_start > time_limit - margin - 60:
                    break
            if time.time() - t_start > time_limit - margin - 60:
                break

        if best_params is not None:
            if best_path_type == 'arc':
                result = minimize(lambda p: objective(p, make_arc_path), best_params,
                                method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 0.005, 'fatol': 0.00005})
                if result.fun < -best_score:
                    best_score = -result.fun
                    best_path = make_arc_path(result.x)
            elif best_path_type == 'cubic':
                result = minimize(lambda p: objective(p, make_cubic_path), best_params,
                                method='Nelder-Mead',
                                options={'maxiter': 300, 'xatol': 0.005, 'fatol': 0.00005})
                if result.fun < -best_score:
                    best_score = -result.fun
                    best_path = make_cubic_path(result.x)

    # Fallback: linear path
    if best_path is None or best_score <= 0:
        t_vals = np.linspace(0, 1, 15)[1:-1]
        xs = -2.0 * t_vals
        ys = np.zeros(len(t_vals))
        as_ = 90.0 * t_vals
        best_path = np.column_stack((xs, ys, as_))

    return best_path
