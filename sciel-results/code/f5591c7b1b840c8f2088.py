"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    def compute_radii(x, y):
        """Compute the maximum radii for given centers (vectorized)."""
        dist_boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0.0)
        return radii

    def _evaluate(centers, radii):
        """Compute sum of radii if valid packing, else -inf. Vectorized."""
        x, y, r = centers[:, 0], centers[:, 1], radii
        if np.any(x - r < -1e-9) or np.any(x + r > 1.0 + 1e-9):
            return -np.inf
        if np.any(y - r < -1e-9) or np.any(y + r > 1.0 + 1e-9):
            return -np.inf
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        if np.any(dist < r[:, np.newaxis] + r[np.newaxis, :] - 1e-9):
            return -np.inf
        return np.sum(radii)

    def _polish(centers, radii, n_iters=50, tol=1e-12):
        """Iteratively recompute max radii for given positions."""
        radii = radii.copy()
        for _ in range(n_iters):
            changed = False
            for i in range(N):
                dists = np.linalg.norm(centers - centers[i], axis=1)
                dists[i] = np.inf
                # max radius considering other circles' radii
                min_dist_others = np.min(dists - radii)
                min_dist_bound = min(centers[i, 0], 1.0 - centers[i, 0],
                                     centers[i, 1], 1.0 - centers[i, 1])
                new_r = max(1e-10, min(min_dist_others, min_dist_bound) - 1e-10)
                if abs(new_r - radii[i]) > tol:
                    changed = True
                radii[i] = new_r
            if not changed:
                break
        return radii

    def _high_precision_polish(centers, radii, n_iters=100):
        """High-precision polish with tighter tolerances."""
        radii = radii.copy()
        for _ in range(n_iters):
            changed = False
            for i in range(N):
                dists = np.linalg.norm(centers - centers[i], axis=1)
                dists[i] = np.inf
                min_dist_others = np.min(dists - radii)
                min_dist_bound = min(centers[i, 0], 1.0 - centers[i, 0],
                                     centers[i, 1], 1.0 - centers[i, 1])
                new_r = max(1e-12, min(min_dist_others, min_dist_bound) - 1e-12)
                if abs(new_r - radii[i]) > 1e-14:
                    changed = True
                radii[i] = new_r
            if not changed:
                break
        return radii

    def _penalty_objective(vars_, penalty_weight=500.0):
        """Objective: minimize -sum(radii) + penalty for violations. Vectorized."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        obj = -np.sum(R)
        # Pairwise overlap penalty
        dx = X[:, np.newaxis] - X[np.newaxis, :]
        dy = Y[:, np.newaxis] - Y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        overlap = R[:, np.newaxis] + R[np.newaxis, :] - dist
        overlap = np.maximum(overlap, 0.0)
        pen = np.sum(overlap**2)
        # Boundary penalties
        pen += np.sum(np.maximum(-(X - R), 0.0)**2)
        pen += np.sum(np.maximum(-(1.0 - X - R), 0.0)**2)
        pen += np.sum(np.maximum(-(Y - R), 0.0)**2)
        pen += np.sum(np.maximum(-(1.0 - Y - R), 0.0)**2)
        return obj + penalty_weight * pen

    def _penalty_gradient(vars_, penalty_weight=500.0):
        """Analytical gradient of penalty objective. Vectorized."""
        X = vars_[0::3]
        Y = vars_[1::3]
        R = vars_[2::3]
        grad = np.zeros_like(vars_)
        grad[2::3] = -1.0
        w = penalty_weight
        # Pairwise
        dx = X[:, np.newaxis] - X[np.newaxis, :]
        dy = Y[:, np.newaxis] - Y[np.newaxis, :]
        dist = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist, np.inf)
        overlap = R[:, np.newaxis] + R[np.newaxis, :] - dist
        active = overlap > 0.0
        factor = 2.0 * np.where(active, overlap, 0.0)
        inv_dist = np.where(dist > 1e-10, 1.0 / dist, 0.0)
        # grad_X[i] += w * sum_j factor[i,j] * (-dx[i,j] / dist[i,j])
        gx = -w * np.sum(factor * inv_dist * dx, axis=1)
        gy = -w * np.sum(factor * inv_dist * dy, axis=1)
        gr = w * np.sum(factor, axis=1)
        grad[0::3] += gx
        grad[1::3] += gy
        grad[2::3] += gr
        # Boundary
        v1 = X - R
        a1 = v1 < 0.0
        grad[0::3] += w * 2.0 * np.where(a1, v1, 0.0)
        grad[2::3] += w * (-2.0) * np.where(a1, v1, 0.0)
        v2 = 1.0 - X - R
        a2 = v2 < 0.0
        grad[0::3] += w * (-2.0) * np.where(a2, v2, 0.0)
        grad[2::3] += w * (-2.0) * np.where(a2, v2, 0.0)
        v3 = Y - R
        a3 = v3 < 0.0
        grad[1::3] += w * 2.0 * np.where(a3, v3, 0.0)
        grad[2::3] += w * (-2.0) * np.where(a3, v3, 0.0)
        v4 = 1.0 - Y - R
        a4 = v4 < 0.0
        grad[1::3] += w * (-2.0) * np.where(a4, v4, 0.0)
        grad[2::3] += w * (-2.0) * np.where(a4, v4, 0.0)
        return grad

    def _smooth_objective(params, alpha=80.0):
        """Smooth objective using soft-min via log-sum-exp (2-var: x, y)."""
        x = params[0::2]
        y = params[1::2]
        dist_boundary = np.minimum(np.minimum(x, 1.0 - x), np.minimum(y, 1.0 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha
        boundary_terms = np.stack([x, 1.0 - x, y, 1.0 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha
        radii_soft = np.maximum(radii_soft, 0.0)
        return -np.sum(radii_soft)

    def _objective_2var(params):
        """Objective to minimize (negative sum of radii) for 2-var (x, y)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x0[idx] = (i + 0.5) / side
                y0[idx] = (j + 0.5) / side
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x0[k] = 0.5
                y0[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    def hex_init():
        """Hexagonal close-packing initialization, scaled to fit in unit square."""
        pts = []
        for row in range(8):
            for col in range(8):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                pts.append((x, y))
        pts = np.array(pts)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.92 / max(width, height)
        init_pts[:, 0] = (init_pts[:, 0] - min_x) * scale
        init_pts[:, 1] = (init_pts[:, 1] - min_y) * scale
        current_w = init_pts.max(axis=0) - init_pts.min(axis=0)
        offset_x = (1.0 - current_w[0]) / 2
        offset_y = (1.0 - current_w[1]) / 2
        init_pts[:, 0] += offset_x
        init_pts[:, 1] += offset_y
        params = np.zeros(2 * N)
        params[0::2] = init_pts[:, 0]
        params[1::2] = init_pts[:, 1]
        return params

    def _to_3var(params_2var, centers=None, radii=None):
        """Convert 2-var params to 3-var (x, y, r) with polished radii."""
        if centers is None:
            centers = np.column_stack((params_2var[0::2], params_2var[1::2]))
        if radii is None:
            radii = compute_radii(centers[:, 0], centers[:, 1])
        radii = _polish(centers, radii, n_iters=30)
        out = np.zeros(3 * N)
        out[0::3] = centers[:, 0]
        out[1::3] = centers[:, 1]
        out[2::3] = radii
        return out

    # Choose initial parameters
    params_grid = grid_init()
    params_hex = hex_init()
    obj_grid = _objective_2var(params_grid)
    obj_hex = _objective_2var(params_hex)
    params0 = params_hex if obj_hex < obj_grid else params_grid

    best_centers = np.column_stack((params0[0::2], params0[1::2]))
    best_radii = compute_radii(best_centers[:, 0], best_centers[:, 1])
    best_score = _evaluate(best_centers, best_radii)

    bounds_2var = [(0.0, 1.0)] * (2 * N)
    bounds_3var = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * N

    # Phase 1: Powell on 2-var objective
    try:
        res = minimize(
            _objective_2var, params0, method='Powell', bounds=bounds_2var,
            options={'maxiter': 4000, 'xtol': 1e-12, 'ftol': 1e-12}
        )
        c = np.column_stack((res.x[0::2], res.x[1::2]))
        r = compute_radii(c[:, 0], c[:, 1])
        r = _polish(c, r, n_iters=30)
        s = _evaluate(c, r)
        if s > best_score:
            best_centers, best_radii, best_score = c, r, s
    except Exception:
        pass

    # Phase 2: L-BFGS-B on 3-var penalty objective from multiple starts
    np.random.seed(42)
    for restart in range(12):
        if time.time() > deadline - 30:
            break
        if restart == 0:
            p3 = _to_3var(None, best_centers.copy(), best_radii.copy())
        elif restart == 1:
            p3 = _to_3var(hex_init())
        elif restart == 2:
            p3 = _to_3var(grid_init())
        else:
            # Jittered hex
            rng = np.random.RandomState(1000 + restart)
            c = best_centers.copy()
            c += rng.normal(0, 0.02, size=(N, 2))
            c = np.clip(c, 0.01, 0.99)
            r = compute_radii(c[:, 0], c[:, 1])
            p3 = _to_3var(None, c, r)
        try:
            res = minimize(
                _penalty_objective, p3, jac=_penalty_gradient,
                method='L-BFGS-B', bounds=bounds_3var,
                options={'maxiter': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            c = np.column_stack((res.x[0::3], res.x[1::3]))
            r = np.maximum(res.x[2::3], 0.0)
            r = _polish(c, r, n_iters=50)
            s = _evaluate(c, r)
            if s > best_score:
                best_centers, best_radii, best_score = c, r, s
        except Exception:
            pass

    # Phase 3: Powell refinement on 2-var
    if time.time() < deadline - 20:
        p2 = np.zeros(2 * N)
        p2[0::2] = best_centers[:, 0]
        p2[1::2] = best_centers[:, 1]
        try:
            res = minimize(
                _objective_2var, p2, method='Powell', bounds=bounds_2var,
                options={'maxiter': 5000, 'xtol': 1e-13, 'ftol': 1e-13}
            )
            c = np.column_stack((res.x[0::2], res.x[1::2]))
            r = compute_radii(c[:, 0], c[:, 1])
            r = _polish(c, r, n_iters=50)
            s = _evaluate(c, r)
            if s > best_score:
                best_centers, best_radii, best_score = c, r, s
        except Exception:
            pass

    # Phase 4: Local perturbation refinement (vectorized scoring)
    if time.time() < deadline - 12:
        x_best = best_centers[:, 0].copy()
        y_best = best_centers[:, 1].copy()
        r_best = best_radii.copy()
        sum_best = best_score

        for _ in range(150):
            if time.time() > deadline - 8:
                break
            improved = False
            for i in range(N):
                for step_size in [0.005, 0.003, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
                    for dx_try in [-step_size, 0, step_size]:
                        for dy_try in [-step_size, 0, step_size]:
                            if dx_try == 0 and dy_try == 0:
                                continue
                            x_try = x_best.copy()
                            y_try = y_best.copy()
                            x_try[i] += dx_try
                            y_try[i] += dy_try
                            if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                                continue
                            c_try = np.column_stack((x_try, y_try))
                            r_try = compute_radii(x_try, y_try)
                            r_try = _polish(c_try, r_try, n_iters=20)
                            s_try = _evaluate(c_try, r_try)
                            if s_try > sum_best + 1e-14:
                                x_best, y_best, r_best = x_try, y_try, r_try
                                sum_best = s_try
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
            if not improved:
                break

        best_centers = np.column_stack((x_best, y_best))
        best_radii = r_best
        best_score = sum_best

    # Phase 5: Final high-precision polish
    if time.time() < deadline - 5:
        best_radii = _high_precision_polish(best_centers, best_radii, n_iters=100)
        s = _evaluate(best_centers, best_radii)
        if s > best_score:
            best_score = s

        # Final local refinement
        if time.time() < deadline - 2:
            x_best = best_centers[:, 0].copy()
            y_best = best_centers[:, 1].copy()
            r_best = best_radii.copy()
            sum_best = best_score
            for _ in range(60):
                if time.time() > deadline - 1:
                    break
                improved = False
                for i in range(N):
                    for step_size in [0.0005, 0.0002, 0.0001, 0.00005]:
                        for dx_try in [-step_size, 0, step_size]:
                            for dy_try in [-step_size, 0, step_size]:
                                if dx_try == 0 and dy_try == 0:
                                    continue
                                x_try = x_best.copy()
                                y_try = y_best.copy()
                                x_try[i] += dx_try
                                y_try[i] += dy_try
                                if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                                    continue
                                c_try = np.column_stack((x_try, y_try))
                                r_try = compute_radii(x_try, y_try)
                                r_try = _high_precision_polish(c_try, r_try, n_iters=50)
                                s_try = _evaluate(c_try, r_try)
                                if s_try > sum_best + 1e-15:
                                    x_best, y_best, r_best = x_try, y_try, r_try
                                    sum_best = s_try
                                    improved = True
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if not improved:
                    break
            best_centers = np.column_stack((x_best, y_best))
            best_radii = r_best
            best_score = sum_best

    # Final construction
    radii = best_radii * 0.999999
    result = np.column_stack((best_centers[:, 0], best_centers[:, 1], radii))
    return result

# EVOLVE-BLOCK-END
