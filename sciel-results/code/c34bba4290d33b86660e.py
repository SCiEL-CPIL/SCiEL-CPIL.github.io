"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize, linprog
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 145  # safety margin (150s limit)

    def compute_radii(x, y):
        """Compute the maximum radii for given centers."""
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, np.inf)
        min_dist = np.min(dist_matrix, axis=1)
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        return radii

    def lp_radii(x, y):
        """
        Use Linear Programming to maximize sum of radii for fixed centers.
        Constraints: r_i <= dist_boundary_i, r_i + r_j <= dist_ij for all i<j
        """
        n = len(x)
        # Variables: r_0, ..., r_{n-1}
        # Objective: maximize sum(r) -> minimize -sum(r)
        c = -np.ones(n)
        
        # Inequality constraints: A_ub @ r <= b_ub
        # r_i <= dist_boundary_i
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # r_i + r_j <= dist_ij
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        A_ub_rows = []
        b_ub_vals = []
        
        # Boundary constraints
        for i in range(n):
            row = np.zeros(n)
            row[i] = 1.0
            A_ub_rows.append(row)
            b_ub_vals.append(dist_boundary[i])
        
        # Pairwise constraints
        for i in range(n):
            for j in range(i + 1, n):
                row = np.zeros(n)
                row[i] = 1.0
                row[j] = 1.0
                A_ub_rows.append(row)
                b_ub_vals.append(dist_matrix[i, j])
        
        A_ub = np.array(A_ub_rows)
        b_ub = np.array(b_ub_vals)
        
        bounds = [(0, None)] * n
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.success:
                return -res.fun  # sum of radii
            else:
                return compute_radii(x, y).sum()
        except Exception:
            return compute_radii(x, y).sum()

    def objective_lp(params):
        """Objective to minimize (negative sum of radii) using LP."""
        x = params[0::2]
        y = params[1::2]
        # Check time
        if time.time() > deadline - 5:
            return 0  # Return neutral to stop optimization
        return -lp_radii(x, y)

    def objective_fast(params):
        """Fast objective for initial optimization (no LP)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def smooth_objective(params, alpha=200.0):
        """Smooth objective using soft-min via log-sum-exp."""
        x = params[0::2]
        y = params[1::2]

        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))

        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(dist_matrix, 1e10)

        # Soft-min of pairwise distances
        scaled = -alpha * dist_matrix
        row_max = np.max(scaled, axis=1, keepdims=True)
        exp_scaled = np.exp(scaled - row_max)
        np.fill_diagonal(exp_scaled, 0.0)
        log_sum = np.log(np.sum(exp_scaled, axis=1)) + row_max[:, 0]
        min_dist_soft = -log_sum / alpha

        # Soft-min of boundary distances
        boundary_terms = np.stack([x, 1 - x, y, 1 - y], axis=1)
        scaled_b = -alpha * boundary_terms
        row_max_b = np.max(scaled_b, axis=1, keepdims=True)
        exp_b = np.exp(scaled_b - row_max_b)
        log_sum_b = np.log(np.sum(exp_b, axis=1)) + row_max_b[:, 0]
        min_boundary_soft = -log_sum_b / alpha

        # Soft-min of (boundary, dist/2)
        a = min_boundary_soft
        b = min_dist_soft / 2.0
        scaled_ab = -alpha * np.stack([a, b], axis=1)
        row_max_ab = np.max(scaled_ab, axis=1, keepdims=True)
        exp_ab = np.exp(scaled_ab - row_max_ab)
        log_sum_ab = np.log(np.sum(exp_ab, axis=1)) + row_max_ab[:, 0]
        radii_soft = -log_sum_ab / alpha

        radii_soft = np.maximum(radii_soft, 0)
        return -np.sum(radii_soft)

    def grid_init():
        side = int(np.ceil(np.sqrt(N)))  # 6
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

    def hex_init_scaled():
        """Hexagonal lattice scaled to fit in unit square, taking N closest to center."""
        points = []
        for row in range(10):
            for col in range(10):
                x = col + (0.5 if row % 2 else 0)
                y = row * np.sqrt(3) / 2
                points.append((x, y))
        pts = np.array(points)
        center = pts.mean(axis=0)
        pts = pts - center
        dists = np.linalg.norm(pts, axis=1)
        idx = np.argsort(dists)[:N]
        init_pts = pts[idx].copy()
        min_x, min_y = init_pts.min(axis=0)
        max_x, max_y = init_pts.max(axis=0)
        width = max_x - min_x
        height = max_y - min_y
        scale = 0.9 / max(width, height)
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

    def random_init(seed):
        rng = np.random.RandomState(seed)
        params = np.zeros(2 * N)
        params[0::2] = rng.uniform(0.05, 0.95, N)
        params[1::2] = rng.uniform(0.05, 0.95, N)
        return params

    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = grid_init()
    best_obj = objective_fast(best_params)

    # Phase 1: Powell from multiple initializations with fast objective
    initializations = [grid_init(), hex_init_scaled()]
    for init_params in initializations:
        if time.time() > deadline - 80:
            break
        try:
            res = minimize(
                objective_fast, init_params, method='Powell', bounds=bounds,
                options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 2: Smooth objective optimization with L-BFGS-B from multiple starts
    np.random.seed(42)
    
    for restart in range(12):
        if time.time() > deadline - 60:
            break

        if restart == 0:
            params_start = best_params.copy()
        elif restart == 1:
            params_start = grid_init()
        elif restart == 2:
            params_start = hex_init_scaled()
        elif restart == 3:
            # Rotated hex
            x0, y0 = hex_init_scaled()[0::2], hex_init_scaled()[1::2]
            cx, cy = 0.5, 0.5
            angle = np.pi / 6
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = (x0 - cx) * cos_a - (y0 - cy) * sin_a + cx
            yr = (x0 - cx) * sin_a + (y0 - cy) * cos_a + cy
            params_start = np.zeros(2 * N)
            params_start[0::2] = np.clip(xr, 0.01, 0.99)
            params_start[1::2] = np.clip(yr, 0.01, 0.99)
        elif restart == 4:
            params_start = random_init(100)
        elif restart == 5:
            params_start = random_init(200)
        elif restart == 6:
            params_start = random_init(300)
        else:
            # Jittered best
            params_start = best_params + np.random.normal(0, 0.02, 2 * N)
            params_start = np.clip(params_start, 0.01, 0.99)

        try:
            res = minimize(
                smooth_objective, params_start, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            true_obj = objective_fast(res.x)
            if true_obj < best_obj:
                best_params = res.x
                best_obj = true_obj
        except Exception:
            pass

    # Phase 3: Powell refinement from best smooth result with fast objective
    if time.time() < deadline - 40:
        try:
            res = minimize(
                objective_fast, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 3000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 4: LP-based local refinement
    # Use Powell with LP objective for final precision
    if time.time() < deadline - 15:
        try:
            res = minimize(
                objective_lp, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 1000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 5: Final high-precision polish with fast objective
    if time.time() < deadline - 5:
        try:
            res = minimize(
                objective_fast, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 5000, 'xtol': 1e-14, 'ftol': 1e-14}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)

    # Shrink radii slightly for floating point safety
    radii *= 0.999999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
