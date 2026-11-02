"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 142  # safety margin (150s limit)

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

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    def smooth_objective(params, alpha=200.0):
        """Smooth objective using soft-min via log-sum-exp for gradient-based methods."""
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

    def voronoi_inradii(x, y):
        """
        Compute inradii using Voronoi cells.
        For each point, the inradius is the minimum distance to:
        1. The boundary of the unit square
        2. The Voronoi edges (perpendicular bisectors with neighboring points)
        
        This is equivalent to: min(dist_to_boundary, min_j dist(i,j)/2)
        which is what compute_radii does, but we use this name for clarity.
        """
        return compute_radii(x, y)

    # Grid initialization
    side = int(np.ceil(np.sqrt(N)))  # 6
    def grid_init():
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
        """Hexagonal close-packing initialization, tuned for n=32."""
        x0 = np.zeros(N)
        y0 = np.zeros(N)
        idx = 0
        # Use a 6x6 hex grid to get exactly 36 points, take first 32
        # Spacing chosen so circles fit nicely
        s = 1.0 / 5.5  # ~0.182
        for row in range(6):
            for col in range(6):
                if idx >= N:
                    break
                x0[idx] = (col + 0.5) * s
                y0[idx] = (row + 0.5) * s * np.sqrt(3) / 2
                if row % 2 == 1:
                    x0[idx] += s / 2
                # Center the hex grid
                x0[idx] = x0[idx] - 0.5 + 0.5
                y0[idx] = y0[idx] - 0.5 + 0.5
                x0[idx] = np.clip(x0[idx], 0.02, 0.98)
                y0[idx] = np.clip(y0[idx], 0.02, 0.98)
                idx += 1
        params = np.zeros(2 * N)
        params[0::2] = x0
        params[1::2] = y0
        return params

    params0 = grid_init()
    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Phase 1: Powell from multiple initializations
    initializations = [grid_init(), hex_init()]
    for init_params in initializations:
        if time.time() > deadline - 50:
            break
        try:
            res = minimize(
                objective, init_params, method='Powell', bounds=bounds,
                options={'maxiter': 6000, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 2: Smooth objective optimization with L-BFGS-B from multiple starts
    np.random.seed(42)
    
    # Use smooth objective with L-BFGS-B for gradient-based optimization
    for restart in range(16):
        if time.time() > deadline - 40:
            break

        if restart == 0:
            params_start = best_params.copy()
        elif restart == 1:
            params_start = grid_init()
        elif restart == 2:
            params_start = hex_init()
        elif restart == 3:
            # Rotated hex
            x0, y0 = hex_init()[0::2], hex_init()[1::2]
            # Rotate 30 degrees
            cx, cy = 0.5, 0.5
            angle = np.pi / 6
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = (x0 - cx) * cos_a - (y0 - cy) * sin_a + cx
            yr = (x0 - cx) * sin_a + (y0 - cy) * cos_a + cy
            params_start = np.zeros(2 * N)
            params_start[0::2] = np.clip(xr, 0.01, 0.99)
            params_start[1::2] = np.clip(yr, 0.01, 0.99)
        elif restart == 4:
            # Rotated hex 60 degrees
            x0, y0 = hex_init()[0::2], hex_init()[1::2]
            cx, cy = 0.5, 0.5
            angle = np.pi / 3
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            xr = (x0 - cx) * cos_a - (y0 - cy) * sin_a + cx
            yr = (x0 - cx) * sin_a + (y0 - cy) * cos_a + cy
            params_start = np.zeros(2 * N)
            params_start[0::2] = np.clip(xr, 0.01, 0.99)
            params_start[1::2] = np.clip(yr, 0.01, 0.99)
        elif restart == 5:
            # Scaled hex (tighter)
            x0, y0 = hex_init()[0::2], hex_init()[1::2]
            cx, cy = 0.5, 0.5
            scale = 0.92
            xr = (x0 - cx) * scale + cx
            yr = (y0 - cy) * scale + cy
            params_start = np.zeros(2 * N)
            params_start[0::2] = np.clip(xr, 0.01, 0.99)
            params_start[1::2] = np.clip(yr, 0.01, 0.99)
        else:
            # Jittered grid with varying noise
            noise_scale = 0.03 + 0.02 * (restart - 6)
            x_rand = (np.arange(side) + 0.5 + np.random.randn(side) * noise_scale) / side
            y_rand = (np.arange(side) + 0.5 + np.random.randn(side) * noise_scale) / side
            params_start = np.zeros(2 * N)
            idx = 0
            for i in range(side):
                for j in range(side):
                    if idx >= N:
                        break
                    params_start[2 * idx] = np.clip(x_rand[i], 0.01, 0.99)
                    params_start[2 * idx + 1] = np.clip(y_rand[j], 0.01, 0.99)
                    idx += 1
            if idx < N:
                for k in range(idx, N):
                    params_start[2 * k] = 0.5
                    params_start[2 * k + 1] = 0.5

        try:
            # Use smooth objective with L-BFGS-B
            res = minimize(
                smooth_objective, params_start, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            # Evaluate with the true (non-smooth) objective
            true_obj = objective(res.x)
            if true_obj < best_obj:
                best_params = res.x
                best_obj = true_obj
        except Exception:
            pass

    # Phase 3: Powell refinement from best smooth result
    if time.time() < deadline - 30:
        try:
            res = minimize(
                objective, best_params, method='Powell', bounds=bounds,
                options={'maxiter': 6000, 'xtol': 1e-13, 'ftol': 1e-13}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 4: Local refinement using Nelder-Mead on true objective
    if time.time() < deadline - 20:
        for nm_restart in range(5):
            if time.time() > deadline - 10:
                break
            try:
                if nm_restart == 0:
                    nm_start = best_params.copy()
                else:
                    # Small random perturbation of best params
                    nm_start = best_params.copy()
                    nm_start += np.random.randn(2 * N) * 0.002
                    nm_start = np.clip(nm_start, 0.0, 1.0)
                res_nm = minimize(
                    objective, nm_start, method='Nelder-Mead',
                    options={'maxiter': 3000, 'xatol': 1e-14, 'fatol': 1e-14,
                             'adaptive': True}
                )
                if res_nm.fun < best_obj:
                    best_params = res_nm.x
                    best_obj = res_nm.fun
            except Exception:
                pass

    # Phase 5: Final high-precision polish with multiple Powell passes
    if time.time() < deadline - 5:
        for polish_round in range(3):
            if time.time() > deadline - 2:
                break
            try:
                res = minimize(
                    objective, best_params, method='Powell', bounds=bounds,
                    options={'maxiter': 10000, 'xtol': 1e-15, 'ftol': 1e-15}
                )
                if res.fun < best_obj - 1e-15:
                    best_params = res.x
                    best_obj = res.fun
                else:
                    break
            except Exception:
                break

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)

    # Shrink radii slightly for floating point safety
    radii *= 0.999999

    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
