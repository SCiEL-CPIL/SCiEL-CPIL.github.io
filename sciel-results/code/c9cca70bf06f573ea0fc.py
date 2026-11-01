"""Circle packing in unit square n=32."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 32

# EVOLVE-BLOCK-START

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin (150s limit)

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

    def sum_radii(x, y):
        """Sum of radii (the value we want to maximize)."""
        return np.sum(compute_radii(x, y))

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        return -sum_radii(x, y)

    def hex_init():
        """Hexagonal lattice initialization for better packing density."""
        x = np.zeros(N)
        y = np.zeros(N)
        # Hexagonal packing: rows offset by half spacing
        # Try to fit 32 circles in a hex pattern
        # 6 columns, ~6 rows with offsets
        cols = 6
        rows = 6
        dx = 1.0 / cols
        dy = 1.0 / (rows * np.sqrt(3) / 2)
        idx = 0
        for r in range(rows):
            for c in range(cols):
                if idx >= N:
                    break
                # Offset odd rows by half dx
                x[idx] = (c + (0.5 if r % 2 == 1 else 0)) * dx
                x[idx] = np.clip(x[idx], 0.01, 0.99)
                y[idx] = (r + 0.5) * dy
                y[idx] = np.clip(y[idx], 0.01, 0.99)
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x[k] = 0.5
                y[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x
        params[1::2] = y
        return params

    def grid_init():
        """Square grid initialization."""
        side = int(np.ceil(np.sqrt(N)))  # 6
        x = np.zeros(N)
        y = np.zeros(N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                x[idx] = (i + 0.5) / side
                y[idx] = (j + 0.5) / side
                idx += 1
        if idx < N:
            for k in range(idx, N):
                x[k] = 0.5
                y[k] = 0.5
        params = np.zeros(2 * N)
        params[0::2] = x
        params[1::2] = y
        return params

    def random_init(seed):
        """Random initialization."""
        rng = np.random.RandomState(seed)
        x = rng.uniform(0.05, 0.95, N)
        y = rng.uniform(0.05, 0.95, N)
        params = np.zeros(2 * N)
        params[0::2] = x
        params[1::2] = y
        return params

    def smooth_objective(params, alpha=100.0):
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

    def powell_refine(params, maxiter=5000):
        """Refine parameters using Powell's method."""
        bounds = [(0.0, 1.0)] * (2 * N)
        try:
            res = minimize(
                objective, params, method='Powell', bounds=bounds,
                options={'maxiter': maxiter, 'xtol': 1e-12, 'ftol': 1e-12}
            )
            return res.x
        except Exception:
            return params

    def lbfgsb_refine(params, maxiter=3000):
        """Refine parameters using L-BFGS-B with smooth objective."""
        bounds = [(0.0, 1.0)] * (2 * N)
        try:
            res = minimize(
                smooth_objective, params, method='L-BFGS-B', bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            return res.x
        except Exception:
            return params

    def coordinate_descent(x, y, max_iter=200):
        """Coordinate descent: move one circle at a time to improve sum of radii."""
        x = x.copy()
        y = y.copy()
        current_sum = sum_radii(x, y)
        
        step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
        
        for it in range(max_iter):
            if time.time() > deadline - 10:
                break
            improved = False
            for i in range(N):
                for step in step_sizes:
                    for dx in [-step, 0, step]:
                        for dy in [-step, 0, step]:
                            if dx == 0 and dy == 0:
                                continue
                            x_try = x.copy()
                            y_try = y.copy()
                            x_try[i] += dx
                            y_try[i] += dy
                            if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                                continue
                            s_try = sum_radii(x_try, y_try)
                            if s_try > current_sum + 1e-15:
                                x, y = x_try, y_try
                                current_sum = s_try
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
        return x, y, current_sum

    # Phase 1: Generate diverse initializations
    inits = [
        hex_init(),
        grid_init(),
        random_init(42),
        random_init(123),
        random_init(789),
        random_init(321),
    ]
    
    # Add jittered hex
    for seed in [10, 20]:
        rng = np.random.RandomState(seed)
        p = hex_init()
        p += rng.normal(0, 0.03, size=p.shape)
        p = np.clip(p, 0.01, 0.99)
        inits.append(p)
    
    # Phase 2: Optimize each initialization
    best_params = None
    best_obj = np.inf
    
    for i, init in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        # L-BFGS-B with smooth objective
        p1 = lbfgsb_refine(init, maxiter=2000)
        obj1 = objective(p1)
        
        # Powell refinement
        p2 = powell_refine(p1, maxiter=3000)
        obj2 = objective(p2)
        
        if obj2 < best_obj:
            best_params = p2
            best_obj = obj2
    
    # Phase 3: Coordinate descent refinement
    x_best = best_params[0::2].copy()
    y_best = best_params[1::2].copy()
    x_best, y_best, sum_best = coordinate_descent(x_best, y_best, max_iter=100)
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        final_params = np.zeros(2 * N)
        final_params[0::2] = x_best
        final_params[1::2] = y_best
        final_params = powell_refine(final_params, maxiter=8000)
        final_obj = objective(final_params)
        if final_obj < best_obj:
            best_params = final_params
            best_obj = final_obj
    
    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)
    
    # Shrink radii slightly for floating point safety
    radii *= 0.999999
    
    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
