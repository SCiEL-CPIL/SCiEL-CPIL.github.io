"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 32

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize, linprog
import time
import itertools

def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 135  # 135s out of 150s limit, safety margin

    # We will use a combined approach:
    # 1. Start with a good grid initialization.
    # 2. Use a non-smooth objective (sum of radii) optimized via Powell.
    # 3. Add a linear programming step to potentially improve packing by
    #    solving a relaxed problem or using LP to guide the optimization.
    # 4. Use local search with small perturbations.

    def compute_radii(x, y):
        """Compute the maximum radii for given centers."""
        # Distance to boundaries
        dist_boundary = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # Pairwise distances
        dx = x[:, np.newaxis] - x[np.newaxis, :]
        dy = y[:, np.newaxis] - y[np.newaxis, :]
        dist_matrix = np.sqrt(dx**2 + dy**2)
        
        # Set diagonal to inf
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Minimum distance to any other circle
        min_dist = np.min(dist_matrix, axis=1)
        
        # Radius is limited by boundary and half the min distance to other circles
        radii = np.minimum(dist_boundary, min_dist / 2.0)
        radii = np.maximum(radii, 0)
        
        return radii

    def objective(params):
        """Objective to minimize (negative sum of radii)."""
        x = params[0::2]
        y = params[1::2]
        radii = compute_radii(x, y)
        return -np.sum(radii)

    # Initial guess: Grid
    side = int(np.ceil(np.sqrt(N)))  # 6 for N=32
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

    params0 = np.zeros(2 * N)
    params0[0::2] = x0
    params0[1::2] = y0

    bounds = [(0.0, 1.0)] * (2 * N)

    best_params = params0.copy()
    best_obj = objective(params0)

    # Phase 1: Powell optimization from grid start
    try:
        res = minimize(
            objective,
            params0,
            method='Powell',
            bounds=bounds,
            options={'maxiter': 3000, 'xtol': 1e-10, 'ftol': 1e-10}
        )
        if res.fun < best_obj:
            best_params = res.x
            best_obj = res.fun
    except Exception:
        pass

    # Phase 2: Try linear programming relaxation to get a better configuration
    # The LP approach: maximize sum(r_i) subject to:
    #   r_i <= x_i, r_i <= 1-x_i, r_i <= y_i, r_i <= 1-y_i
    #   r_i + r_j <= dist(i,j)  (linearized as r_i + r_j <= |x_i-x_j| + |y_i-y_j| is not linear)
    # Instead, use the constraint: (x_i-x_j)^2 + (y_i-y_j)^2 >= (r_i+r_j)^2
    # This is non-convex. We can use a convex relaxation or just use LP to find
    # a feasible configuration with large radii.
    
    # Alternative: Use scipy.optimize.linprog to solve a linearized version
    # For each pair (i,j), we can add constraint:
    #   r_i + r_j <= x_i - x_j + M*(1-u_ij)  ... this gets complex.
    
    # Instead, let's do a local search with random restarts and perturbations.
    
    # Phase 3: Random restarts with Powell
    np.random.seed(42)
    num_restarts = 8
    
    for restart in range(num_restarts):
        if time.time() > deadline - 15:
            break
        
        # Generate random initial configuration
        # Use a jittered grid as starting point
        x_rand = (np.arange(side) + 0.5 + np.random.randn(side) * 0.1) / side
        y_rand = (np.arange(side) + 0.5 + np.random.randn(side) * 0.1) / side
        
        params_rand = np.zeros(2 * N)
        idx = 0
        for i in range(side):
            for j in range(side):
                if idx >= N:
                    break
                params_rand[2*idx] = np.clip(x_rand[i], 0.05, 0.95)
                params_rand[2*idx+1] = np.clip(y_rand[j], 0.05, 0.95)
                idx += 1
        if idx < N:
            for k in range(idx, N):
                params_rand[2*k] = 0.5
                params_rand[2*k+1] = 0.5
        
        try:
            res = minimize(
                objective,
                params_rand,
                method='Powell',
                bounds=bounds,
                options={'maxiter': 2000, 'xtol': 1e-10, 'ftol': 1e-10}
            )
            if res.fun < best_obj:
                best_params = res.x
                best_obj = res.fun
        except Exception:
            pass

    # Phase 4: Local refinement with small perturbations
    if time.time() < deadline - 10:
        # Try to improve by perturbing each circle slightly
        x_best = best_params[0::2].copy()
        y_best = best_params[1::2].copy()
        radii_best = compute_radii(x_best, y_best)
        
        # Try moving each circle in small steps
        for _ in range(50):
            if time.time() > deadline - 5:
                break
            improved = False
            for i in range(N):
                for dx_try in [-0.01, -0.005, 0, 0.005, 0.01]:
                    for dy_try in [-0.01, -0.005, 0, 0.005, 0.01]:
                        x_try = x_best.copy()
                        y_try = y_best.copy()
                        x_try[i] += dx_try
                        y_try[i] += dy_try
                        
                        # Check bounds
                        if x_try[i] < 0 or x_try[i] > 1 or y_try[i] < 0 or y_try[i] > 1:
                            continue
                        
                        radii_try = compute_radii(x_try, y_try)
                        sum_try = np.sum(radii_try)
                        sum_best = np.sum(radii_best)
                        
                        if sum_try > sum_best + 1e-12:
                            x_best = x_try
                            y_best = y_try
                            radii_best = radii_try
                            improved = True
                            break
                    if improved:
                        break
            if not improved:
                break
        
        # Update best_params
        best_params = np.zeros(2 * N)
        best_params[0::2] = x_best
        best_params[1::2] = y_best
        best_obj = objective(best_params)

    # Final construction
    x = best_params[0::2]
    y = best_params[1::2]
    radii = compute_radii(x, y)
    
    # Shrink radii slightly to ensure strict disjointness (floating point safety)
    radii *= 0.99999
    
    result = np.column_stack((x, y, radii))
    return result

# EVOLVE-BLOCK-END
