"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for BBOB-style Schwefel f20 on [-5, 5]^5.
    
    Strategy:
      - Diverse initial points to find a good basin quickly.
      - (1+1)-ES with 1/5 success rule for fast, high-frequency updates (maximizes AOCC).
      - Stagnation detection with random restarts to escape local minima.
      - Final polish pass with scipy L-BFGS-B to refine the best solution.
      - Deadline guard to ensure we return within 330s wall-clock.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin
        
        rng = np.random.default_rng(42)
        
        # Initial points: origin, random, and structured points for diversity
        init_points = [np.zeros(dim)]
        for _ in range(4):
            init_points.append(rng.uniform(-5, 5, dim))
        # Add points at boundaries
        init_points.append(np.full(dim, 4.0))
        init_points.append(np.full(dim, -4.0))
        
        # Evaluate initial points
        best_x = None
        best_f = np.inf
        for x0 in init_points:
            if time.time() > deadline or budget <= 0:
                break
            x0 = np.clip(x0, -5, 5)
            fx = func(x0)
            budget -= 1
            if fx < best_f:
                best_f = fx
                best_x = x0.copy()
        
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            budget -= 1
        
        # (1+1)-ES with stagnation detection
        sigma = 2.0
        sigma_min = 0.005
        sigma_max = 5.0
        
        stagnation_count = 0
        max_stagnation = 30  # restart after 30 consecutive failures
        
        while budget > 0 and time.time() < deadline:
            # Generate offspring
            y = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
            fy = func(y)
            budget -= 1
            
            if fy < best_f:
                best_x = y
                best_f = fy
                sigma *= 1.3  # Moderate increase on success
                stagnation_count = 0
            else:
                sigma *= 0.9  # Gentle decrease on failure
                stagnation_count += 1
            
            sigma = np.clip(sigma, sigma_min, sigma_max)
            
            # Stagnation detection: if stuck for too long, try a random restart
            if stagnation_count >= max_stagnation:
                # Try a few random restarts to escape the local minimum
                restart_success = False
                for _ in range(min(3, budget)):
                    if time.time() > deadline:
                        break
                    x_restart = np.clip(rng.uniform(-5, 5, dim), -5, 5)
                    fr = func(x_restart)
                    budget -= 1
                    if fr < best_f:
                        best_x = x_restart
                        best_f = fr
                        restart_success = True
                        break
                
                if restart_success:
                    sigma = 2.0  # Reset sigma for new basin
                    stagnation_count = 0
                    max_stagnation = 20  # Be more aggressive with restarts
                else:
                    # No better point found; keep going but allow more stagnation
                    max_stagnation += 10
                    stagnation_count = 0
            
            # Periodic coarse-to-fine refinement around current best
            # This provides a structural guarantee of local improvement
            if stagnation_count % 15 == 14 and budget > 5:
                # Do a small grid search around the current best
                grid_size = 3
                step = sigma * 0.5
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        if dim >= 2:
                            for k in range(-1, 2):
                                if dim >= 3:
                                    for l in range(-1, 2):
                                        if dim >= 4:
                                            for m in range(-1, 2):
                                                if dim >= 5:
                                                    if budget <= 0 or time.time() > deadline:
                                                        break
                                                    x_grid = best_x.copy()
                                                    x_grid[0] += i * step
                                                    x_grid[1] += j * step
                                                    x_grid[2] += k * step
                                                    x_grid[3] += l * step
                                                    x_grid[4] += m * step
                                                    x_grid = np.clip(x_grid, -5, 5)
                                                    fg = func(x_grid)
                                                    budget -= 1
                                                    if fg < best_f:
                                                        best_f = fg
                                                        best_x = x_grid.copy()
                                                        sigma = max(sigma * 0.7, sigma_min)
                                                        stagnation_count = 0
                                                else:
                                                    break
                                        else:
                                            break
                                else:
                                    break
                        else:
                            break
                # Reset stagnation if we found something
                if stagnation_count > 0:
                    stagnation_count = 0
                    max_stagnation = 25
        
        # Final polish pass with scipy L-BFGS-B
        # Only if we have enough budget and time left
        if budget > 10 and time.time() < deadline - 5:
            try:
                calls_used_in_scipy = [0]
                
                def wrapped_func(x):
                    nonlocal budget
                    if budget <= 0 or time.time() > deadline:
                        return best_f
                    x = np.clip(x, -5, 5)
                    fx = func(x)
                    budget -= 1
                    calls_used_in_scipy[0] += 1
                    return fx

                # More aggressive: use more iterations and tighter tolerances
                maxit = min(100, max(5, budget))
                
                result = minimize(
                    wrapped_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': maxit, 'ftol': 1e-15, 'gtol': 1e-10, 'maxfun': budget}
                )

                if result.fun < best_f:
                    best_x = np.clip(result.x, -5, 5)
                    best_f = result.fun

            except Exception:
                pass
        
        return best_x, best_f
# EVOLVE-BLOCK-END
