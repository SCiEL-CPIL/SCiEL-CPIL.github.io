"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid (1+1)-ES with adaptive sigma, LHS exploration, and Powell refinement.
    Uses (1+1)-ES as the primary search mechanism for its robustness on multimodal
    landscapes, combined with Powell for high-precision local refinement.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 330s. We stop at 310s to be safe.
        deadline = start_time + 310.0
        budget = self.budget
        dim = self.dim

        def check_time():
            return time.time() < deadline

        calls_made = 0
        best_x = None
        best_f = float('inf')

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return None
            if time.time() > deadline - 5:
                return None
            try:
                val = float(func(x))
            except Exception:
                return None
            calls_made += 1
            if val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        rng = np.random.default_rng(42)

        # Phase 1: LHS global sampling for diverse starting points
        # Use a significant portion of the budget for exploration to find good basins.
        n_starts = max(10, min(50, budget // 4))
        
        # Latin Hypercube Sampling
        for i in range(n_starts):
            if not check_time() or calls_made >= budget:
                break
            x = np.empty(dim)
            for d in range(dim):
                # Ensure stratification: each dimension gets a unique stratum per point index
                perm = rng.permutation(n_starts)
                x[d] = -5.0 + 10.0 * (perm[i] + rng.random()) / n_starts
            x = np.clip(x, -5, 5)
            fx = safe_call(x)
            if fx is None:
                break

        if best_x is None:
            # Fallback
            x = rng.uniform(-5, 5, dim)
            fx = safe_call(x)
            if fx is not None:
                best_f = fx
                best_x = x.copy()

        # Phase 2: (1+1)-ES refinement with adaptive restarts
        if best_x is not None and check_time() and calls_made < budget:
            x = best_x.copy()
            fx = best_f
            
            # Initial sigma: scale relative to box size
            sigma = 1.0 
            
            # Adaptive parameters
            success_count = 0
            total_steps = 0
            window = 10
            no_improve_count = 0
            max_no_improve = 200  # Restart if no improvement for 200 steps
            
            while check_time() and calls_made < budget:
                # Generate offspring
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_call(y)
                if fy is None:
                    break

                total_steps += 1
                
                # Update best
                if fy < fx:
                    x, fx = y, fy
                    success_count += 1
                    no_improve_count = 0
                    # Increase sigma slightly on success
                    sigma *= 1.1
                else:
                    # Decrease sigma on failure
                    sigma *= 0.95
                    no_improve_count += 1

                # Adaptive sigma adjustment based on success rate over a window
                if total_steps >= window:
                    success_rate = success_count / total_steps
                    # 1/5 success rule: aim for 20% success rate
                    if success_rate > 0.25:
                        sigma *= 1.2
                    elif success_rate < 0.15:
                        sigma *= 0.8
                    success_count = 0
                    total_steps = 0

                # Clamp sigma
                sigma = max(sigma, 1e-6)
                sigma = min(sigma, 5.0) # Box is [-5,5], so max useful step is ~5
                
                # Stagnation handling: if no improvement for a long time, restart locally
                if no_improve_count > max_no_improve:
                    # Perform a local random jump to escape local minimum
                    # Jump size proportional to current sigma but larger
                    jump_size = max(sigma * 2.0, 0.5)
                    x = np.clip(x + jump_size * rng.standard_normal(dim), -5, 5)
                    fx = safe_call(x)
                    if fx is None:
                        break
                    sigma = 1.0 # Reset sigma
                    no_improve_count = 0
                    total_steps = 0
                    success_count = 0

        # Phase 3: Powell refinement for high-precision convergence
        # Use remaining budget for Powell to polish the solution
        if best_x is not None and check_time() and calls_made < budget:
            from scipy.optimize import minimize
            
            # Allocate remaining budget to Powell
            remaining_budget = budget - calls_made
            if remaining_budget > 10:
                powell_budget = int(remaining_budget * 0.5)  # Use half of remaining for Powell
                
                def wrapper(x):
                    nonlocal calls_made
                    if calls_made >= budget:
                        return np.inf
                    if time.time() > deadline - 5:
                        return np.inf
                    try:
                        x_clipped = np.clip(x, -5, 5)
                        val = float(func(x_clipped))
                        calls_made += 1
                        return val
                    except Exception:
                        return np.inf

                try:
                    result = minimize(
                        wrapper,
                        best_x,
                        method='Powell',
                        options={'maxiter': 1000, 'maxfev': powell_budget, 'xatol': 1e-8, 'fatol': 1e-10}
                    )
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
