"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Coordinate descent with adaptive per-dimension step sizes for sharp ridge."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290  # safety margin before 300s kill

        def safe_call(x):
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5.0, 5.0)
            try:
                return float(func(x))
            except Exception:
                return np.inf

        # Global bookkeeping
        best_x = None
        best_f = np.inf
        calls_made = 0

        rng = np.random.default_rng(42)

        # Initial coarse sampling to find a good starting point
        n_init = min(15, budget // 10)
        candidates = []
        for _ in range(n_init):
            x = rng.uniform(-4, 4, dim)
            fx = safe_call(x)
            calls_made += 1
            candidates.append((fx, x.copy()))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        candidates.sort(key=lambda t: t[0])
        x0 = candidates[0][1].copy()
        f0 = candidates[0][0]

        remaining = budget - calls_made
        if remaining <= 0:
            return best_x, best_f

        # Coordinate descent with adaptive step sizes per dimension
        x = x0.copy()
        fx = f0
        # Initialize step sizes: larger for dimensions where we expect more variation
        steps = np.ones(dim) * (8.0 / dim)  # total range / dim
        min_step = 1e-6
        max_step = 10.0

        # Track recent successes per dimension for adaptive stepping
        success_rate = np.ones(dim) / (dim + 1)  # Beta prior
        total_attempts = np.ones(dim)

        # Main coordinate descent loop
        while calls_made < budget and time.time() < deadline:
            # Iterate through dimensions in random order to avoid bias
            dim_order = rng.permutation(dim)
            
            for d in dim_order:
                if calls_made >= budget or time.time() > deadline:
                    break
                
                # Try positive and negative steps along dimension d
                found_improvement = False
                for sign in [1, -1]:
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    y = x.copy()
                    y[d] += sign * steps[d]
                    y = np.clip(y, -5.0, 5.0)
                    fy = safe_call(y)
                    calls_made += 1
                    
                    if fy < fx - 1e-14:
                        # Improvement found
                        x = y
                        fx = fy
                        found_improvement = True
                        # Increase step size for this dimension (exploration)
                        steps[d] = min(steps[d] * 1.5, max_step)
                        # Update success rate
                        success_rate[d] = (success_rate[d] * total_attempts[d] + 1) / (total_attempts[d] + 1)
                        total_attempts[d] += 1
                        
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        # No improvement: decrease step size
                        steps[d] = max(steps[d] * 0.5, min_step)
                        success_rate[d] = (success_rate[d] * total_attempts[d]) / (total_attempts[d] + 1)
                        total_attempts[d] += 1
                
                # If no improvement in either direction, try a smaller step
                if not found_improvement:
                    if calls_made >= budget or time.time() > deadline:
                        break
                    
                    # Try half the current step
                    y = x.copy()
                    y[d] += 0.5 * steps[d]
                    y = np.clip(y, -5.0, 5.0)
                    fy = safe_call(y)
                    calls_made += 1
                    
                    if fy < fx - 1e-14:
                        x = y
                        fx = fy
                        if fx < best_f:
                            best_f = fx
                            best_x = x.copy()
                    else:
                        y = x.copy()
                        y[d] -= 0.5 * steps[d]
                        y = np.clip(y, -5.0, 5.0)
                        fy = safe_call(y)
                        calls_made += 1
                        
                        if fy < fx - 1e-14:
                            x = y
                            fx = fy
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
