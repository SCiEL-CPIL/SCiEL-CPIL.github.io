"""Portfolio search optimizer for BBOB f20 (Schwefel) with periodicity-aware initialization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start (1+1)-ES with periodicity-aware initialization and L-BFGS-B polish.
    Leverages Schwefel's 4π periodicity to place initial points near the global minimum.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0
        dim = self.dim
        total_budget = self.budget

        best_x = None
        best_f = float('inf')
        call_count = 0

        def try_func(x):
            nonlocal best_x, best_f, call_count
            if call_count >= total_budget:
                return np.inf
            if time.time() > deadline:
                return np.inf
            try:
                fx = func(x)
                call_count += 1
                if np.isfinite(fx) and fx < best_f:
                    best_f = fx
                    best_x = x.copy()
                return fx
            except Exception:
                return np.inf

        # Phase 1: Periodicity-aware initialization
        # Schwefel's global minimum is at x* = 4.2096 per dimension
        # With hidden shift/rotation, we place points near this region and diverse locations
        x_star = np.full(dim, 4.2096)
        
        # Structured starting points exploiting periodicity
        starts = []
        # Near the known global minimum
        starts.append(x_star.copy())
        starts.append(x_star - 4.0 * np.ones(dim))  # Previous period
        starts.append(np.zeros(dim))
        starts.append(np.ones(dim) * 2.0)
        starts.append(np.ones(dim) * -2.0)
        
        # Random diverse points (more for coverage)
        rng_init = np.random.default_rng(42)
        for _ in range(8):
            starts.append(rng_init.uniform(-5, 5, dim))

        # Evaluate all starting points
        candidates = []
        for x0 in starts:
            if call_count >= total_budget or time.time() > deadline:
                break
            x0_clipped = np.clip(x0, -5, 5)
            fx = try_func(x0_clipped)
            if np.isfinite(fx):
                candidates.append((fx, x0_clipped))

        if not candidates:
            # Fallback: at least one call
            if best_x is None:
                x = np.zeros(dim)
                fx = try_func(x)
                if np.isfinite(fx):
                    best_f = fx
                    best_x = x
            return best_x, best_f

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Multi-start (1+1)-ES on top candidates
        # Allocate budget: ~350 for ES, ~100 for L-BFGS-B, ~50 for final polish
        es_budget = min(350, total_budget - 150)
        
        top_k = min(4, len(candidates))
        es_per_start = max(30, es_budget // top_k)

        for i in range(top_k):
            if call_count >= total_budget - 200 or time.time() > deadline - 20:
                break

            x = candidates[i][1].copy()
            fx = candidates[i][0]

            # If this is not the best, evaluate it properly (it was already evaluated)
            # Run (1+1)-ES from this start
            rng = np.random.default_rng(100 + i * 7919)
            sigma = 1.0
            sigma_min = 1e-6
            sigma_max = 4.0
            no_improve = 0
            steps = 0
            successes = 0

            for _ in range(es_per_start):
                if call_count >= total_budget - 150:
                    break
                if time.time() > deadline - 20:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = try_func(y)
                if not np.isfinite(fy):
                    break

                steps += 1
                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    no_improve = 0
                    sigma = min(sigma * 1.2, sigma_max)
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.9, sigma_min)

                # Adaptive sigma based on success rate
                if steps >= 10:
                    success_rate = successes / steps
                    if success_rate > 0.4:
                        sigma = min(sigma * 1.1, sigma_max)
                    elif success_rate < 0.1:
                        sigma = max(sigma * 0.85, sigma_min)
                    if steps >= 50:
                        successes = 0
                        steps = 0

                # Escape from local minima
                if no_improve > 15:
                    sigma = min(sigma * 2.0, sigma_max)
                    no_improve = 0

        # Phase 3: L-BFGS-B polish on best found
        if best_x is not None and call_count < total_budget - 50 and time.time() < deadline - 10:
            bounds = [(-5.0, 5.0)] * dim
            
            def local_func(x):
                return try_func(x)

            try:
                res = minimize(local_func, best_x, method='L-BFGS-B', bounds=bounds,
                               options={'maxiter': 200, 'maxfun': 250, 'gtol': 1e-10})
                if np.isfinite(res.fun) and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass

        # Phase 4: Fine-grained ES polish with adaptive sigma
        if best_x is not None and call_count < total_budget - 2 and time.time() < deadline - 2:
            rng_polish = np.random.default_rng(999)
            x = best_x.copy()
            fx = best_f
            sigma = 0.05
            remaining = total_budget - call_count
            no_improve = 0
            for _ in range(min(remaining, 50)):
                if call_count >= total_budget:
                    break
                if time.time() > deadline - 1:
                    break
                y = np.clip(x + sigma * rng_polish.standard_normal(dim), -5, 5)
                fy = try_func(y)
                if not np.isfinite(fy):
                    break
                if fy < fx:
                    x, fx = y, fy
                    sigma = min(sigma * 1.05, 0.5)
                    no_improve = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    no_improve += 1
                    sigma = max(sigma * 0.85, 1e-6)
                    if no_improve > 20:
                        # Try a larger jump to escape
                        y2 = np.clip(x + sigma * 3.0 * rng_polish.standard_normal(dim), -5, 5)
                        fy2 = try_func(y2)
                        if np.isfinite(fy2) and fy2 < fx:
                            x, fx = y2, fy2
                            sigma = min(sigma * 1.5, 0.5)
                            no_improve = 0
                            if fx < best_f:
                                best_f = fx
                                best_x = x.copy()
                        else:
                            sigma = max(sigma * 0.7, 1e-6)

        # Fallback: ensure at least one call was made
        if best_x is None:
            x = np.zeros(dim)
            fx = try_func(x)
            if np.isfinite(fx):
                best_f = fx
                best_x = x

        return best_x, best_f
# EVOLVE-BLOCK-END
