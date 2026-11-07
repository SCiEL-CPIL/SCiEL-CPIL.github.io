"""
Portfolio optimizer for BBOB f3 (Rastrigin) with multi-scale exploration and local refinement.
Strategy:
1. Coarse random exploration to find promising basins.
2. Multi-scale local refinement:
   - Large step size to escape local minima (Rastrigin has many local minima at integers).
   - Medium step size for basin hopping.
   - Small step size for fine polishing.
3. Use Nelder-Mead for final high-precision polish.
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 50s safety margin before 330s kill

        best_x = None
        best_f = float('inf')
        calls_made = 0
        max_calls = self.budget

        def check_deadline():
            return (time.time() - start_time) > deadline or calls_made >= max_calls

        def update_best(x, f):
            nonlocal best_x, best_f, calls_made
            calls_made += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        rng = np.random.default_rng(42)

        # Phase 1: Coarse exploration - evaluate many random points
        # Rastrigin has local minima at integer points, so we sample broadly
        n_explore = 30
        candidates = [np.zeros(self.dim)]
        # Add structured points: center of each axis
        for i in range(self.dim):
            for val in [-4, -2, 0, 2, 4]:
                pt = np.zeros(self.dim)
                pt[i] = val
                candidates.append(pt)
        # Add random points
        for _ in range(n_explore):
            candidates.append(rng.uniform(-4.5, 4.5, self.dim))

        for x in candidates:
            if check_deadline():
                break
            x_clipped = np.clip(x, -5, 5)
            f = func(x_clipped)
            update_best(x_clipped, f)

        if best_x is None:
            x = np.zeros(self.dim)
            f = func(x)
            update_best(x, f)
            return best_x, best_f

        # Phase 2: Multi-scale local search
        # For each of several scales, do a local search from the current best
        scales = [2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]

        remaining = max_calls - calls_made
        # Allocate budget proportionally: more budget for larger scales initially
        # But also ensure we have budget for final polish
        # Let's give each scale a fraction of remaining budget
        n_scales = len(scales)
        # Reserve some budget for final Nelder-Mead polish
        nm_budget = min(100, remaining // 5)
        es_budget = remaining - nm_budget
        per_scale = max(1, es_budget // n_scales)

        for scale in scales:
            if check_deadline():
                break

            # Start from current best
            x = best_x.copy()
            f = best_f
            sigma = scale

            # (1+1)-ES with adaptive sigma
            success_count = 0
            window = []
            window_size = 15

            for _ in range(per_scale):
                if check_deadline():
                    break

                step = sigma * rng.standard_normal(self.dim)
                y = np.clip(x + step, -5, 5)
                fy = func(y)
                calls_made += 1

                if fy < f:
                    x = y
                    f = fy
                    success_count += 1
                    window.append(1)
                    # Update global best
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                else:
                    window.append(0)

                # Adapt sigma based on success rate in last window_size steps
                if len(window) >= window_size:
                    success_rate = sum(window[-window_size:]) / window_size
                    if success_rate > 0.3:
                        sigma *= 1.1
                    elif success_rate < 0.1:
                        sigma *= 0.8
                    window = window[-window_size:]

        # Phase 3: Final high-precision polish with Nelder-Mead
        if not check_deadline():
            nm_calls = max_calls - calls_made
            if nm_calls > 10:
                # Use scipy's minimize with Nelder-Mead
                try:
                    from scipy.optimize import minimize

                    def wrapper(x):
                        nonlocal calls_made
                        if calls_made >= max_calls:
                            return best_f
                        calls_made += 1
                        return func(np.clip(x, -5, 5))

                    # Run Nelder-Mead from current best
                    result = minimize(
                        wrapper,
                        best_x,
                        method='Nelder-Mead',
                        options={
                            'maxiter': nm_calls,
                            'xatol': 1e-8,
                            'fatol': 1e-10
                        }
                    )

                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f

# EVOLVE-BLOCK-END
