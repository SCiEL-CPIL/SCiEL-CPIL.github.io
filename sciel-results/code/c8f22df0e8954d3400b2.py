import numpy as np
from scipy.optimize import minimize
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0
        n = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)
        best_x = None
        best_fx = np.inf
        eval_count = 0

        def wrapped(x):
            nonlocal eval_count, best_x, best_fx
            if eval_count >= budget:
                return 1e10
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            eval_count += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x.copy()
            return fx

        def time_ok(margin=5.0):
            return time.time() < deadline - margin

        # Phase 1: Diverse initial points
        starts = []
        # Center point
        starts.append(np.zeros(n))
        # Random points in [-4, 4]
        for _ in range(10):
            starts.append(rng.uniform(-4, 4, n))
        # Axis-aligned points
        for d in range(n):
            for val in [-3.0, -1.5, 1.5, 3.0]:
                pt = np.zeros(n)
                pt[d] = val
                starts.append(pt)
        # More random points for diversity
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, n))

        # Sort by norm to prioritize center
        starts.sort(key=lambda x: np.linalg.norm(x))

        # Phase 2: Evaluate all starting points to establish baseline
        for x0 in starts:
            if not time_ok(10.0):
                break
            if eval_count >= budget:
                break
            wrapped(x0)

        if best_x is None:
            best_x = np.zeros(n)
            if eval_count < budget:
                wrapped(best_x)
            else:
                best_fx = 1e10

        # Phase 3: L-BFGS-B optimization from best and top starting points
        # First: polish the best point found
        if time_ok(10.0) and eval_count < budget:
            remaining = budget - eval_count
            maxiter = max(1, min(remaining, 40))
            try:
                res = minimize(
                    wrapped, best_x, method='L-BFGS-B',
                    bounds=[(-5, 5)] * n,
                    options={'maxiter': maxiter, 'maxls': 20, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_fx:
                    best_fx = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        # Second: try other starting points, prioritizing those close to best
        if time_ok(10.0) and eval_count < budget:
            # Generate new candidates near the best point
            local_starts = []
            for scale in [0.5, 1.0, 2.0]:
                offset = rng.normal(0, scale, n)
                local_starts.append(best_x + offset)

            # Also try some of the original starts that weren't the best
            for x0 in starts:
                if np.linalg.norm(x0 - best_x) < 3.0:
                    local_starts.append(x0)

            # Sort by distance to best
            local_starts.sort(key=lambda x: np.linalg.norm(x - best_x))

            for x0 in local_starts[:8]:
                if not time_ok(10.0):
                    break
                if eval_count >= budget:
                    break

                x0_clipped = np.clip(x0, -5, 5)
                # Quick evaluation
                if eval_count < budget:
                    wrapped(x0_clipped)

                # Short optimization from this point
                if time_ok(10.0) and eval_count < budget:
                    remaining = budget - eval_count
                    maxiter = max(1, min(remaining, 25))
                    try:
                        res = minimize(
                            wrapped, x0_clipped, method='L-BFGS-B',
                            bounds=[(-5, 5)] * n,
                            options={'maxiter': maxiter, 'maxls': 15, 'ftol': 1e-10, 'gtol': 1e-7}
                        )
                        if res.fun < best_fx:
                            best_fx = res.fun
                            best_x = np.clip(res.x, -5, 5)
                    except Exception:
                        pass

        # Phase 4: Final polish if budget remains
        if time_ok(5.0) and eval_count < budget and best_x is not None:
            remaining = budget - eval_count
            if remaining >= 5:
                try:
                    res = minimize(
                        wrapped, best_x, method='L-BFGS-B',
                        bounds=[(-5, 5)] * n,
                        options={'maxiter': remaining, 'maxls': 20, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    if res.fun < best_fx:
                        best_fx = res.fun
                        best_x = np.clip(res.x, -5, 5)
                except Exception:
                    pass

        if best_x is None:
            best_x = np.zeros(n)
            best_fx = func(best_x)

        return best_x, best_fx
