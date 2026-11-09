"""Multi-restart SciPy optimizer for BBOB f9 (Rosenbrock rotated)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart optimizer for minimizing func on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            call_count += 1
            return val

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Generate diverse starting points tailored for Rosenbrock valley
        rng = np.random.default_rng(42)
        starts = []
        
        # Phase 1: Structured points along the Rosenbrock valley direction
        # The Rosenbrock valley is roughly along the line x1=x2=...=xn
        for scale in np.arange(-4.0, 4.5, 0.5):
            starts.append(np.full(dim, scale))
        
        # Phase 2: Points with slight offsets from the valley
        for scale in np.arange(-3.0, 3.5, 1.0):
            for offset in [0.5, -0.5, 1.0, -1.0]:
                pt = np.full(dim, scale)
                # Offset one coordinate to explore off-valley
                pt[0] += offset
                starts.append(pt.copy())
        
        # Phase 3: Random points with bias toward center (where minimum likely is)
        for _ in range(20):
            pt = rng.uniform(-4, 4, dim)
            # Slightly pull toward center to bias toward likely minimum region
            pt = 0.7 * pt + 0.3 * np.zeros(dim)
            starts.append(pt)
        
        # Phase 4: Axis-aligned exploration
        for i in range(dim):
            for val in [0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -1.5, -2.0, -3.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)
        
        # Phase 5: Points near boundaries
        for i in range(dim):
            for val in [3.5, 4.0, -3.5, -4.0]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)
        
        # Deduplicate and limit to reasonable number
        unique_starts = []
        seen = set()
        for s in starts:
            key = tuple(np.round(s, 2))
            if key not in seen:
                seen.add(key)
                unique_starts.append(s)
        
        starts = unique_starts[:40]  # Limit to 40 starts to leave budget for optimization
        
        # Shuffle starts
        perm = rng.permutation(len(starts))
        starts = [starts[i] for i in perm]

        # Adaptive restart strategy with tighter budget management
        # Reserve 30% of budget for final polish
        polish_budget = int(budget * 0.3)
        search_budget = budget - polish_budget
        
        base_maxfev = 25
        min_maxfev = 8
        max_maxfev = 60
        current_maxfev = base_maxfev

        for x0 in starts:
            if time.time() > deadline or call_count >= search_budget:
                break
            
            # Skip if we've already found a very good solution and this start is far
            if best_f < 0.5 and best_x is not None:
                dist = np.linalg.norm(x0 - best_x)
                if dist > 3.5:
                    continue

            # Adaptive maxfev: give more budget if we're making progress
            try:
                res = minimize(
                    safe_func, 
                    x0, 
                    method='L-BFGS-B', 
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 100, 'maxfev': current_maxfev, 'gtol': 1e-6}
                )
                if res.fun < best_f:
                    # Made progress, allow more budget for next restart
                    current_maxfev = min(max_maxfev, int(current_maxfev * 1.2))
                    update_best(res.x, res.fun)
                else:
                    # No improvement, reduce budget for next restart
                    current_maxfev = max(min_maxfev, int(current_maxfev * 0.6))
            except Exception:
                pass

        # Final high-precision polish from the best point with remaining budget
        if best_x is not None and time.time() < deadline and call_count < budget:
            remaining = budget - call_count
            if remaining > 10:
                # First polish: L-BFGS-B with tight tolerances
                try:
                    res = minimize(
                        safe_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 300, 'maxfev': remaining, 'gtol': 1e-10, 'ftol': 1e-14}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

            # Second polish: even tighter if budget remains
            if best_x is not None and time.time() < deadline and call_count < budget:
                remaining = budget - call_count
                if remaining > 5:
                    try:
                        res = minimize(
                            safe_func, 
                            best_x, 
                            method='L-BFGS-B', 
                            bounds=[(-5, 5)] * dim,
                            options={'maxiter': 500, 'maxfev': remaining, 'gtol': 1e-12, 'ftol': 1e-16}
                        )
                        if res.fun < best_f:
                            update_best(res.x, res.fun)
                    except Exception:
                        pass

        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            if call_count < budget:
                best_f = func(best_x)
                call_count += 1
            else:
                best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
