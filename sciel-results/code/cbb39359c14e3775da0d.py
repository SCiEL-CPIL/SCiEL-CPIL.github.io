"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-start optimizer with L-BFGS-B refinement."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # 5s safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        call_count = [0]

        def counted_func(x):
            if time.time() > deadline:
                raise TimeoutError()
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            val = func(x)
            if not np.isfinite(val):
                return 1e10
            return val

        best_x = None
        best_f = np.inf

        # Phase 1: Sobol sampling for diverse initial points
        n_sobol = min(64, budget // 5)
        if n_sobol < 8:
            n_sobol = 8

        # Generate Sobol points in [0,1]^dim, map to [-5,5]^dim
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        sobol_pts = sampler.random(n_sobol)
        sobol_pts = sobol_pts * 10.0 - 5.0  # map to [-5, 5]

        # Add center and a few random points
        extra_pts = [np.zeros(dim)]
        for _ in range(min(4, max(0, budget - n_sobol - 10))):
            extra_pts.append(rng.uniform(-4, 4, dim))

        all_starts = np.vstack([sobol_pts, np.array(extra_pts)])

        # Evaluate all starts
        start_values = []
        for pt in all_starts:
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break
            try:
                fv = counted_func(pt)
            except Exception:
                fv = 1e10
            start_values.append((fv, pt.copy()))

        if not start_values:
            # Fallback: just evaluate center
            pt = np.zeros(dim)
            try:
                fv = counted_func(pt)
            except Exception:
                fv = 1e10
            start_values.append((fv, pt.copy()))

        # Sort by function value
        start_values.sort(key=lambda t: t[0])

        # Phase 2: Apply L-BFGS-B to the top-k starts
        k = min(15, len(start_values))
        if budget - call_count[0] < 10:
            k = 1

        for i in range(k):
            if call_count[0] >= budget:
                break
            if time.time() > deadline:
                break

            f0, x0 = start_values[i]

            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

            # Skip L-BFGS-B if we've already spent too much budget
            remaining = budget - call_count[0]
            if remaining < 20:
                break

            try:
                res = minimize(
                    counted_func,
                    x0,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': max(1, remaining // 2), 'maxls': 20}
                )
                if res.success and np.isfinite(res.fun):
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = np.clip(res.x, -5, 5).copy()
            except Exception:
                pass

        # Phase 3: If we have budget left, do a few more local refinements
        remaining = budget - call_count[0]
        if remaining > 30 and best_x is not None:
            # Try a few more random perturbations around best_x
            for _ in range(min(remaining // 3, 10)):
                if call_count[0] >= budget:
                    break
                if time.time() > deadline:
                    break
                perturbation = rng.standard_normal(dim) * 0.5
                x_try = np.clip(best_x + perturbation, -5, 5)
                try:
                    fv = counted_func(x_try)
                except Exception:
                    fv = 1e10
                if fv < best_f:
                    best_f = fv
                    best_x = x_try.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            if call_count[0] < budget:
                try:
                    best_f = counted_func(best_x)
                except Exception:
                    best_f = 1e10

        return best_x, best_f
# EVOLVE-BLOCK-END
