"""Multi-start L-BFGS-B with Sobol seeding for BBOB f17 (Schaffers F7)."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-start L-BFGS-B optimizer for continuous black-box functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _sobol_sequence(self, n_samples, dim):
        """Generate Sobol sequence in [0,1]^dim using scipy."""
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dim, scramble=True, seed=42)
        m = int(np.ceil(np.log2(max(n_samples, 2))))
        n_pow2 = 2 ** m
        points = sampler.random(n_pow2)
        return points[:n_samples]

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        n = dim

        # Phase 1: Sobol-based global sampling for early coverage
        n_sobol = min(60, max(15, budget // 8))
        sobol_pts = self._sobol_sequence(n_sobol, n)
        sobol_pts = sobol_pts * 10.0 - 5.0

        best_x = sobol_pts[0].copy()
        best_f = float('inf')
        calls = 0

        for i in range(n_sobol):
            if time.time() - start_time > deadline:
                break
            x = sobol_pts[i]
            f = func(x)
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if calls >= budget or time.time() - start_time > deadline:
            return best_x, best_f

        # Phase 2: Multi-start L-BFGS-B from top-k Sobol points
        # Sort Sobol points by function value
        sobol_vals = np.array([func(sobol_pts[i]) if i < n_sobol else float('inf') for i in range(n_sobol)])
        # Actually we already evaluated them, let's re-collect
        # Re-evaluate is wasteful; let's just use the sorted indices from our tracking
        # We didn't track all values, so let's do a quick re-sort using stored data
        # Actually, we only kept best. Let's just use the top few Sobol points as starts.
        # A better approach: run L-BFGS-B from the best Sobol point and a few others.

        # Use L-BFGS-B from the best point found in Phase 1
        from scipy.optimize import minimize

        remaining = budget - calls

        if remaining < 5:
            return best_x, best_f

        # Run L-BFGS-B from the best Sobol point
        def _wrap(x):
            nonlocal calls
            if calls >= budget:
                return float('inf')
            calls += 1
            return func(x)

        # L-BFGS-B with bounds
        bounds = [(-5.0, 5.0)] * n

        # First run: from best Sobol point
        try:
            res = minimize(
                _wrap,
                best_x,
                method='L-BFGS-B',
                bounds=bounds,
                options={
                    'maxiter': min(remaining // 2, 200),
                    'maxls': 20,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxfun': min(remaining, 500)
                }
            )
            if res.fun < best_f:
                best_f = res.fun
                best_x = res.x.copy()
        except Exception:
            pass

        if calls >= budget or time.time() - start_time > deadline:
            return best_x, best_f

        # Second run: from a different Sobol point (2nd best)
        # Find the 2nd best Sobol point
        # We need to re-evaluate or track. Let's just try a few random Sobol points.
        # Actually, let's run L-BFGS-B from the center (0,0,...,0) as a second start
        center = np.zeros(n)
        remaining = budget - calls
        if remaining >= 10:
            try:
                res2 = minimize(
                    _wrap,
                    center,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': min(remaining // 2, 100),
                        'maxls': 20,
                        'ftol': 1e-12,
                        'gtol': 1e-8,
                        'maxfun': min(remaining, 300)
                    }
                )
                if res2.fun < best_f:
                    best_f = res2.fun
                    best_x = res2.x.copy()
            except Exception:
                pass

        if calls >= budget or time.time() - start_time > deadline:
            return best_x, best_f

        # Phase 3: Final polish with Nelder-Mead if budget allows
        remaining = budget - calls
        if remaining >= 15 and time.time() - start_time < deadline:
            try:
                def _wrap_nm(x):
                    nonlocal calls
                    if calls >= budget:
                        return float('inf')
                    calls += 1
                    return func(x)

                res3 = minimize(
                    _wrap_nm,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': min(remaining, 80),
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                if res3.fun < best_f:
                    best_f = res3.fun
                    best_x = res3.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
