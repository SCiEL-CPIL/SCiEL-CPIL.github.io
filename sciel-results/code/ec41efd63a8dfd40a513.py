"""Sobol-seed multi-start Nelder-Mead for BBOB f23 (Katsuura)."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Sobol-based multi-start local optimization for Katsuura."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 285.0

        calls = 0

        def try_eval(x):
            nonlocal calls
            if calls >= budget or time.time() > deadline:
                return None
            x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
            f = float(func(x))
            calls += 1
            return f

        # Phase 1: Deterministic Sobol sampling for global coverage
        # Use scipy's Sobol quasi-random sequence for low-discrepancy coverage
        from scipy.stats import qmc

        # Generate Sobol points; use enough to cover the space well
        n_sobol = min(128, budget)
        sobol = qmc.Sobol(d=dim, scramble=True, seed=42)
        # Number of points must be a power of 2 for Sobol
        n_sobol = 1 << (n_sobol - 1).bit_length()
        n_sobol = min(n_sobol, budget)
        # Ensure at least 32 points
        n_sobol = max(32, n_sobol)

        sobol_points = sobol.random(n_sobol)  # in [0,1]^d
        # Map to [-5, 5]^d
        sobol_points = sobol_points * 10.0 - 5.0

        best_f = np.inf
        best_x = None
        candidates = []

        for i in range(n_sobol):
            if time.time() > deadline or calls >= budget:
                break
            x = sobol_points[i]
            f = try_eval(x)
            if f is None:
                break
            candidates.append((f, x.copy()))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: Multi-start Nelder-Mead from top-k diverse points
        # Select top points that are sufficiently diverse
        top_k = min(12, len(candidates))
        diverse_starts = []
        for f, x in candidates:
            if len(diverse_starts) >= top_k:
                break
            # Check diversity: minimum distance from already selected starts
            diverse = True
            for _, dx in diverse_starts:
                if np.linalg.norm(x - dx) < 1.2:
                    diverse = False
                    break
            if diverse:
                diverse_starts.append((f, x))

        # If not enough diverse points, add more from the list
        if len(diverse_starts) < min(6, top_k):
            for f, x in candidates:
                if len(diverse_starts) >= min(6, top_k):
                    break
                if (f, x) not in diverse_starts:
                    diverse_starts.append((f, x))

        # Nelder-Mead settings - tighter convergence for Katsuura
        nm_options = {
            'maxiter': 300,
            'xatol': 1e-8,
            'fatol': 1e-10,
            'adaptive': True,
        }

        # Run Nelder-Mead from each diverse start
        for f_init, x_init in diverse_starts:
            if time.time() > deadline or calls >= budget:
                break

            # Wrap func with budget/time guard
            def nm_func(x):
                if calls >= budget or time.time() > deadline:
                    return 1e10
                x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
                f = float(func(x))
                calls += 1
                return f

            try:
                res = minimize(nm_func, x_init, method='Nelder-Mead', options=nm_options)
                if res.success or res.fun < best_f:
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
            except Exception:
                pass

        # Phase 3: If budget remains, do a focused refinement around the best point
        if calls < budget and time.time() < deadline and best_x is not None:
            # Use a local pattern search / coordinate descent refinement
            rng = np.random.default_rng(12345)
            sigma_refine = 0.05
            for _ in range(min(100, budget - calls)):
                if time.time() > deadline or calls >= budget:
                    break
                x_try = best_x + sigma_refine * rng.standard_normal(dim)
                x_try = np.clip(x_try, -5, 5)
                f_try = try_eval(x_try)
                if f_try is None:
                    break
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    sigma_refine *= 0.85
                else:
                    sigma_refine *= 1.05
                    sigma_refine = min(sigma_refine, 0.5)

        if best_x is None:
            best_x = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
