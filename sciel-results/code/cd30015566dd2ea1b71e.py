"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution


class Optimizer:
    """Multi-start L-BFGS-B with Sobol sampling for BBOB f11 (Discus)."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        n = self.dim
        budget = self.budget

        # State
        best_x = np.array([0.0] * n)
        best_fx = np.inf
        calls_made = 0

        def safe_call(x):
            nonlocal calls_made, best_x, best_fx
            if calls_made >= budget or time.time() > deadline:
                return None
            x_clipped = np.clip(x, -5.0, 5.0)
            fx = func(x_clipped)
            calls_made += 1
            if fx < best_fx:
                best_fx = fx
                best_x = x_clipped.copy()
            return fx

        # Sobol sampling for initial points
        def sobol_sample(n, num_samples, seed=42):
            """Generate Sobol samples in [-5, 5]^n."""
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=n, seed=seed, scramble=True)
            samples = sampler.random(num_samples)
            return samples * 10.0 - 5.0  # Map [0,1] to [-5,5]

        # Phase 1: Global exploration with Sobol sampling
        num_global = min(50, budget // 10)
        if num_global > 0:
            sobol_points = sobol_sample(n, num_global)
            for pt in sobol_points:
                if time.time() > deadline or calls_made >= budget:
                    break
                safe_call(pt)

        # Phase 2: Multi-start L-BFGS-B from best points
        # Collect top candidates for local optimization
        # We'll do multiple L-BFGS-B runs from different starting points

        # Generate diverse starting points for L-BFGS-B
        rng = np.random.default_rng(123)
        num_local_starts = min(20, (budget - calls_made) // 15)

        # Start from best point found so far, plus some random and Sobol points
        start_points = [best_x.copy()]

        # Add some random points
        for _ in range(min(num_local_starts - 1, 10)):
            if calls_made >= budget or time.time() > deadline:
                break
            pt = rng.uniform(-5.0, 5.0, n)
            start_points.append(pt)

        # Add some Sobol points
        if num_local_starts > 11:
            extra_sobol = sobol_sample(n, min(num_local_starts - 11, 10), seed=99)
            start_points.extend([pt for pt in extra_sobol])

        # Run L-BFGS-B from each starting point
        bounds = [(-5.0, 5.0)] * n

        for start_pt in start_points:
            if time.time() > deadline or calls_made >= budget:
                break

            # Create a wrapper that counts calls and enforces budget
            local_calls = [0]

            def lbfgs_func(x):
                nonlocal local_calls
                if time.time() > deadline or calls_made >= budget:
                    return np.inf
                fx = safe_call(x)
                if fx is None:
                    return np.inf
                local_calls[0] += 1
                return fx

            try:
                result = minimize(
                    lbfgs_func,
                    start_pt,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 50, 'maxfun': 20}
                )
            except Exception:
                continue

        # Phase 3: If budget remains, do more fine-tuning around best point
        if calls_made < budget and time.time() < deadline:
            # Use differential evolution for more global search
            remaining = budget - calls_made
            if remaining > 30:
                def de_func(x):
                    if time.time() > deadline or calls_made >= budget:
                        return np.inf
                    fx = safe_call(x)
                    if fx is None:
                        return np.inf
                    return fx

                try:
                    de_result = differential_evolution(
                        de_func,
                        bounds=bounds,
                        maxiter=5,
                        popsize=10,
                        tol=0.01,
                        seed=42,
                        polish=False
                    )
                except Exception:
                    pass

        # Final polish with L-BFGS-B from best point
        if calls_made < budget and time.time() < deadline:
            def final_func(x):
                if time.time() > deadline or calls_made >= budget:
                    return np.inf
                fx = safe_call(x)
                if fx is None:
                    return np.inf
                return fx

            try:
                final_result = minimize(
                    final_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'maxfun': 50, 'ftol': 1e-12, 'gtol': 1e-8}
                )
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
