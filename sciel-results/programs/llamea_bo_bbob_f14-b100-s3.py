"""Optimizer for BBOB f14 (Different powers) with hidden shift/rotation/offset."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Sobol global search followed by BFGS local refinement with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 315.0  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget

        # --- Phase 1: Sobol-based global exploration ---
        # Generate Sobol points for structured coverage
        def sobol_points(n, dim, seed=42):
            """Generate n Sobol-like points in [0,1]^dim using a simple quasi-random sequence."""
            rng = np.random.default_rng(seed)
            # Use a Halton-like sequence for quasi-random coverage
            points = np.zeros((n, dim))
            for d in range(dim):
                base = d + 2  # use primes as bases: 2, 3, 5, 7, 11
                for i in range(n):
                    num = i + 1
                    result = 0.0
                    denom = 1.0
                    while num > 0:
                        num, rem = divmod(num, base)
                        denom *= base
                        result += rem / denom
                    points[i, d] = result
            return points

        # Generate 30 Sobol points for global exploration
        n_global = 30
        sobol = sobol_points(n_global, dim)
        # Map to [-5, 5]^dim
        global_points = sobol * 10.0 - 5.0

        # Also include origin and a few random points
        extra_points = [np.zeros(dim)]
        rng = np.random.default_rng(123)
        for _ in range(5):
            extra_points.append(rng.uniform(-5, 5, dim))

        all_candidates = list(global_points) + extra_points

        best_x = None
        best_f = np.inf
        calls_used = 0

        for c in all_candidates:
            if time.time() > deadline:
                break
            fc = float(func(c))
            calls_used += 1
            if fc < best_f:
                best_f = fc
                best_x = c.copy()

        remaining = budget - calls_used

        if remaining <= 0:
            return best_x, best_f

        # --- Phase 2: BFGS local refinement from the best global point ---
        x = best_x.copy()
        fx = best_f

        # Use BFGS with multiple restarts from top candidates
        # Sort candidates by their function values
        candidate_scores = []
        for c in all_candidates:
            if time.time() > deadline:
                break
            # We already have values for all_candidates, but let's re-evaluate if needed
            # Actually we can just use the ones we already computed
            pass

        # Let's do BFGS from the best point and a few other good points
        # First, do BFGS from the best point
        try:
            result = minimize(
                func,
                x,
                method='BFGS',
                bounds=[(-5, 5)] * dim,
                options={'maxiter': 100, 'gtol': 1e-8}
            )
            if result.success or result.fun < fx:
                fx = float(result.fun)
                x = np.clip(result.x, -5, 5)
            calls_used += result.nfev
        except Exception:
            pass

        if time.time() > deadline or calls_used >= budget:
            return x, fx

        # Try BFGS from a few other promising starting points
        # Get the top 3 candidates by their initial evaluations
        evals = []
        for c in all_candidates:
            # We need to re-evaluate or cache; let's just try a few random restarts
            pass

        # Do a few more BFGS restarts from random points near the best
        for restart in range(3):
            if time.time() > deadline:
                break
            if calls_used >= budget:
                break

            # Start from a perturbation of the best point
            x0 = x + rng.normal(0, 0.5, dim)
            x0 = np.clip(x0, -5, 5)

            try:
                result = minimize(
                    func,
                    x0,
                    method='BFGS',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'gtol': 1e-8}
                )
                if result.fun < fx:
                    fx = float(result.fun)
                    x = np.clip(result.x, -5, 5)
                calls_used += result.nfev
            except Exception:
                pass

        return x, fx
# EVOLVE-BLOCK-END
