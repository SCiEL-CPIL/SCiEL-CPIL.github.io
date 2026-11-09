"""Combined optimizer: Sobol global scan + CMA-ES refinement with adaptive restarts."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math
import scipy.stats.qmc
import scipy.optimize


class Optimizer:
    """Sobol global scan followed by population-based CMA-ES refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Phase 1: Global scan using Sobol quasi-random sequence for better coverage
        n_scan = min(50, budget // 5)
        sampler = scipy.stats.qmc.Sobol(d=dim, scramble=True, seed=42)
        points = sampler.random(n_scan)
        points = points * 10 - 5  # Map to [-5, 5]^dim

        best_x = None
        best_f = np.inf
        candidates = []

        for x in points:
            if time.time() > deadline - 5.0:
                break
            fx = func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            candidates.append((fx, x.copy()))

        if best_x is None:
            x0 = rng.uniform(-4, 4, dim)
            fx0 = func(x0)
            return x0, fx0

        # Sort candidates by function value
        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES refinement on top candidates
        remaining = budget - len(candidates)
        if remaining <= 0:
            return best_x, best_f

        # Take top 3 candidates for refinement
        top_k = min(3, len(candidates))
        top_candidates = candidates[:top_k]

        # Allocate budget: 60% to best, 25% to second, 15% to third
        weights = [0.6, 0.25, 0.15][:top_k]
        total_w = sum(weights)
        weights = [w / total_w for w in weights]
        refine_budgets = [max(10, int(remaining * w)) for w in weights]

        # CMA-ES refinement function
        def run_cma_es(start_x, start_f, n_evals):
            x = start_x.copy()
            fx = start_f
            sigma = 2.0  # Large initial sigma for multimodal landscape

            # Use scipy's CMA-ES implementation
            # We need to wrap func to count evaluations and enforce deadline
            eval_count = [0]
            
            def wrapped_func(z):
                if time.time() > deadline - 2.0:
                    return 1e10  # Penalize if time is up
                if eval_count[0] >= n_evals:
                    return 1e10
                eval_count[0] += 1
                z = np.clip(z, -5, 5)
                return func(z)

            # Set up CMA-ES options
            options = {
                'maxiter': n_evals,
                'verbose': 0,
                'seed': 42,
                'bounds': [(-5.0, 5.0)] * dim
            }

            try:
                result = scipy.optimize.minimize(
                    wrapped_func,
                    x,
                    method='CMA-ES',
                    options=options
                )
                if result.fun < fx:
                    fx = result.fun
                    x = result.x.copy()
            except Exception:
                # Fallback to (1+1)-ES if CMA-ES fails
                x, fx = self._fallback_es(x, fx, n_evals, func, deadline)

            return x, fx

        # Fallback (1+1)-ES if CMA-ES fails
        def _fallback_es(self, x, fx, n_evals, func, deadline):
            sigma = 1.0
            successes = 0
            total = 0
            stagnant = 0

            for _ in range(n_evals):
                if time.time() > deadline - 2.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    successes += 1
                    stagnant = 0
                else:
                    stagnant += 1

                total += 1
                if total % 10 == 0:
                    success_rate = successes / total
                    if success_rate > 0.3:
                        sigma *= 1.5
                    elif success_rate < 0.1:
                        sigma *= 0.7
                    successes = 0
                    total = 0

                if stagnant > 50:
                    x = np.clip(x + rng.standard_normal(dim) * 2.0, -5, 5)
                    fx = func(x)
                    sigma = 2.0
                    stagnant = 0

            return x, fx

        # Refine each top candidate
        for rank, (fx_init, x_init) in enumerate(top_candidates):
            if time.time() > deadline - 10.0:
                break
            if refine_budgets[rank] <= 0:
                continue

            x_refined, f_refined = run_cma_es(x_init, fx_init, refine_budgets[rank])
            if f_refined < best_f:
                best_x, best_f = x_refined.copy(), f_refined

        # Phase 3: Final polish with very small steps
        if time.time() < deadline - 3.0 and remaining > 5:
            x = best_x.copy()
            fx = best_f
            sigma = 0.05

            polish_steps = min(20, remaining)
            for _ in range(polish_steps):
                if time.time() > deadline - 1.0:
                    break

                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = func(y)

                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.05
                    if fx < best_f:
                        best_x, best_f = x.copy(), fx
                else:
                    sigma *= 0.9
                    if sigma < 0.005:
                        sigma = 0.02

        return best_x, best_f
# EVOLVE-BLOCK-END
