"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """DE global search + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(0)

        # Track global best
        best_x = np.zeros(dim)
        best_f = func(best_x)
        calls_used = 1

        # Phase 1: Differential Evolution for global exploration
        N = 8  # population size
        F = 0.8  # scaling factor
        CR = 0.9  # crossover rate

        # Initialize population
        pop = np.array([best_x] + [
            rng.uniform(-5, 5, dim) for _ in range(N - 1)
        ])
        pop_f = np.array([best_f] + [
            func(pop[i]) for i in range(1, N)
        ])
        calls_used += N - 1

        # DE loop
        max_de_iter = 40
        for it in range(max_de_iter):
            if calls_used >= budget or time.time() > deadline - 30.0:
                break

            for i in range(N):
                if calls_used >= budget or time.time() > deadline - 30.0:
                    break

                # Select three distinct indices
                a, b, c = rng.choice(N, size=3, replace=False)
                while a == i or b == i or c == i or a == b or b == c or a == c:
                    a, b, c = rng.choice(N, size=3, replace=False)

                # Create trial vector
                v = pop[a] + F * (pop[b] - pop[c])
                v = np.clip(v, -5, 5)

                # Crossover
                r = rng.random()
                jrand = rng.integers(dim)
                u = np.where(rng.random(dim) < CR, v, pop[i])
                u[jrand] = v[jrand]
                u = np.clip(u, -5, 5)

                # Evaluate
                fu = func(u)
                calls_used += 1

                # Selection
                if fu <= pop_f[i]:
                    pop[i] = u
                    pop_f[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()

        # Phase 2: Local refinement using Nelder-Mead from top candidates
        # Sort population by fitness and refine the best few
        sorted_idx = np.argsort(pop_f)
        num_refine = min(3, N)
        refine_budget = max(50, budget - calls_used)

        for k in range(num_refine):
            if calls_used >= budget or time.time() > deadline - 10.0:
                break

            x0 = pop[sorted_idx[k]].copy()
            f0 = pop_f[sorted_idx[k]]

            # Nelder-Mead with limited function evaluations
            def wrapper(x):
                nonlocal calls_used
                x_clipped = np.clip(x, -5, 5)
                calls_used += 1
                return func(x_clipped)

            try:
                result = minimize(
                    wrapper, x0, method='Nelder-Mead',
                    options={'maxiter': refine_budget // num_refine,
                             'xatol': 1e-6, 'fatol': 1e-6}
                )
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
