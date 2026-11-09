"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Deterministic global search using Differential Evolution followed by single high-precision L-BFGS-B polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Count function calls
        call_count = [0]

        def counted_func(x):
            x = np.asarray(x, dtype=float).ravel()
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            call_count[0] += 1
            return val

        # --- Phase 1: Differential Evolution with adaptive parameters ---
        rng = np.random.default_rng(42)

        # Population size: larger for better global coverage
        pop_size = 40
        # Allocate 85% of budget to DE for thorough global search
        de_budget = int(budget * 0.85)

        # Latin Hypercube sampling for better initial coverage
        pop = np.zeros((pop_size, dim))
        for d in range(dim):
            # Divide [-5, 5] into pop_size equal intervals, sample one point per interval
            intervals = np.linspace(-5.0, 5.0, pop_size + 1)
            points = rng.uniform(intervals[:-1], intervals[1:])
            # Shuffle to decorrelate dimensions
            perm = rng.permutation(pop_size)
            pop[:, d] = points[perm]
        pop_f = np.array([counted_func(ind) for ind in pop])

        # Track best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # DE Loop with adaptive F and CR
        F_start, F_end = 0.8, 0.4
        CR_start, CR_end = 0.9, 0.6
        total_generations = max(1, (de_budget - call_count[0]) // pop_size)
        gen = 0

        while call_count[0] < de_budget:
            gen += 1
            # Adaptive parameters: decrease F and CR over time for convergence
            progress = min(1.0, gen / max(1, total_generations))
            F = F_start + (F_end - F_start) * progress
            CR = CR_start + (CR_end - CR_start) * progress

            # Find current best
            best_idx = np.argmin(pop_f)
            best_x = pop[best_idx].copy()
            best_f = pop_f[best_idx]

            for i in range(pop_size):
                if call_count[0] >= de_budget:
                    break

                # Select three distinct random indices different from i
                mask = np.ones(pop_size, dtype=bool)
                mask[i] = False
                candidates = np.where(mask)[0]
                if len(candidates) < 3:
                    break
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # Mutation: v = pop[a] + F * (pop[b] - pop[c])
                v = pop[a] + F * (pop[b] - pop[c])

                # Binary crossover with adaptive CR
                j_rand = rng.integers(0, dim)
                trial = np.where(
                    (rng.random(dim) < CR) | (np.arange(dim) == j_rand),
                    v,
                    pop[i]
                )

                # Clip to bounds
                trial = np.clip(trial, -5.0, 5.0)

                # Evaluate trial
                trial_f = counted_func(trial)

                # Selection
                if trial_f <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = trial_f

        # After DE, find best
        best_idx = np.argmin(pop_f)
        best_x = pop[best_idx].copy()
        best_f = pop_f[best_idx]

        # --- Phase 2: Single high-precision L-BFGS-B polish on best candidate ---
        remaining = budget - call_count[0]
        if remaining >= 10:
            bounds = [(-5.0, 5.0)] * dim
            max_fun = remaining - 1  # Leave 1 call buffer
            
            try:
                result = minimize(
                    counted_func,
                    best_x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': max(10, max_fun // (dim + 1)),
                        'maxfun': max_fun,
                        'ftol': 1e-15,
                        'gtol': 1e-15,
                        'pgtol': 1e-15
                    }
                )
                if result.fun < best_f:
                    best_x = result.x
                    best_f = result.fun
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
