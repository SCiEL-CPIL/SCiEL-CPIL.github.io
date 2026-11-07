"""Population-based Differential Evolution with adaptive mutation and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DE/rand/1 with adaptive F, population size 20, plus multi-restart Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size
        pop_size = 20
        # Budget allocation: DE phase uses most, polish uses ~50 calls
        polish_budget = 50
        de_budget = max(budget - polish_budget, 10)

        # Initialize population: mix of uniform and Gaussian around center
        pop = np.empty((pop_size, dim))
        n_uniform = pop_size // 2
        pop[:n_uniform] = rng.uniform(-5, 5, size=(n_uniform, dim))
        n_gauss = pop_size - n_uniform
        pop[n_uniform:] = rng.normal(0, 2.5, size=(n_gauss, dim))
        pop = np.clip(pop, -5, 5)

        fits = np.array([func(ind) for ind in pop])

        # Track best
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]

        # Adaptive mutation factor F: start at 0.8
        F = 0.8
        success_count = 0
        trial_count = 0
        last_best_f = best_f
        no_improve_gen = 0

        calls_used = pop_size  # initial population evaluations

        # DE loop
        while calls_used < de_budget:
            # Adaptive F: smooth adaptation
            if trial_count >= 15:
                success_rate = success_count / trial_count
                target_F = 0.6 + 0.6 * success_rate
                F = 0.7 * F + 0.3 * np.clip(target_F, 0.3, 1.0)
                success_count = 0
                trial_count = 0

            # Check for stagnation
            if best_f < last_best_f - 1e-10:
                no_improve_gen = 0
                last_best_f = best_f
            else:
                no_improve_gen += 1

            # If stuck for many generations, inject diversity
            if no_improve_gen > 4:
                n_reinject = min(5, pop_size // 4)
                worst_indices = np.argsort(fits)[-n_reinject:]
                for idx in worst_indices:
                    pop[idx] = rng.uniform(-5, 5, size=dim)
                    fits[idx] = func(pop[idx])
                    calls_used += 1
                    if fits[idx] < best_f:
                        best_f = fits[idx]
                        best_x = pop[idx].copy()
                    if calls_used >= de_budget:
                        break

            for i in range(pop_size):
                if calls_used >= de_budget:
                    break

                # Select three distinct random indices different from i
                candidates = np.array([j for j in range(pop_size) if j != i])
                a, b, c = rng.choice(candidates, size=3, replace=False)

                # DE/rand/1 mutation
                mutant = pop[a] + F * (pop[b] - pop[c])
                mutant = np.clip(mutant, -5, 5)

                # Evaluate mutant
                fm = func(mutant)
                calls_used += 1
                trial_count += 1

                # Selection: replace if mutant is better
                if fm < fits[i]:
                    pop[i] = mutant
                    fits[i] = fm
                    success_count += 1

                    # Update global best
                    if fm < best_f:
                        best_f = fm
                        best_x = mutant.copy()

        # Nelder-Mead polish: try multiple top solutions with careful call counting
        if calls_used < budget:
            call_counter = {'count': calls_used}

            def bounded_func(x):
                x_clipped = np.clip(x, -5, 5)
                call_counter['count'] += 1
                return func(x_clipped)

            # Get top 5 solutions from population for more diverse polish starts
            top_indices = np.argsort(fits)[:5]
            nm_starts = [pop[idx].copy() for idx in top_indices]
            # Also include best_x
            if not any(np.allclose(s, best_x, atol=1e-6) for s in nm_starts):
                nm_starts.append(best_x.copy())

            # Budget per NM run
            nm_total_budget = budget - calls_used
            n_starts = min(len(nm_starts), 4)
            nm_per_budget = max(1, nm_total_budget // n_starts)

            for start in nm_starts[:n_starts]:
                if call_counter['count'] >= budget:
                    break

                remaining = budget - call_counter['count']
                maxiter = max(1, remaining // 5)
                maxfev = max(1, remaining)

                try:
                    result = minimize(
                        bounded_func,
                        start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': maxiter,
                            'xatol': 1e-6,
                            'fatol': 1e-8,
                            'adaptive': True,
                            'maxfev': maxfev
                        }
                    )
                    # Update calls_used
                    calls_used = call_counter['count']
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    calls_used = call_counter['count']

        return best_x, best_f
# EVOLVE-BLOCK-END
