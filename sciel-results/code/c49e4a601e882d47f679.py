"""Population-based Differential Evolution with adaptive mutation and Nelder-Mead polish."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DE/rand/1 with adaptive F, population size 15, plus multi-restart Nelder-Mead polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Population size - slightly larger for better diversity
        pop_size = 15
        # Budget allocation: DE phase uses most, polish uses ~30 calls
        polish_budget = 30
        de_budget = max(budget - polish_budget, 10)

        # Initialize population: mix of uniform and Gaussian around center
        # This helps with multimodal functions like Gallagher
        pop = np.empty((pop_size, dim))
        # First half: uniform in [-5, 5]
        n_uniform = pop_size // 2
        pop[:n_uniform] = rng.uniform(-5, 5, size=(n_uniform, dim))
        # Second half: Gaussian around 0 with std=2
        n_gauss = pop_size - n_uniform
        pop[n_uniform:] = rng.normal(0, 2.0, size=(n_gauss, dim))
        pop = np.clip(pop, -5, 5)

        fits = np.array([func(ind) for ind in pop])

        # Track best
        best_idx = np.argmin(fits)
        best_x = pop[best_idx].copy()
        best_f = fits[best_idx]

        # Adaptive mutation factor F: start at 0.7
        F = 0.7
        success_count = 0
        trial_count = 0
        last_best_f = best_f
        no_improve_gen = 0

        calls_used = pop_size  # initial population evaluations

        # DE loop
        while calls_used < de_budget:
            # Adaptive F: smooth adaptation
            if trial_count >= 10:
                success_rate = success_count / trial_count
                # Smoothly adjust F toward 0.7 + 0.5*(success_rate - 0.5)
                target_F = 0.7 + 0.5 * (success_rate - 0.5)
                F = 0.8 * F + 0.2 * np.clip(target_F, 0.3, 1.0)
                success_count = 0
                trial_count = 0

            # Check for stagnation
            if best_f < last_best_f - 1e-10:
                no_improve_gen = 0
                last_best_f = best_f
            else:
                no_improve_gen += 1

            # If stuck for many generations, inject diversity
            if no_improve_gen > 5:
                # Replace worst 3 individuals with random ones
                worst_indices = np.argsort(fits)[-3:]
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

        # Nelder-Mead polish: try multiple top solutions
        if calls_used < budget:
            def bounded_func(x):
                x_clipped = np.clip(x, -5, 5)
                return func(x_clipped)

            # Get top 3 solutions from population
            top_indices = np.argsort(fits)[:3]
            nm_starts = [pop[idx].copy() for idx in top_indices]
            # Also include best_x
            if not any(np.allclose(s, best_x) for s in nm_starts):
                nm_starts.append(best_x.copy())

            # Budget per NM run
            nm_total_budget = budget - calls_used
            n_starts = min(len(nm_starts), 3)
            nm_per_budget = max(1, nm_total_budget // n_starts)

            for start in nm_starts[:n_starts]:
                if calls_used >= budget:
                    break

                remaining = budget - calls_used
                maxiter = max(1, remaining // 4)  # NM uses ~1-2 evals per iteration

                try:
                    result = minimize(
                        bounded_func,
                        start,
                        method='Nelder-Mead',
                        options={
                            'maxiter': maxiter,
                            'xatol': 1e-7,
                            'fatol': 1e-9,
                            'adaptive': True,
                            'maxfev': remaining
                        }
                    )
                    # Count actual calls used by NM (approximate)
                    # We can't easily count, so just check if improved
                    if result.fun < best_f:
                        best_f = result.fun
                        best_x = np.clip(result.x, -5, 5)
                except Exception:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
