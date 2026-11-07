"""Rewritten optimizer using Nelder-Mead with random restarts and time guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead simplex with random initialization and time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320  # safety margin before 330s kill

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Phase 1: Random initialization - sample 20 random points
        n_init = 20
        candidates = []
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            candidates.append((fx, x.copy()))
            if time.time() > deadline:
                break

        if not candidates:
            # Fallback: single random point
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            return x, fx

        # Sort by fitness
        candidates.sort(key=lambda t: t[0])
        best_fx, best_x = candidates[0]

        # Phase 2: Nelder-Mead refinement on top candidates
        # Use remaining budget for local refinement
        remaining_budget = budget - len(candidates)
        if remaining_budget < 10:
            return best_x, best_fx

        # Try Nelder-Mead on top 3 candidates
        n_top = min(3, len(candidates))
        nm_budget_per = remaining_budget // n_top

        for i in range(n_top):
            if time.time() > deadline:
                break

            fx_init, x_init = candidates[i]
            if fx_init >= best_fx:
                break

            # Nelder-Mead parameters
            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma = 0.5  # shrink
            initial_step = 0.5

            # Build initial simplex
            simplex = [x_init.copy()]
            fvals = [fx_init]
            calls_used = 1

            for d in range(dim):
                x_trial = x_init.copy()
                x_trial[d] += initial_step
                x_trial = np.clip(x_trial, -5, 5)
                ft = func(x_trial)
                simplex.append(x_trial)
                fvals.append(ft)
                calls_used += 1

            # Nelder-Mead iteration
            nm_calls = 0
            max_nm_iters = nm_budget_per - calls_used

            for it in range(max_nm_iters):
                if time.time() > deadline:
                    break

                # Sort simplex by fitness
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]

                # Check convergence
                if abs(fvals[0] - fvals[-1]) < 1e-10:
                    break

                worst = simplex[-1]
                second_worst = simplex[-2]
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_reflect = centroid + alpha * (centroid - worst)
                x_reflect = np.clip(x_reflect, -5, 5)
                f_reflect = func(x_reflect)
                nm_calls += 1

                if f_reflect < fvals[0]:
                    # Expansion
                    x_expand = centroid + gamma * (x_reflect - centroid)
                    x_expand = np.clip(x_expand, -5, 5)
                    f_expand = func(x_expand)
                    nm_calls += 1

                    if f_expand < f_reflect:
                        simplex[-1] = x_expand
                        fvals[-1] = f_expand
                    else:
                        simplex[-1] = x_reflect
                        fvals[-1] = f_reflect
                elif f_reflect < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_reflect
                    fvals[-1] = f_reflect
                else:
                    # Contraction
                    x_contra = centroid + rho * (worst - centroid)
                    x_contra = np.clip(x_contra, -5, 5)
                    f_contra = func(x_contra)
                    nm_calls += 1

                    if f_contra < fvals[-1]:
                        simplex[-1] = x_contra
                        fvals[-1] = f_contra
                    else:
                        # Shrink
                        for j in range(1, dim + 1):
                            simplex[j] = simplex[0] + sigma * (simplex[j] - simplex[0])
                            simplex[j] = np.clip(simplex[j], -5, 5)
                            fvals[j] = func(simplex[j])
                            nm_calls += 1

                # Update best
                current_best = min(fvals)
                if current_best < best_fx:
                    best_fx = current_best
                    best_x = simplex[np.argmin(fvals)].copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
