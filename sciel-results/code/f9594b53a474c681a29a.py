"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based differential evolution with Nelder-Mead polishing for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_f = np.inf
        call_count = 0

        def safe_call(x):
            nonlocal call_count, best_x, best_f
            if call_count >= budget:
                return None
            if time.time() > deadline:
                return None
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            call_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # --- Phase 1: Differential Evolution with small population ---
        # Population size lambda=5 for DE
        lam = 5
        n_de = 5  # number of DE individuals
        F = 0.8   # mutation scale factor
        CR = 0.5  # crossover rate

        # Initialize population
        pop = []
        pop_f = []

        # Include a few structured starts for diversity
        structured = [
            np.zeros(dim),
            rng.uniform(-4, 4, dim),
            rng.uniform(-3, 3, dim),
            rng.uniform(-5, 5, dim),
            rng.uniform(-2, 2, dim),
        ]

        for i in range(n_de):
            x = np.clip(structured[i], -5.0, 5.0)
            fx = safe_call(x)
            if fx is None:
                break
            pop.append(x)
            pop_f.append(fx)

        # Run DE for a portion of the budget
        de_budget_fraction = 0.5  # use 50% of budget for DE
        de_max_evals = int(budget * de_budget_fraction)
        de_evals_used = n_de

        while de_evals_used < de_max_evals:
            if time.time() > deadline - 10.0:
                break

            # Sort population by fitness
            order = np.argsort(pop_f)
            pop = [pop[i] for i in order]
            pop_f = [pop_f[i] for i in order]

            # For each individual, create a trial
            for i in range(n_de):
                if de_evals_used >= de_max_evals:
                    break
                if time.time() > deadline - 10.0:
                    break

                # Select three distinct individuals other than i
                candidates = [j for j in range(n_de) if j != i]
                if len(candidates) < 3:
                    break
                idx_a, idx_b, idx_c = rng.choice(candidates, size=3, replace=False)

                # Mutation: x_best + F * (x_a - x_b)  (using best as reference for attractive sector)
                v = pop[0] + F * (pop[idx_a] - pop[idx_b])

                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                trial = np.clip(trial, -5.0, 5.0)

                ft = safe_call(trial)
                de_evals_used += 1
                if ft is None:
                    break

                # Selection
                if ft <= pop_f[i]:
                    pop[i] = trial
                    pop_f[i] = ft

        # --- Phase 2: Nelder-Mead polishing from the best DE result ---
        remaining = budget - call_count
        if remaining > 0 and not (time.time() > deadline - 10.0):
            # Run Nelder-Mead from the best individual found by DE
            x0 = pop[0].copy() if pop else np.zeros(dim)

            def nelder_mead(x0, max_evals):
                nonlocal best_x, best_f
                n = dim
                simplex = [x0.copy()]
                for i in range(n):
                    xi = x0.copy()
                    step = 0.5
                    if xi[i] > 0:
                        xi[i] -= step
                    else:
                        xi[i] += step
                    simplex.append(np.clip(xi, -5, 5))

                fvals = []
                for i in range(n + 1):
                    fv = safe_call(simplex[i])
                    if fv is None:
                        return
                    fvals.append(fv)

                alpha, gamma, rho, sigma = 1.0, 2.0, 0.5, 0.5
                max_iter = max(1, (max_evals - n - 1) // (n + 1))

                for _ in range(max_iter):
                    if time.time() > deadline - 5.0:
                        break
                    if call_count >= budget:
                        break

                    # Sort simplex
                    order = np.argsort(fvals)
                    simplex = [simplex[i] for i in order]
                    fvals = [fvals[i] for i in order]

                    centroid = np.mean(simplex[:-1], axis=0)

                    # Reflection
                    xr = centroid + alpha * (centroid - simplex[-1])
                    xr = np.clip(xr, -5, 5)
                    fr = safe_call(xr)
                    if fr is None:
                        break

                    if fr < fvals[0]:
                        # Expansion
                        xe = centroid + gamma * (centroid - simplex[-1])
                        xe = np.clip(xe, -5, 5)
                        fe = safe_call(xe)
                        if fe is None:
                            break
                        if fe < fr:
                            simplex[-1] = xe
                            fvals[-1] = fe
                        else:
                            simplex[-1] = xr
                            fvals[-1] = fr
                    elif fr < fvals[-2]:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    else:
                        # Contraction
                        if fr < fvals[-1]:
                            xc = centroid + rho * (xr - centroid)
                        else:
                            xc = centroid - rho * (centroid - simplex[-1])
                        xc = np.clip(xc, -5, 5)
                        fc = safe_call(xc)
                        if fc is None:
                            break
                        if fc < fvals[-1]:
                            simplex[-1] = xc
                            fvals[-1] = fc
                        else:
                            # Shrink
                            for i in range(1, n + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                simplex[i] = np.clip(simplex[i], -5, 5)
                                fs = safe_call(simplex[i])
                                if fs is None:
                                    return
                                fvals[i] = fs

                    # Update global best
                    idx_best = np.argmin(fvals)
                    if fvals[idx_best] < best_f:
                        best_f = fvals[idx_best]
                        best_x = simplex[idx_best].copy()

            nelder_mead(x0, remaining)

        return best_x, best_f
# EVOLVE-BLOCK-END
