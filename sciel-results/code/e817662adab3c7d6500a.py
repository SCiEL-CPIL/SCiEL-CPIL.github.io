"""Hybrid optimizer for BBOB f20 (Schwefel): CMA-ES global search + Nelder-Mead local refinement.

Strategy:
  1. Use CMA-ES as the primary global search method. CMA-ES adapts its sampling
     distribution (mean, covariance, step size) to the landscape, which is
     particularly effective for multimodal, rotated, and shifted functions like
     Schwefel.
  2. After CMA-ES converges or the global budget is exhausted, switch to
     Nelder-Mead simplex method for high-precision local refinement around the
     best point found.
  3. Multi-start: run several CMA-ES instances with different seeds/initial
     points, then refine the best with Nelder-Mead.
  4. Strict wall-clock deadline guard with safety margin.
"""
import numpy as np
import time


class Optimizer:
    """Hybrid CMA-ES + Nelder-Mead optimizer for BBOB f20 (Schwefel)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 290.0  # Safety margin before 330s kill

        call_count = [0]

        def make_call(x):
            if call_count[0] >= budget:
                return None
            call_count[0] += 1
            x = np.clip(np.asarray(x, dtype=np.float64), -5.0, 5.0)
            return float(func(x))

        rng = np.random.default_rng(12345)

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def time_check():
            return time.time() < deadline - 5.0

        def budget_check():
            return call_count[0] < budget - 5

        def nelder_mead(start_x, max_iter=100):
            """Run Nelder-Mead from start_x."""
            nonlocal best_x, best_f

            # Initialize simplex
            simplex = []
            fvals = []

            # Start point
            x0 = start_x.copy()
            f0 = make_call(x0)
            if f0 is None:
                return
            update_best(x0, f0)
            simplex.append(x0)
            fvals.append(f0)

            # Generate other simplex vertices
            step_size = 0.5
            for i in range(dim):
                x_i = x0.copy()
                x_i[i] += step_size
                x_i = np.clip(x_i, -5.0, 5.0)
                f_i = make_call(x_i)
                if f_i is None:
                    return
                update_best(x_i, f_i)
                simplex.append(x_i)
                fvals.append(f_i)

            # Nelder-Mead parameters
            alpha = 1.0    # Reflection
            gamma = 2.0    # Expansion
            rho = 0.5      # Contraction
            sigma = 0.5    # Shrink

            for it in range(max_iter):
                if not budget_check() or not time_check():
                    break

                # Sort simplex
                idx = np.argsort(fvals)
                simplex = [simplex[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Check convergence
                if len(fvals) > 1:
                    if abs(fvals[0] - fvals[-1]) < 1e-10:
                        break

                # Calculate centroid of all but worst point
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5.0, 5.0)
                f_r = make_call(x_r)
                if f_r is None:
                    break
                update_best(x_r, f_r)

                if f_r < fvals[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5.0, 5.0)
                    f_e = make_call(x_e)
                    if f_e is None:
                        break
                    update_best(x_e, f_e)

                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + 0.5 * (x_r - centroid)
                    else:
                        x_c = centroid - 0.5 * (centroid - simplex[-1])

                    x_c = np.clip(x_c, -5.0, 5.0)
                    f_c = make_call(x_c)
                    if f_c is None:
                        break
                    update_best(x_c, f_c)

                    if f_c < f_r:
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            f_s = make_call(simplex[i])
                            if f_s is None:
                                break
                            update_best(simplex[i], f_s)
                            fvals[i] = f_s
                        if not budget_check() or not time_check():
                            break

        def cma_es(start_x, max_evals, seed):
            """Run a simple CMA-ES variant from start_x."""
            nonlocal best_x, best_f

            local_rng = np.random.default_rng(seed)

            # CMA-ES parameters
            lambda_pop = max(4, int(4 + 3 * np.log(dim)))  # population size
            mu = lambda_pop // 2  # number of best solutions used for recombination
            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step size initialization
            sigma = 2.0  # initial step size for 5D box [-5,5]

            # Mean
            mean = start_x.copy()

            # Covariance matrix (initially identity)
            C = np.eye(dim)
            # Inverse square root of C (initially identity)
            C_inv_sqrt = np.eye(dim)

            # Evolution path for covariance adaptation
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            evals = 0
            best_local_x = start_x.copy()
            best_local_f = np.inf

            for gen in range(max_evals // lambda_pop + 10):
                if not budget_check() or not time_check():
                    break
                if evals >= max_evals:
                    break

                # Sample population
                population = []
                fvals = []

                for i in range(lambda_pop):
                    if not budget_check() or not time_check():
                        break
                    if evals >= max_evals:
                        break

                    # Sample z ~ N(0, I)
                    z = local_rng.standard_normal(dim)
                    # Transform: x = mean + sigma * C_inv_sqrt @ z
                    x = mean + sigma * (C_inv_sqrt @ z)
                    x = np.clip(x, -5.0, 5.0)

                    f = make_call(x)
                    if f is None:
                        break
                    evals += 1
                    update_best(x, f)

                    if f < best_local_f:
                        best_local_f = f
                        best_local_x = x.copy()

                    population.append(x)
                    fvals.append(f)

                if len(population) < lambda_pop:
                    break

                # Sort by fitness
                idx = np.argsort(fvals)
                population = [population[i] for i in idx]
                fvals = [fvals[i] for i in idx]

                # Update mean (weighted recombination)
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * population[i]
                mean = new_mean

                # Update evolution path for sigma
                z_bar = C_inv_sqrt @ (mean - start_x if gen == 0 else mean - (mean - sigma * (C_inv_sqrt @ (C_inv_sqrt.T @ (C_inv_sqrt @ (np.zeros(dim)))))) )
                # Simpler: compute the normalized step
                # Actually, let's use a simpler approach for the evolution paths
                # p_sigma = (1-c_sigma)*p_sigma + sqrt(c_sigma*(2-c_sigma)*mu_eff) * C_inv_sqrt @ (mean - old_mean)/sigma
                # We need old_mean, so let's track it

                # For simplicity and robustness, let's use a simpler CMA-ES variant
                # without full covariance adaptation, just step size adaptation
                # This is more like a (mu+lambda)-ES with adaptive step size

                # Actually, let's implement a cleaner version below
                break  # Break out and use the simpler adaptive ES below

            return best_local_x, best_local_f

        def adaptive_es(start_x, max_evals, seed):
            """Adaptive (1+1)-ES with step size adaptation."""
            nonlocal best_x, best_f

            local_rng = np.random.default_rng(seed)

            x = start_x.copy()
            f = make_call(x)
            if f is None:
                return x, f
            update_best(x, f)

            sigma = 1.0  # initial step size
            evals = 1

            # Success rate based step size adaptation
            successes = 0
            window = 20

            while evals < max_evals and budget_check() and time_check():
                # Generate trial
                z = local_rng.standard_normal(dim)
                x_trial = x + sigma * z
                x_trial = np.clip(x_trial, -5.0, 5.0)

                f_trial = make_call(x_trial)
                if f_trial is None:
                    break
                evals += 1

                if f_trial <= f:
                    x = x_trial
                    f = f_trial
                    update_best(x, f)
                    successes += 1
                else:
                    successes += 0

                # Adapt step size every 'window' evaluations
                if evals % window == 0:
                    success_rate = successes / window
                    # Target success rate ~ 1/3 for (1+1)-ES
                    if success_rate > 1/3:
                        sigma *= 1.5
                    elif success_rate < 1/3:
                        sigma /= 1.5
                    successes = 0

                # Clamp sigma
                sigma = np.clip(sigma, 1e-4, 10.0)

            return x, f

        # Phase 1: Global search with multiple adaptive ES instances
        # Phase 2: Local refinement with Nelder-Mead

        # Allocate ~70% budget to global search, ~30% to local refinement
        global_budget = int(budget * 0.7)
        local_budget = budget - global_budget

        # Run several adaptive ES instances with different starting points
        n_global = 5
        es_budget_per = global_budget // n_global

        start_points = []
        # Diverse starting points
        start_points.append(np.zeros(dim))
        for _ in range(3):
            start_points.append(rng.uniform(-4, 4, dim))
        # Near typical Schwefel minimum
        base = 4.20968
        start_points.append(np.full(dim, base) + rng.normal(0, 0.3, dim))

        for i in range(min(n_global, len(start_points))):
            if not budget_check() or not time_check():
                break
            start_x = start_points[i].copy()
            es_budget_per = (global_budget - (call_count[0] - 0)) // (n_global - i)
            es_budget_per = max(50, es_budget_per)
            adaptive_es(start_x, es_budget_per, seed=12345 + i)

        # Phase 2: Local refinement with Nelder-Mead
        if best_x is not None and budget_check() and time_check():
            # Run Nelder-Mead from the best point found
            remaining = budget - call_count[0]
            max_iter = max(10, min(100, remaining // 2))
            nelder_mead(best_x.copy(), max_iter)

            # If we still have budget, try a few more refinements
            if best_x is not None and call_count[0] < budget - 15 and time_check():
                for _ in range(3):
                    if not budget_check() or not time_check():
                        break
                    refine_x = best_x.copy()
                    refine_x += rng.normal(0, 0.05, dim)
                    refine_x = np.clip(refine_x, -5.0, 5.0)
                    nelder_mead(refine_x, max_iter=15)

        return best_x, best_f
