"""
Hybrid optimizer: Diagonal CMA-ES for global exploration + Nelder-Mead for local refinement.
The Discus function (f11) is highly anisotropic with a flat plateau. Diagonal CMA-ES adapts
per-dimension step sizes to handle anisotropy, while Nelder-Mead provides efficient local
convergence. Multi-start strategy with adaptive budget allocation.
"""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s kill

        best_x = np.zeros(dim)
        best_f = np.inf
        total_calls = 0

        # Helper to safely call func and track budget
        def safe_call(x):
            nonlocal total_calls, best_f, best_x
            if total_calls >= budget:
                return None
            x = np.clip(x, -5.0, 5.0)
            total_calls += 1
            f_val = func(x)
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        # --- Diagonal CMA-ES ---
        def diag_cma_es(x0, f0, max_calls, sigma0=1.0):
            """
            Diagonal CMA-ES: population-based optimizer with per-dimension step sizes.
            Uses rank-1 and rank-mu covariance updates on the diagonal.
            """
            nonlocal total_calls, best_f, best_x

            if total_calls >= budget:
                return best_x, best_f

            lam = max(4, 2 * dim + 2)  # population size
            mu = lam // 2  # number of weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Learning rates
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initial state
            mean = x0.copy()
            sigma = sigma0
            ps = np.zeros(dim)  # evolution path for sigma
            pc = np.zeros(dim)  # evolution path for covariance
            diag = np.ones(dim)  # diagonal of covariance matrix

            start_calls = total_calls

            while total_calls - start_calls < max_calls and total_calls < budget:
                if time.time() > deadline - 5:
                    break

                # Generate population
                offspring = []
                for i in range(lam):
                    z = np.random.standard_normal(dim)
                    xi = mean + sigma * z * np.sqrt(diag)
                    xi = np.clip(xi, -5.0, 5.0)
                    fi = safe_call(xi)
                    if fi is None:
                        break
                    offspring.append((xi, fi))

                if len(offspring) < lam:
                    break

                # Sort by fitness
                offspring.sort(key=lambda t: t[1])

                # Update mean
                new_mean = np.zeros(dim)
                for i in range(mu):
                    new_mean += weights[i] * offspring[i][0]
                mean = new_mean

                # Update sigma using evolution path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (mean - x0)
                # x0 is the old mean before update; we need to track it properly
                # Let's restructure: keep track of old_mean

                # Update diagonal covariance using rank-1 and rank-mu
                # For simplicity, use a simplified update
                for i in range(dim):
                    # Rank-1 update
                    diff = (mean - x0) / sigma
                    diag[i] *= np.exp(c_1 * (diff[i] ** 2 - 1))
                    # Rank-mu update (simplified)
                    for j in range(mu):
                        diff_j = (offspring[j][0] - x0) / sigma
                        diag[i] *= np.exp(c_mu * weights[j] * (diff_j[i] ** 2 - 1))

                # Clip diagonal to reasonable range
                diag = np.clip(diag, 1e-6, 1e6)

                # Update sigma based on path length
                ps_norm = np.linalg.norm(ps)
                chi_n = dim * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
                sigma *= np.exp(c_sigma / c_sigma * (ps_norm / chi_n - 1))
                sigma = np.clip(sigma, 1e-8, 100)

                # Check convergence
                if sigma < 1e-8:
                    break

                # Update x0 for next iteration
                x0 = mean

            return best_x, best_f

        # --- Nelder-Mead with adaptive step sizes ---
        def nelder_mead(x0, f0, max_calls, initial_step=1.0):
            """
            Nelder-Mead with adaptive step sizes and convergence checks.
            """
            nonlocal total_calls, best_f, best_x

            if total_calls >= budget:
                return best_x, best_f

            # Initialize simplex
            simplex = [(x0.copy(), f0)]
            start_calls = total_calls

            # Expand simplex with adaptive step size
            for i in range(dim):
                if total_calls >= budget or total_calls - start_calls >= max_calls:
                    break
                xi = x0.copy()
                step = initial_step
                xi[i] += step
                xi = np.clip(xi, -5.0, 5.0)
                fi = safe_call(xi)
                if fi is None:
                    break
                simplex.append((xi, fi))

            if len(simplex) < dim + 1:
                return best_x, best_f

            simplex.sort(key=lambda t: t[1])

            alpha = 0.5
            gamma = 2.0
            rho = 0.5
            sigma_shrink = 0.5

            no_improve = 0
            prev_best = best_f

            while total_calls - start_calls < max_calls and total_calls < budget:
                if time.time() > deadline - 5:
                    break

                # Check convergence
                x_center = np.mean([s[0] for s in simplex], axis=0)
                spread = np.max(np.abs(np.array([s[0] for s in simplex]) - x_center))
                if spread < 1e-6:
                    break

                x_worst = simplex[-1][0]
                f_worst = simplex[-1][1]
                x_second_worst = simplex[-2][0]
                f_second_worst = simplex[-2][1]

                x_centroid = np.mean([s[0] for s in simplex[:-1]], axis=0)

                # Reflection
                x_reflect = x_centroid + gamma * (x_centroid - x_worst)
                x_reflect = np.clip(x_reflect, -5.0, 5.0)
                f_reflect = safe_call(x_reflect)
                if f_reflect is None:
                    break

                if f_reflect < simplex[0][1]:
                    # Expansion
                    x_expand = x_centroid + alpha * (x_reflect - x_worst)
                    x_expand = np.clip(x_expand, -5.0, 5.0)
                    f_expand = safe_call(x_expand)
                    if f_expand is None:
                        break
                    if f_expand < f_reflect:
                        simplex[-1] = (x_expand, f_expand)
                    else:
                        simplex[-1] = (x_reflect, f_reflect)
                elif f_reflect < f_second_worst:
                    simplex[-1] = (x_reflect, f_reflect)
                else:
                    # Contraction
                    if f_reflect < f_worst:
                        x_contract = x_centroid + rho * (x_reflect - x_worst)
                    else:
                        x_contract = x_centroid - rho * (x_worst - x_centroid)
                    x_contract = np.clip(x_contract, -5.0, 5.0)
                    f_contract = safe_call(x_contract)
                    if f_contract is None:
                        break
                    if f_contract <= min(f_reflect, f_worst):
                        simplex[-1] = (x_contract, f_contract)
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            if total_calls >= budget:
                                break
                            new_x = simplex[0][0] + sigma_shrink * (simplex[i][0] - simplex[0][0])
                            new_x = np.clip(new_x, -5.0, 5.0)
                            new_f = safe_call(new_x)
                            if new_f is None:
                                break
                            simplex[i] = (new_x, new_f)
                        if total_calls >= budget:
                            break

                simplex.sort(key=lambda t: t[1])

                # Check for stagnation
                if best_f < prev_best - 1e-8:
                    no_improve = 0
                    prev_best = best_f
                else:
                    no_improve += 1
                    if no_improve > 20:
                        break

            return best_x, best_f

        # --- Multi-Start Strategy ---
        rng = np.random.default_rng(42)

        # Phase 1: Global exploration with Diagonal CMA-ES
        # Use ~40% of budget for global search
        cma_budget = int(budget * 0.4)

        # Start CMA-ES from center
        x0_cma = np.zeros(dim)
        f0_cma = safe_call(x0_cma)
        if f0_cma is not None:
            diag_cma_es(x0_cma.copy(), f0_cma, cma_budget, sigma0=2.0)

        # Phase 2: Local refinement with Nelder-Mead from best point
        # Use ~40% of budget for local refinement
        nm_budget = int(budget * 0.4)
        if total_calls < budget:
            nelder_mead(best_x.copy(), best_f, nm_budget, initial_step=0.5)

        # Phase 3: Additional local refinement from diverse points
        # Use remaining budget
        remaining = budget - total_calls
        if remaining > 10:
            # Try a few more diverse starting points
            for i in range(min(3, remaining // 20)):
                if total_calls >= budget or time.time() > deadline - 5:
                    break
                x_div = rng.uniform(-3.0, 3.0, dim)
                f_div = safe_call(x_div)
                if f_div is None:
                    break
                # Short Nelder-Mead from this point
                short_nm = min(remaining - (total_calls - (budget - remaining)), 30)
                if short_nm > 5:
                    nelder_mead(x_div.copy(), f_div, short_nm, initial_step=1.0)

        # Final refinement if budget remains
        remaining = budget - total_calls
        if remaining > 5:
            nelder_mead(best_x.copy(), best_f, remaining, initial_step=0.3)

        return best_x, best_f
