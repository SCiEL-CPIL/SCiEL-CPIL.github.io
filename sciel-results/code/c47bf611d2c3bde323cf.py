"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    CMA-ES (Covariance Matrix Adaptation Evolution Strategy) with
    restarts and a final Nelder-Mead polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        start_time = time.time()
        deadline = start_time + 300.0  # 300s limit, use 320s for safety

        # Global best
        best_x = None
        best_f = np.inf

        # Wrap func to count calls and enforce deadline
        calls_made = [0]

        def counted_func(x):
            calls_made[0] += 1
            # Check deadline
            if time.time() > deadline:
                raise TimeoutError("Deadline exceeded")
            return func(x)

        # CMA-ES parameters
        # We'll do multiple restarts with decreasing budgets
        # For dim=5, budget=500, we can do about 5-10 restarts

        lambda_ = 4 * dim  # population size ~ 4*dim = 20
        mu = lambda_ // 2  # number of parents

        # We'll do several restarts, each with a portion of the budget
        # First restart gets more budget, subsequent ones get less

        # Strategy: 
        # - Do one main CMA-ES run with most of the budget
        # - Then do a few short restarts from different initializations
        # - Finally, do a Nelder-Mead polish

        # Main CMA-ES run
        main_budget = int(budget * 0.6)  # 60% of budget for main run
        restart_budgets = [int(budget * 0.1)] * 3  # 3 restarts of 10% each
        # Remaining ~10% for Nelder-Mead polish

        def cma_es(x0, sigma0, max_evals, func_wrap):
            """Run CMA-ES from x0 with sigma0 for at most max_evals evaluations."""
            dim = self.dim
            lambda_ = 4 * dim
            mu = lambda_ // 2

            # Weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Step size control
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma

            # Mean update
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))

            # Initial values
            x_mean = np.array(x0, dtype=float)
            sigma = sigma0
            C = np.eye(dim)
            C_inv = np.eye(dim)
            pc = np.zeros(dim)
            ps = np.zeros(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            # Track evaluations used
            evals = 0

            # Initial population
            best_x = x_mean.copy()
            best_f = func_wrap(x_mean)
            evals += 1

            while evals < max_evals:
                # Generate population
                # Use Cholesky decomposition of C
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback to identity if Cholesky fails
                    C = np.eye(dim)
                    L = np.eye(dim)

                # Generate lambda_ samples
                zs = np.random.default_rng(int(time.time() * 1000) + evals).standard_normal((lambda_, dim))
                ys = np.array([sigma * (L @ z) for z in zs])

                # Evaluate population
                fs = np.array([func_wrap(x_mean + y) for y in ys])
                evals += lambda_

                # Update best
                idx_best = np.argmin(fs)
                if fs[idx_best] < best_f:
                    best_f = fs[idx_best]
                    best_x = (x_mean + ys[idx_best]).copy()

                # Sort by fitness
                order = np.argsort(fs)
                y_sorted = ys[order]
                f_sorted = fs[order]

                # Update mean
                x_mean_new = x_mean + sigma * np.sum(weights[:, None] * y_sorted[:mu], axis=0)

                # Update path
                C_sqrt_inv = np.linalg.inv(L)
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_sqrt_inv @ (x_mean_new - x_mean) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals + 1))) < (1.4 + 2 / (dim + 1)) * chi_n)

                pc = (1 - c_c) * pc + hsig * np.sqrt(c_c * (2 - c_c) * mu_eff) * (x_mean_new - x_mean) / sigma

                # Update covariance
                delta_hsig = (1 - hsig) * c_c * (2 - c_c)
                rank_one = np.outer(pc, pc)
                rank_mu = np.sum(weights[:, None, None] * np.outer(y_sorted[:mu], y_sorted[:mu]), axis=0)
                C = (1 + c_sigma * (c_sigma - d_sigma) + delta_hsig) * C + c_1 * (rank_one - C) + c_mu * (rank_mu - C)

                # Ensure symmetry
                C = (C + C.T) / 2

                x_mean = x_mean_new

                # Update sigma
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

                # Clip to bounds
                x_mean = np.clip(x_mean, -5, 5)

                # Safety: if sigma gets too small, break
                if sigma < 1e-10:
                    break

            return best_x, best_f

        # Nelder-Mead polish
        def nelder_mead(x0, max_evals, func_wrap):
            """Simple Nelder-Mead implementation."""
            dim = self.dim
            # Initial simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += 0.5 if abs(xi[i]) < 4.5 else -0.5
                simplex.append(xi)

            fs = [func_wrap(s) for s in simplex]
            evals = dim + 1

            alpha = 1.0  # reflection
            gamma = 2.0  # expansion
            rho = 0.5    # contraction
            sigma_nm = 0.5  # shrink

            while evals < max_evals:
                # Sort
                order = np.argsort(fs)
                simplex = [simplex[i] for i in order]
                fs = [fs[i] for i in order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func_wrap(x_r)
                evals += 1

                if f_r < fs[0]:
                    # Expansion
                    x_e = centroid + gamma * (centroid - simplex[-1])
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func_wrap(x_e)
                    evals += 1
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fs[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fs[-1] = f_r
                elif f_r < fs[-2]:
                    simplex[-1] = x_r
                    fs[-1] = f_r
                else:
                    # Contraction
                    if f_r < fs[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = func_wrap(x_c)
                    else:
                        x_c = centroid - rho * (centroid - simplex[-1])
                        f_c = func_wrap(x_c)
                    evals += 1
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func_wrap(x_c) if f_c is None else f_c
                    # Actually we already computed f_c above, but let's be safe
                    if evals >= max_evals:
                        break
                    if f_c < min(f_r, fs[-1]):
                        simplex[-1] = x_c
                        fs[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            fs[i] = func_wrap(simplex[i])
                            evals += 1
                            if evals >= max_evals:
                                break

                if evals >= max_evals:
                    break

            # Return best
            idx_best = np.argmin(fs)
            return simplex[idx_best], fs[idx_best]

        # Main optimization loop
        rng = np.random.default_rng(42)

        # First, do a quick random search to find a good starting point
        # Use 10% of budget for random search
        random_budget = int(budget * 0.05)
        for i in range(random_budget):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = counted_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Main CMA-ES run
        if calls_made[0] < budget:
            try:
                x0 = best_x if best_x is not None else rng.uniform(-5, 5, dim)
                sigma0 = 2.0
                x_res, f_res = cma_es(x0, sigma0, main_budget, counted_func)
                if f_res < best_f:
                    best_f = f_res
                    best_x = x_res
            except (TimeoutError, Exception):
                pass

        # Restart CMA-ES from a few different points
        for i in range(3):
            if calls_made[0] >= budget:
                break
            if time.time() > deadline:
                break
            try:
                x0 = rng.uniform(-4, 4, dim)
                sigma0 = 1.5
                x_res, f_res = cma_es(x0, sigma0, restart_budgets[i], counted_func)
                if f_res < best_f:
                    best_f = f_res
                    best_x = x_res
            except (TimeoutError, Exception):
                pass

        # Final Nelder-Mead polish
        if calls_made[0] < budget and best_x is not None:
            if time.time() > deadline:
                pass
            else:
                try:
                    nm_budget = min(budget - calls_made[0], int(budget * 0.15))
                    if nm_budget > dim + 2:
                        x_res, f_res = nelder_mead(best_x, nm_budget, counted_func)
                        if f_res < best_f:
                            best_f = f_res
                            best_x = x_res
                except (TimeoutError, Exception):
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
