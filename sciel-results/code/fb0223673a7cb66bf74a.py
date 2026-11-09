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
        deadline = start_time + 285  # safety margin

        rng = np.random.default_rng(42)
        evals_used = 0
        best_x = np.zeros(dim)
        best_f = float('inf')

        def evaluate(x):
            nonlocal evals_used, best_x, best_f
            if evals_used >= budget or time.time() > deadline:
                return best_f
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            f = float(func(x))
            evals_used += 1
            if f < best_f:
                best_f = f
                best_x = x.copy()
            return f

        # Phase 1: CMA-ES for global exploration
        # Use ~60% of budget for CMA-ES
        cma_budget = int(budget * 0.6)
        cma_budget = max(cma_budget, 50)

        # CMA-ES parameters
        lambda_ = 10
        mu = lambda_ // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights**2)

        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (weights[0] - 0.5) / ((dim + 2)**2 + mu_eff))

        x_mean = rng.uniform(-4, 4, dim)
        sigma = 1.5
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        try:
            C_chol = np.linalg.cholesky(C)
        except np.linalg.LinAlgError:
            C_chol = np.eye(dim)

        x_mean_prev = x_mean.copy()
        stagnation_counter = 0
        best_f_cma = float('inf')
        max_stagnation = 30  # restart if no improvement for 30 generations

        while evals_used + lambda_ <= cma_budget and time.time() < deadline:
            # Generate population
            ys = rng.standard_normal((lambda_, dim))
            xs = np.array([x_mean + sigma * (C_chol @ y) for y in ys])

            # Evaluate
            fits = np.array([evaluate(x) for x in xs])

            # Check for NaN/inf
            if np.any(~np.isfinite(fits)):
                fits = np.where(np.isfinite(fits), fits, 1e10)

            # Update best
            idx_min = np.argmin(fits)
            if fits[idx_min] < best_f_cma:
                best_f_cma = fits[idx_min]
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Sort
            order = np.argsort(fits)
            xs_sorted = xs[order]
            fits_sorted = fits[order]

            # Update mean
            x_mean_new = np.sum(weights[:, None] * xs_sorted[:mu], axis=0)

            # Update evolution paths
            y_mean = (x_mean_new - x_mean_prev) / sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_mean
            norm_ps = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals_used // lambda_)))
            hsig = 1 if norm_ps < (1.4 + 2 / (dim + 1)) * np.sqrt((1 - (1 - c_sigma)) / (1 - (1 - c_sigma) ** (2 * (evals_used // lambda_)))) else 0

            pc = (1 - c_c) * pc + hsig * c_c * y_mean

            # Update covariance matrix
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            outer = np.outer(y_mean, y_mean)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum(
                weights[:mu, None] * np.outer(ys_sorted[:mu] if hasattr(ys_sorted, '__getitem__') else [ys[o] for o in order[:mu]], [ys_sorted[:mu] if hasattr(ys_sorted, '__getitem__') else [ys[o] for o in order[:mu]]]), axis=0
            )

            # Actually, let me fix the covariance update properly
            # Recompute with proper sorted ys
            ys_sorted = np.array([ys[o] for o in order[:mu]])
            delta_hsig = (1 - hsig) * c_c * (2 - c_c)
            C = (1 + c_1 * delta_hsig - c_1 - c_mu) * C + c_1 * np.outer(pc, pc) + c_mu * np.sum(
                weights[:mu, None] * np.outer(ys_sorted, ys_sorted), axis=0
            )

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_ps / np.sqrt(1 - (1 - c_sigma) ** (2 * (evals_used // lambda_))) - 1))
            sigma = np.clip(sigma, 1e-10, 100.0)

            # Ensure C is symmetric positive definite
            C = (C + C.T) / 2
            try:
                C_chol = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            except np.linalg.LinAlgError:
                C_chol = np.eye(dim)

            x_mean_prev = x_mean
            x_mean = x_mean_new

            # Check for stagnation - restart with smaller sigma
            if stagnation_counter >= max_stagnation:
                # Restart around best point with reduced sigma
                x_mean = best_x.copy()
                sigma *= 0.5
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                C_chol = np.eye(dim)
                x_mean_prev = x_mean.copy()
                stagnation_counter = 0

        # Phase 2: Nelder-Mead refinement from best point
        nm_budget = budget - evals_used
        if nm_budget < 10:
            nm_budget = 10

        # Nelder-Mead simplex method
        def nelder_mead(func_eval, x0, dim, budget_nm, rng):
            """Nelder-Mead simplex optimization"""
            nonlocal evals_used, best_x, best_f

            # Initialize simplex
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i - 1] = x0[i - 1] + 0.1
                if simplex[i][i - 1] > 5.0:
                    simplex[i][i - 1] = x0[i - 1] - 0.1

            # Evaluate initial simplex
            simplex_f = np.array([func_eval(simplex[i]) for i in range(dim + 1)])

            alpha = 1.0   # reflection
            gamma = 2.0   # expansion
            rho = 0.5     # contraction
            sigma_nm = 0.5  # shrink

            max_iters = budget_nm // 2  # each iteration uses ~2 evaluations on average
            max_iters = max(max_iters, 5)

            for it in range(max_iters):
                if evals_used >= budget_nm or time.time() > deadline:
                    break

                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]

                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)

                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref = np.clip(x_ref, -5.0, 5.0)
                f_ref = func_eval(x_ref)

                if f_ref < simplex_f[0]:
                    # Expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp = np.clip(x_exp, -5.0, 5.0)
                    f_exp = func_eval(x_exp)
                    if f_exp < f_ref:
                        simplex[-1] = x_exp
                        simplex_f[-1] = f_exp
                    else:
                        simplex[-1] = x_ref
                        simplex_f[-1] = f_ref
                elif f_ref < simplex_f[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref
                    simplex_f[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < simplex_f[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    x_con = np.clip(x_con, -5.0, 5.0)
                    f_con = func_eval(x_con)

                    if f_con < min(f_ref, simplex_f[-1]):
                        simplex[-1] = x_con
                        simplex_f[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5.0, 5.0)
                            simplex_f[i] = func_eval(simplex[i])

                # Check convergence
                if np.max(simplex_f) - np.min(simplex_f) < 1e-10:
                    break

            # Return best point from simplex
            idx_best = np.argmin(simplex_f)
            return simplex[idx_best], simplex_f[idx_best]

        # Run NM from best point
        if evals_used < budget and time.time() < deadline:
            nm_x, nm_f = nelder_mead(evaluate, best_x, dim, nm_budget, rng)
            if nm_f < best_f:
                best_f = nm_f
                best_x = nm_x.copy()

        # Phase 3: Local random search around best point for final refinement
        remaining = budget - evals_used
        if remaining > 20 and time.time() < deadline:
            # Try small perturbations
            step_size = 0.05
            for _ in range(min(remaining, 50)):
                if evals_used >= budget or time.time() > deadline:
                    break
                x_try = best_x + rng.normal(0, step_size, dim)
                x_try = np.clip(x_try, -5.0, 5.0)
                f_try = evaluate(x_try)
                if f_try >= best_f:
                    step_size *= 0.7

        # Return best found
        return best_f
