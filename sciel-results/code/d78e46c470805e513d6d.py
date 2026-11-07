"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard for the BBOB-style task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_run(self, func, budget, dim, rng, start_mean, start_sigma, deadline):
        """Run a single CMA-ES instance and return (best_x, best_f, evals_used)."""
        # CMA-ES Parameters
        # For dim=5, use a larger population for better exploration
        lambda_pop = 15
        if lambda_pop < 10:
            lambda_pop = 10

        mu = lambda_pop // 2
        # Weights: w_i = ln(mu + 0.5) - ln(i) for i in 1..mu
        weights = np.zeros(mu)
        for i in range(mu):
            weights[i] = np.log(mu + 0.5) - np.log(i + 1)
        weights /= np.sum(weights)

        mu_eff = 1.0 / np.sum(weights**2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1. / mu_eff) / ((dim + 2)**2 + mu_eff))

        # Initialization
        mean = start_mean.copy()
        sigma = start_sigma
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Precompute Cholesky of C (initially Identity)
        C_sqrt = np.eye(dim)

        best_x = mean.copy()
        best_f = np.inf

        evals = 0

        # Helper to check time
        def time_exceeded():
            return time.time() > deadline

        # Main Loop
        while evals < budget and not time_exceeded():
            # Generate population
            pop = np.zeros((lambda_pop, dim))
            for i in range(lambda_pop):
                if time_exceeded():
                    break
                # Sample from Normal(mean, sigma^2 C)
                # y ~ N(0, I), x = mean + sigma * C_sqrt @ y
                y = rng.standard_normal(dim)
                z = C_sqrt @ y
                pop[i] = mean + sigma * z
                # Clip to box [-5, 5]
                np.clip(pop[i], -5, 5, out=pop[i])
                evals += 1

            if time_exceeded():
                break

            # Evaluate fitness
            fitness = np.zeros(lambda_pop)
            for i in range(lambda_pop):
                if evals > budget:
                    break
                try:
                    f_val = func(pop[i])
                except Exception:
                    f_val = np.inf
                fitness[i] = f_val
                if f_val < best_f:
                    best_f = f_val
                    best_x = pop[i].copy()

            if time_exceeded():
                break

            # Sort population by fitness
            idx = np.argsort(fitness)
            pop = pop[idx]
            fitness = fitness[idx]

            # Update Mean
            # w is defined for the first mu individuals
            mean_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)

            # Calculate step for CSA
            # pc = (1 - c_c) * pc + sqrt(c_c * (2 - c_c) * mu_eff) * C^-1/2 * (mean_new - mean) / sigma
            # We need C^-1/2. We have C_sqrt = C^1/2. So C^-1/2 = inv(C_sqrt).
            # To avoid explicit inverse, we can solve C_sqrt @ v = (mean_new - mean) / sigma
            delta_mean = (mean_new - mean) / sigma
            try:
                v = np.linalg.solve(C_sqrt, delta_mean)
            except np.linalg.LinAlgError:
                v = delta_mean  # Fallback

            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * v

            # Update sigma
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^-1/2 * (mean_new - mean) / sigma
            # Note: The standard CMA-ES update for ps uses the same v as above?
            # Yes, ps is the evolution path in the search space (transformed by C^-1/2).
            # Actually, standard formula:
            # ps = (1 - c_sigma) * ps + sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C^-1/2 * (x_new - x_old) / sigma
            # Here (x_new - x_old) is (mean_new - mean).
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v

            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

            # Update Covariance Matrix
            # C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1-c_c)^2) * np.eye(dim)) 
            #    + c_mu * sum(w_i * (y_i @ y_i.T)) where y_i = (x_i - mean_old)/sigma
            # Note: The rank-mu update uses the selected individuals.
            # Let's compute the rank-1 part and rank-mu part.

            # Rank-1 part
            # The term (1 - (1-c_c)^2) * I is added to pc @ pc.T in the standard formula?
            # Formula: C = (1 - c_1 - c_mu) * C + c_1 * (pc @ pc.T + (1 - (1-c_c)^2) * I) + c_mu * H
            # where H = sum_{i=1}^mu w_i * y_i @ y_i.T

            # Calculate H
            H = np.zeros((dim, dim))
            for i in range(mu):
                # y_i = (pop[i] - mean) / sigma  (Using old mean)
                y_i = (pop[i] - mean) / sigma
                H += weights[i] * np.outer(y_i, y_i)

            # Update C
            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * np.eye(dim))
            C += c_mu * H

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2

            # Update Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # If not PD, add small regularization
                C += 1e-12 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    break  # Give up if matrix becomes unstable

            # Update mean
            mean = mean_new

            # Termination check: if sigma is very small or no improvement in recent iterations
            if sigma < 1e-12:
                break

        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin: 300s limit, stop at 285s

        dim = self.dim
        budget = self.budget

        best_x_global = None
        best_f_global = np.inf

        rng = np.random.default_rng(0)

        # Main CMA-ES run with most of the budget
        # Reserve ~50 evals for potential restart/polish
        main_budget = budget - 50
        if main_budget < 100:
            main_budget = budget

        # First: a quick random search to get a decent initial point
        n_init = min(20, main_budget)
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            try:
                f = func(x)
            except Exception:
                f = np.inf
            if f < best_f_global:
                best_f_global = f
                best_x_global = x

        if best_x_global is None:
            best_x_global = np.zeros(dim)
            try:
                best_f_global = func(best_x_global)
            except Exception:
                best_f_global = np.inf

        # Main CMA-ES run starting from the best point found
        start_mean = best_x_global.copy()
        start_sigma = 2.0

        try:
            best_x, best_f, evals_used = self._cma_es_run(
                func, main_budget, dim, rng, start_mean, start_sigma, deadline
            )
        except Exception:
            best_x, best_f, evals_used = best_x_global, best_f_global, 0

        if best_f < best_f_global:
            best_f_global = best_f
            best_x_global = best_x

        # If we have remaining budget, do a short local polish with Nelder-Mead
        remaining = budget - n_init - evals_used
        if remaining > 10 and best_x_global is not None and time.time() < deadline - 5:
            try:
                from scipy.optimize import minimize
                # Use Nelder-Mead for local refinement
                res = minimize(
                    func,
                    best_x_global,
                    method='Nelder-Mead',
                    options={'maxiter': remaining, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < best_f_global:
                    best_f_global = res.fun
                    best_x_global = res.x
            except Exception:
                pass

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
