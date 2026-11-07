"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts and time guard for the BBOB-style task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cma_es_run(self, func, budget, dim, rng, start_mean, start_sigma, deadline, global_best_f=None):
        """Run a single CMA-ES instance and return (best_x, best_f, evals_used)."""
        # CMA-ES Parameters
        lambda_pop = 9 + 3 * int(np.floor(np.log10(4 * dim)))
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
        best_f = func(best_x)
        evals = 1

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
            mean_new = np.sum(weights[:, np.newaxis] * pop[:mu], axis=0)

            # Calculate step for CSA
            delta_mean = (mean_new - mean) / sigma
            try:
                v = np.linalg.solve(C_sqrt, delta_mean)
            except np.linalg.LinAlgError:
                v = delta_mean

            # Update pc
            pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mu_eff) * v

            # Update sigma
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * v
            sigma *= np.exp((c_sigma / d_sigma) * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

            # Update Covariance Matrix
            H = np.zeros((dim, dim))
            for i in range(mu):
                y_i = (pop[i] - mean) / sigma
                H += weights[i] * np.outer(y_i, y_i)

            C = (1 - c_1 - c_mu) * C
            C += c_1 * (np.outer(pc, pc) + (1 - (1 - c_c)**2) * np.eye(dim))
            C += c_mu * H

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2

            # Update Cholesky factor
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C += 1e-12 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    break

            # Update mean
            mean = mean_new

            # Termination check
            if sigma < 1e-10:
                break

        return best_x, best_f, evals

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin: 300s limit, stop at 290s

        dim = self.dim
        budget = self.budget

        # Strategy: 
        # 1. Start with a global exploration run (large sigma)
        # 2. Use remaining budget for fine-tuning around best point

        best_x_global = None
        best_f_global = np.inf

        rng = np.random.default_rng(42)

        # Phase 1: Global exploration - use most of the budget
        phase1_budget = int(budget * 0.7)
        start_mean = np.zeros(dim)
        start_sigma = 3.0

        try:
            best_x, best_f, evals_used = self._cma_es_run(
                func, phase1_budget, dim, rng, start_mean, start_sigma, deadline
            )
            best_x_global = best_x
            best_f_global = best_f
        except Exception:
            pass

        # Phase 2: Local refinement with remaining budget
        remaining_budget = budget - phase1_budget
        if remaining_budget > 10 and best_x_global is not None:
            start_mean = best_x_global.copy()
            start_sigma = 0.5
            
            try:
                best_x, best_f, evals_used = self._cma_es_run(
                    func, remaining_budget, dim, rng, start_mean, start_sigma, deadline
                )
                if best_f < best_f_global:
                    best_f_global = best_f
                    best_x_global = best_x
            except Exception:
                pass

        # If no global best was found, fallback
        if best_x_global is None:
            for _ in range(min(10, budget)):
                x = rng.uniform(-5, 5, dim)
                try:
                    f = func(x)
                except Exception:
                    f = np.inf
                if f < best_f_global:
                    best_f_global = f
                    best_x_global = x

        return best_x_global, best_f_global
# EVOLVE-BLOCK-END
