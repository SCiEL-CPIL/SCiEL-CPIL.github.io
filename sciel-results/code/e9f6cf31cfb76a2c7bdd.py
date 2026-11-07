"""Rewritten optimizer using CMA-ES for the Rosenbrock function."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer with adaptive restarts and boundary handling for the Rosenbrock function on [-5, 5]^5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # ~10 for dim=5
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def evaluate(x):
            nonlocal calls, best_x, best_f
            # Apply penalty for out-of-bounds
            penalty = 0.0
            for xi in x:
                if xi < -5.0:
                    penalty += 100.0 * (-5.0 - xi) ** 2
                elif xi > 5.0:
                    penalty += 100.0 * (xi - 5.0) ** 2
            
            x_clipped = np.clip(x, -5.0, 5.0)
            f = func(x_clipped)
            f += penalty
            calls += 1
            if f < best_f:
                best_f = f
                best_x = x_clipped.copy()
            return f

        # Initial evaluation at center
        mean = np.zeros(dim)
        f0 = evaluate(mean)
        best_x = mean.copy()
        best_f = f0

        # CMA-ES state
        sigma = 2.0
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        
        stagnation_counter = 0
        max_stagnation = 20
        prev_best_f = best_f

        while calls < budget:
            old_mean = mean.copy()
            old_C = C.copy()

            # Determine how many we can evaluate in this generation
            remaining = budget - calls
            current_lambda = min(lambda_, remaining)
            
            if current_lambda <= 0:
                break

            # Pre-compute Cholesky factor if C is valid
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                L = eigvecs @ np.diag(np.sqrt(eigvals))

            # Generate population
            Z = rng.standard_normal((current_lambda, dim))
            Pop = mean + sigma * (Z @ L.T)
            
            # Evaluate population
            Pop_F = np.array([evaluate(x) for x in Pop])

            if calls >= budget:
                break

            # Sort by fitness
            sorted_idx = np.argsort(Pop_F)
            Pop = Pop[sorted_idx]
            Pop_F = Pop_F[sorted_idx]

            # Update mean using weighted recombination
            new_mean = np.sum(weights[:min(mu, len(Pop))] * Pop[:min(mu, len(Pop))], axis=0)
            mean = new_mean

            # Check for stagnation
            if best_f < prev_best_f - 1e-8:
                prev_best_f = best_f
                stagnation_counter = 0
            else:
                stagnation_counter += 1

            # Adaptive Restart
            if stagnation_counter >= max_stagnation:
                # Restart CMA-ES state centered at best_x
                mean = best_x.copy()
                sigma = max(0.1, sigma * 0.5) # Reduce sigma
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                C = np.eye(dim)
                stagnation_counter = 0
                prev_best_f = best_f
                continue

            # Update evolution paths
            # pc update
            pc = (1 - c_c) * pc + c_c * ((mean - old_mean) / sigma)
            
            # ps update
            try:
                inv_L = np.linalg.inv(L)
                diff_whitened = inv_L @ ((mean - old_mean) / sigma)
            except np.linalg.LinAlgError:
                diff_whitened = (mean - old_mean) / sigma

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff_whitened

            # Update step size (sigma)
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))

            # Update covariance matrix
            C_new = (1 - c_1 - c_mu) * old_C
            C_new += c_1 * (np.outer(pc, pc) + (1 - c_c) * np.eye(dim))
            
            for i in range(min(mu, len(Pop))):
                diff_i = (Pop[i] - old_mean) / sigma
                C_new += c_mu * weights[i] * np.outer(diff_i, diff_i)

            # Ensure C is symmetric positive definite
            C_new = (C_new + C_new.T) / 2
            C_new += 1e-12 * np.eye(dim)
            C = C_new

        return best_x, best_f
# EVOLVE-BLOCK-END
