"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES with adaptive population size and boundary handling."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES parameters
        # Increased population size for better exploration
        lam = 12
        mu = lam // 2  # number of parents

        # Step size: Start larger to find the shift faster
        sigma = 2.0

        # Covariance matrix
        C = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Adaptation parameters
        c_sigma = (mu + 0.5) / (dim + 2)
        d_sigma = 1 + 2 * max(0, (np.sqrt((mu + 1) / (dim + 1)) - 1) / (dim + 2)) + c_sigma
        c_c = (4 + mu / dim) / (dim + 4 + 2 * mu / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu)
        c_mu = min(1 - c_1, 2 * (mu - 2 + 1. / mu) / ((dim + 2) ** 2 + mu))

        # Weights
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Track calls
        calls = 1
        age = 0
        no_improve_count = 0
        prev_best_fx = fx

        while calls < budget:
            # Generate offspring
            offspring = []
            
            # Cholesky factorization
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)
            
            for i in range(lam):
                z = rng.standard_normal(dim)
                y = x + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                
                fy = func(y)
                calls += 1
                
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                
                offspring.append((fy, y))
                
                if calls >= budget:
                    break

            if len(offspring) < mu:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[0])
            
            # Selection
            selected = [y for _, y in offspring[:mu]]
            
            # Weighted mean
            m = np.zeros(dim)
            for i in range(mu):
                m += weights[i] * selected[i]
            
            # Update evolution paths
            diff = (m - x) / sigma
            
            # Update p_sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
            
            # Calculate h_sigma
            age += 1
            norm_p_sigma = np.linalg.norm(p_sigma)
            expected_norm = np.sqrt(1 - (1 - c_sigma) ** (2 * age))
            threshold = (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff + 1) / (dim + 1)) - 1
            
            if expected_norm > 1e-10:
                h_sigma = 1.0 if (norm_p_sigma / expected_norm) < threshold else 0.0
            else:
                h_sigma = 1.0

            # Update p_c
            p_c = (1 - c_c) * p_c + h_sigma * c_c * diff
            
            # Update C
            outer_sum = np.zeros((dim, dim))
            for i in range(mu):
                diff_y = selected[i] - m
                outer_sum += weights[i] * np.outer(diff_y, diff_y)
            
            C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) + c_mu * outer_sum
            
            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (norm_p_sigma / np.sqrt(mu_eff) - 1))
            sigma = max(sigma, 1e-10)
            sigma = min(sigma, 10.0)
            
            # Update mean
            x = m
            x = np.clip(x, -5, 5)
            
            # Stagnation check
            if best_fx < prev_best_fx - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            prev_best_fx = best_fx
            
            # If stuck for a long time and sigma is small, restart
            if no_improve_count > 30 and sigma < 0.5:
                # Restart from best_x with new random direction
                x = best_x.copy()
                sigma = 1.0
                C = np.eye(dim)
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)
                age = 0
                no_improve_count = 0
                # Slight perturbation
                x = x + 0.1 * rng.standard_normal(dim)
                x = np.clip(x, -5, 5)

        return best_x, best_fx
# EVOLVE-BLOCK-END
