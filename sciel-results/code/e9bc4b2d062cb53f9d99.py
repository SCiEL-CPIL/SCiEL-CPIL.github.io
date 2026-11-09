"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES for Katsuura-style multimodal function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 300  # safety margin before 330s kill

        # CMA-ES parameters
        x = np.zeros(n)  # start at origin (minimizer is in [-4,4]^5)
        sigma = 4.0  # large initial sigma for global exploration
        lambda_ = 4 + int(3 * np.log(n))  # population size
        mu = lambda_ // 2
        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / weights.sum()
        # Step size control
        c_sigma = (mueff + 2) / (n + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + c_sigma
        hsig_threshold = 1.4 + 2 / (n + 1)
        # Covariance matrix adaptation
        c_c = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        # Memory
        C = np.eye(n)
        pc = np.zeros(n)
        ps = np.zeros(n)
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        hsig = 0.0
        count_steps = 0
        best_x = x.copy()
        best_f = func(x)
        calls = 1

        for gen in range(1, budget):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            B = np.linalg.cholesky(C)
            offspring = []
            for i in range(lambda_):
                z = np.random.randn(n)
                y = x + sigma * B @ z
                y = np.clip(y, -5, 5)
                offspring.append((y, None))

            # Evaluate offspring
            for i in range(lambda_):
                if calls >= budget:
                    break
                fy = func(offspring[i][0])
                offspring[i] = (offspring[i][0], fy)
                calls += 1
                if fy < best_f:
                    best_f = fy
                    best_x = offspring[i][0].copy()

            if calls >= budget:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[1] if t[1] is not None else float('inf'))
            selected = [offspring[i][0] for i in range(mu)]
            selected_f = [offspring[i][1] for i in range(mu)]

            # Update mean
            m_old = x.copy()
            x_new = sum(weights[i] * selected[i] for i in range(mu))

            # Update evolution paths
            if np.linalg.norm(x_new - m_old) > 0:
                z = (x_new - m_old) / (sigma * np.linalg.norm(x_new - m_old))
                # Transform z
                C_inv_B = np.linalg.inv(B)
                z_transformed = C_inv_B @ z
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * z_transformed
                hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (calls - count_steps)) / chi_n < hsig_threshold
                if hsig:
                    pc = (1 - c_c) * pc + np.sqrt(c_c * (2 - c_c) * mueff) * (x_new - m_old) / sigma
                else:
                    pc = (1 - c_c) * pc
            else:
                hsig = True

            # Update covariance
            delta_hsig = (1.0 if hsig else 0.0) - (1 - c_sigma)
            rank_one = np.outer(ps, ps)
            rank_mu = sum(weights[i] * np.outer(selected[i] - m_old, selected[i] - m_old) for i in range(mu))
            C = (1 + c_c * delta_hsig - c_1 - c_mu) * C + c_1 * (np.outer(pc, pc) + (1 if not hsig else 0) * np.eye(n)) + c_mu * rank_mu

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x = x_new
            count_steps = calls

        return best_x, best_f
# EVOLVE-BLOCK-END
