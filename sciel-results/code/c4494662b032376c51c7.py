"""CMA-ES optimizer for BBOB f6 (Attractive Sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES: covariance matrix adaptation evolution strategy."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        dim = self.dim
        budget = self.budget
        calls = 0

        def check_deadline():
            return time.time() >= deadline

        rng = np.random.default_rng(42)

        # Initial point: try origin and a few probes to find a good starting region
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls += 1
        best_x = x0.copy()
        best_f = f0

        # Quick probe to find a better starting point
        for probe in [np.ones(dim) * 1.0, np.ones(dim) * -1.0,
                      np.array([2.0, 0.0, 0.0, 0.0, 0.0]),
                      np.array([0.0, 2.0, 0.0, 0.0, 0.0]),
                      np.array([0.0, 0.0, 2.0, 0.0, 0.0]),
                      np.array([0.0, 0.0, 0.0, 2.0, 0.0]),
                      np.array([0.0, 0.0, 0.0, 0.0, 2.0])]:
            if calls >= budget or check_deadline():
                break
            fp = func(probe)
            calls += 1
            if fp < best_f:
                best_x = probe.copy()
                best_f = fp

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(dim))
        lam = max(4, int(4 + 3 * np.log(max(dim, 2))))
        # Limit to available budget
        max_iter = max(1, (budget - calls) // lam)
        
        # CMA-ES state
        x_mean = best_x.copy()
        sigma = 1.0  # initial step size
        C = np.eye(dim)  # covariance matrix
        C_inv = np.eye(dim)  # inverse covariance
        # Precompute Cholesky-like factor for sampling
        try:
            C_chol = np.linalg.cholesky(C)
        except:
            C_chol = np.eye(dim)
        
        # Update rates
        mu = lam // 2  # number of selected individuals
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights = weights / weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        
        # Path accumulators
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Damping for sigma update
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        best_x = x_mean.copy()
        best_f = best_f

        for iteration in range(max_iter):
            if check_deadline() or calls >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lam):
                if check_deadline() or calls >= budget:
                    break
                # Sample from N(x_mean, sigma^2 * C)
                z = rng.standard_normal(dim)
                try:
                    y = x_mean + sigma * (C_chol @ z)
                except:
                    y = x_mean + sigma * z
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                offspring.append((y, fy))

            if len(offspring) < mu:
                break

            # Sort by fitness
            offspring.sort(key=lambda x: x[1])
            selected = offspring[:mu]

            # Weighted mean of selected
            x_new_mean = sum(weights[i] * selected[i][0] for i in range(mu))
            x_new_mean = np.array(x_new_mean)

            # Update paths
            try:
                C_inv_sqrt = np.linalg.cholesky(C_inv)
                C_inv_sqrt_T = C_inv_sqrt.T
            except:
                C_inv_sqrt_T = np.eye(dim)

            diff = (x_new_mean - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv_sqrt_T @ diff)
            
            p_sigma_norm = np.linalg.norm(p_sigma)
            h_sigma = 1.0 if p_sigma_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0.0
            
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

            # Update covariance
            delta_h = (1.8 + 2 / (mu_eff + 1)) * (1 - h_sigma)
            rank_one = np.outer(p_c + delta_h * p_c, p_c + delta_h * p_c)
            
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                yi = selected[i][0] - x_mean
                rank_mu += weights[i] * np.outer(yi, yi) / (sigma ** 2)

            C = (1 + c_1 * (rank_one - C) + c_mu * (rank_mu - C)) * C
            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (p_sigma_norm / chi_n - 1))
            
            # Ensure sigma stays reasonable
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Update C_inv and C_chol
            x_mean = x_new_mean
            try:
                C_inv = np.linalg.inv(C)
                C_chol = np.linalg.cholesky(C)
            except:
                # Reset if numerical issues
                C = np.eye(dim)
                C_inv = np.eye(dim)
                C_chol = np.eye(dim)
                sigma = 1.0

            # Track best
            best_in_gen = min(selected, key=lambda x: x[1])
            if best_in_gen[1] < best_f:
                best_f = best_in_gen[1]
                best_x = best_in_gen[0].copy()

        # Final local polish with (1+1)-ES around best found
        if not check_deadline() and calls < budget:
            x = best_x.copy()
            fx = best_f
            sigma_local = max(0.1, sigma * 0.5)
            successes = 0
            total = 0
            for _ in range(min(50, budget - calls)):
                if check_deadline() or calls >= budget:
                    break
                y = x + sigma_local * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = func(y)
                calls += 1
                if fy < fx:
                    x = y
                    fx = fy
                    successes += 1
                total += 1
                if total >= 5:
                    sr = successes / total
                    if sr > 0.2:
                        sigma_local *= 1.2
                    elif sr < 0.2:
                        sigma_local *= 0.8
                    successes = 0
                    total = 0
            if fx < best_f:
                best_x = x
                best_f = fx

        return best_x, best_f
# EVOLVE-BLOCK-END
