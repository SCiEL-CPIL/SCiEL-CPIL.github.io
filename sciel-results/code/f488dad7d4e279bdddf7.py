"""CMA-ES optimizer for the BBOB-style Sphere function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES based optimizer for minimizing a black-box function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for the 300s limit

        d = self.dim
        budget = self.budget

        # Initialize CMA-ES parameters
        # Start from center of the box
        x_mean = np.zeros(d)
        sigma = 2.0  # Initial step size, box is [-5, 5]^5, so range is 10. 2.0 covers most of it.
        
        # CMA-ES weights
        # Standard weights: log-linear
        mu = max(1, d // 2)  # Number of parents
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        
        mu_eff = 1.0 / np.sum(weights**2)
        
        # Learning rates
        c_sigma = (mu_eff + 2) / (d + mu_eff + 5)
        c_c = (4 + mu_eff / d) / (d + 4 + 2 * mu_eff / d)
        c_1 = 2 / ((d + 1.3)**2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((d + 2)**2 + mu_eff))
        
        # Covariance matrix
        C = np.eye(d)
        
        # Evolution path
        p_sigma = np.zeros(d)
        p_c = np.zeros(d)
        
        # Pre-compute sqrt(C) for sampling
        # We will update this each iteration
        try:
            B, D = np.linalg.eigh(C)
            B = B @ np.diag(np.sqrt(np.abs(D)))
        except:
            B = np.eye(d)
            D = np.ones(d)

        best_x = x_mean.copy()
        best_f = func(best_x)
        calls = 1

        # Main loop
        # We need to manage the budget carefully.
        # Each iteration uses 'lambda' evaluations.
        # Let's choose lambda = 4 * d for good exploration.
        lambda_ = 4 * d
        
        # Check if we have enough budget for at least one iteration
        if budget < calls + lambda_:
             # Fallback to simple random search if budget is too small for CMA-ES
             for _ in range(budget - calls):
                 if time.time() > deadline:
                     break
                 x_trial = np.clip(rng.uniform(-5, 5, d), -5, 5)
                 f_trial = func(x_trial)
                 calls += 1
                 if f_trial < best_f:
                     best_x = x_trial
                     best_f = f_trial
             return best_x, best_f

        # Use a fixed seed for reproducibility within the instance, 
        # but the instance itself is fixed.
        rng = np.random.default_rng(0)

        while calls + lambda_ <= budget:
            if time.time() > deadline:
                break

            # 1. Sample lambda_ points
            x_pop = np.zeros((lambda_, d))
            for i in range(lambda_):
                z = rng.standard_normal(d)
                x_pop[i] = x_mean + sigma * (B @ z)
                # Clip to box
                x_pop[i] = np.clip(x_pop[i], -5, 5)

            # 2. Evaluate
            f_pop = np.zeros(lambda_)
            for i in range(lambda_):
                f_pop[i] = func(x_pop[i])
                calls += 1
                if time.time() > deadline:
                    # If we run out of time mid-evaluation, we still have the calls made.
                    # We should break out and return best found.
                    # But we need to ensure we don't exceed budget.
                    # The loop condition checks budget, but time might kill us first.
                    pass

            if time.time() > deadline:
                break

            # 3. Sort by fitness
            order = np.argsort(f_pop)
            x_sorted = x_pop[order]
            f_sorted = f_pop[order]

            # Update best
            if f_sorted[0] < best_f:
                best_f = f_sorted[0]
                best_x = x_sorted[0].copy()

            # 4. Update mean
            x_mean = np.sum(weights[:, None] * x_sorted[:mu], axis=0)

            # 5. Update sigma and C
            # Standard deviation of the selected points
            y_mean = np.sum(weights[:, None] * (x_sorted[:mu] - x_mean), axis=0)
            
            # Evolution path for sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * y_mean
            
            # Norm of p_sigma
            norm_p_sigma = np.linalg.norm(p_sigma)
            
            # Correction term for bias
            chi_n = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d**2))
            
            # Update sigma
            sigma *= np.exp(c_sigma / chi_n * (norm_p_sigma / chi_n - 1))
            
            # Prevent sigma from becoming too small or too large
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution path for C
            y_c = (x_sorted[:mu] - x_mean) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_mean / sigma
            
            # Update C
            # Hessian update
            C = (1 - c_1 - c_mu) * C + c_1 * (2 * np.outer(p_c, p_c) / c_c - 1) + c_mu * np.sum(weights[:, None, None] * np.outer(y_c, y_c), axis=0)
            
            # Symmetrize
            C = (C + C.T) / 2

            # Update B and D (eigendecomposition)
            try:
                B, D = np.linalg.eigh(C)
                # Ensure D is positive
                D = np.sqrt(np.abs(D))
                B = B @ np.diag(D)
            except:
                # If eigendecomposition fails, reset to identity
                C = np.eye(d)
                B = np.eye(d)
                D = np.ones(d)

        # If we have remaining budget, do a local polish
        remaining = budget - calls
        if remaining > 0:
            # Simple local search with decreasing step size
            sigma_polish = 0.1
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(best_x + sigma_polish * rng.standard_normal(d), -5, 5)
                fy = func(y)
                calls += 1
                if fy < best_f:
                    best_x = y
                    best_f = fy
                    sigma_polish *= 1.05
                else:
                    sigma_polish *= 0.9
                    if sigma_polish < 1e-6:
                        sigma_polish = 1e-6

        return best_x, best_f
# EVOLVE-BLOCK-END
