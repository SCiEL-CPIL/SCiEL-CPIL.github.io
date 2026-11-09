"""CMA-ES optimizer for BBOB f13 (Sharp ridge)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES implementation with time guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit

        # CMA-ES parameters
        n = self.dim
        pop_size = int(4 + 3 * np.log(n))  # Standard CMA-ES pop size
        if pop_size < 2:
            pop_size = 2

        # Initialization
        # Start near center, but slightly offset to avoid symmetry issues if any
        x_mean = np.zeros(n)
        sigma = 1.0  # Initial step size
        C = np.eye(n)
        C_sqrt_inv = np.eye(n)
        
        # Adaptation parameters
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= np.sum(weights)
        mu_eff = 1.0 / np.sum(weights ** 2)
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (n + 1)) - 1) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))
        chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

        # State
        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        eval_count = 0
        best_x = x_mean.copy()
        best_f = np.inf

        # Generate initial population
        def generate_population():
            nonlocal eval_count
            z = np.random.standard_normal((pop_size, n))
            # Transform using Cholesky of C
            # We maintain C_sqrt_inv, so y = sigma * (C_sqrt_inv @ z.T).T
            # Actually, standard CMA-ES uses y = sigma * C^{1/2} z.
            # Let's maintain C and use Cholesky decomposition each step or update inverse.
            # For simplicity and stability in 5D, we can use a simpler covariance update
            # or just use the standard algorithm carefully.
            
            # Let's stick to a robust implementation.
            # We need C^{1/2}. Let's compute Cholesky of C each generation? 
            # In 5D, this is cheap.
            try:
                L = np.linalg.cholesky(C)
                y = sigma * (L @ z.T).T
            except np.linalg.LinAlgError:
                # Fallback to diagonal if C becomes ill-conditioned
                diag = np.sqrt(np.diag(C))
                y = sigma * z * diag

            return y

        # Initial evaluation of mean
        f_mean = func(x_mean)
        eval_count += 1
        if f_mean < best_f:
            best_f = f_mean
            best_x = x_mean.copy()

        while eval_count < self.budget:
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            z = np.random.standard_normal((pop_size, n))
            try:
                L = np.linalg.cholesky(C)
                y = sigma * (L @ z.T).T
            except np.linalg.LinAlgError:
                diag = np.sqrt(np.diag(C))
                y = sigma * z * diag

            # Clip to box
            y = np.clip(y, -5, 5)

            # Evaluate
            f_values = np.array([func(y[i]) for i in range(pop_size)])
            eval_count += pop_size

            # Update best
            idx_best = np.argmin(f_values)
            if f_values[idx_best] < best_f:
                best_f = f_values[idx_best]
                best_x = y[idx_best].copy()

            # Sort by fitness
            idx_sorted = np.argsort(f_values)
            y_sorted = y[idx_sorted]
            f_sorted = f_values[idx_sorted]

            # Weighted recombination
            x_new = np.sum(weights[:, np.newaxis] * y_sorted[:mu], axis=0)

            # Update path states
            y_sigma = (x_new - x_mean) / sigma
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / chi_n * y_sigma
            h_sigma = 1.0 if (np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (eval_count - pop_size) / (n * c_sigma)))) < (1.4 + 2 / (n + 1)) * chi_n else 0.0

            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) / chi_n * y_sigma

            # Update covariance
            delta_h = (1 - h_sigma) * c_c * (2 - c_c) * np.outer(p_c, p_c)
            C = (1 - c_1 - c_mu) * C + c_1 * (delta_h + np.outer(p_c, p_c)) + c_mu * np.sum(weights[:, np.newaxis, np.newaxis] * np.outer(y_sorted[:mu] - x_mean, y_sorted[:mu] - x_mean) / sigma**2, axis=0)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))

            # Update mean
            x_mean = x_new

            # Regularize C to ensure symmetry and positive definiteness
            C = (C + C.T) / 2
            # Simple eigenvalue check/fix if needed, but usually stable in 5D
            # If C becomes singular, reset to diagonal
            try:
                np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = np.eye(n)
                p_sigma = np.zeros(n)
                p_c = np.zeros(n)
                sigma = 1.0

        return best_x, best_f
# EVOLVE-BLOCK-END
