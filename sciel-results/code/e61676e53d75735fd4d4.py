"""Optimizer for BBOB f5 (linear slope) with deadline guard and adaptive restarts."""
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 270.0  # safety margin before 300s limit
        rng = np.random.default_rng(42)

        best_x = None
        best_f = np.inf
        calls_made = 0
        budget = self.budget
        dim = self.dim

        def time_exceeded():
            return time.time() >= deadline

        def call_func(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Phase 1: Adaptive (1+1)-ES with sigma adaptation
        # Use a population-based approach for better exploration
        # Start with a reasonable initial point

        # Initial sampling to get a rough idea
        n_initial = min(20, budget)
        best_x = None
        best_f = np.inf

        for i in range(n_initial):
            if time_exceeded() or calls_made >= budget:
                break
            x = rng.uniform(-5, 5, dim)
            f = call_func(x)
            if f < best_f:
                best_f = f
                best_x = x.copy()

        if best_x is None:
            x = np.zeros(dim)
            best_f = call_func(x)
            best_x = x.copy()

        # Phase 2: CMA-ES-like adaptive search
        # We'll implement a simplified CMA-ES with adaptive step size
        x = best_x.copy()
        fx = best_f
        sigma = 1.0  # initial step size
        sigma_min = 1e-6
        sigma_max = 10.0

        # Covariance matrix (start with identity)
        C = np.eye(dim)
        # Inverse covariance
        C_inv = np.eye(dim)
        # Square root of C for sampling
        C_sqrt = np.eye(dim)

        # Adaptation parameters
        alpha_mu = 1.0 / (dim + 4)  # learning rate for sigma
        c_sigma = (2 + 2.0 / (dim + 1)) / (dim + 4 + 2)
        c_c = (4 + dim / dim) / (dim + 4 + 3)
        c_1 = 2 / ((dim + 1.3) ** 2 + dim)
        c_mu = min(1 - c_1, 2 * ((mu_eff + 0.5) / (dim + 2) ** 2 - 1))

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2  # number of parents

        # Weighted recombination
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Evolution path for covariance
        pc = np.zeros(dim)
        # Evolution path for sigma
        ps = np.zeros(dim)

        # Hessian of log det C for preconditioning (approximate)
        def mat_update(C, C_inv, v, w, alpha):
            """Rank-1 update of C and its inverse."""
            C += alpha * np.outer(v, v)
            # Use Sherman-Morrison for inverse update
            Cv = C_inv @ v
            denom = 1 + alpha * (v @ Cv)
            C_inv -= np.outer(Cv, Cv) / denom
            return C, C_inv

        for iteration in range(budget - n_initial):
            if time_exceeded() or calls_made >= budget:
                break

            # Generate population
            offspring = []
            for i in range(lambda_):
                if time_exceeded() or calls_made >= budget:
                    break
                # Sample from N(x, sigma^2 * C)
                z = rng.multivariate_normal(np.zeros(dim), C, 1).flatten()
                y = np.clip(x + sigma * z, -5, 5)
                fy = call_func(y)
                offspring.append((fy, y))

            if not offspring:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update evolution paths
            y_w = np.zeros(dim)
            for i in range(mu):
                y_w += weights[i] * (offspring[i][1] - x) / sigma

            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * y_w
            ps_norm = np.linalg.norm(ps)

            # Update mean
            x_new = x + sigma * y_w
            x_new = np.clip(x_new, -5, 5)

            # Update covariance
            delta_sig = (ps_norm / np.sqrt(1 - (1 - c_sigma) ** (2 * (iteration + 1)))) - 1
            rank_one = np.outer(y_w, y_w)
            rank_mu = np.zeros((dim, dim))
            for i in range(mu):
                zi = (offspring[i][1] - x) / sigma
                rank_mu += weights[i] * np.outer(zi, zi)

            C, C_inv = mat_update(C, C_inv, y_w, None, c_c * (2 - c_c) * delta_sig + c_1)
            # Add rank-mu contribution
            C += c_mu * rank_mu
            # Re-invert (approximate - for small dim this is fine)
            try:
                C_inv = np.linalg.inv(C)
            except np.linalg.LinAlgError:
                C_inv = np.eye(dim)

            # Update sigma
            sigma *= np.exp(c_sigma / sigma_min * (ps_norm / np.sqrt(mu_eff) - np.sqrt(2 * (dim + 1) / 2 * (1 - 1 / (dim + 1)))) * 0.01)
            sigma = np.clip(sigma, sigma_min, sigma_max)

            # Check for improvement
            best_offspring_f = offspring[0][0]
            if best_offspring_f < fx:
                fx = best_offspring_f
                x = offspring[0][1].copy()
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # No improvement, still update x to weighted average
                x = x_new

            # Ensure C is positive definite
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += 1e-10 * np.eye(dim)
                try:
                    C_inv = np.linalg.inv(C)
                except np.linalg.LinAlgError:
                    C_inv = np.eye(dim)

            # Update C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C_sqrt = np.eye(dim)

        # Phase 3: Fine-tuning with smaller steps
        if not time_exceeded() and calls_made < budget:
            sigma_fine = 0.1
            for _ in range(budget - calls_made):
                if time_exceeded() or calls_made >= budget:
                    break
                y = np.clip(x + sigma_fine * rng.standard_normal(dim), -5, 5)
                fy = call_func(y)
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    sigma_fine *= 1.2
                else:
                    sigma_fine *= 0.9

        return best_x, best_f
