"""CMA-ES based optimizer for BBOB-style Rosenbrock function."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES (Covariance Matrix Adaptation Evolution Strategy) for continuous optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_x = None
        self.best_f = np.inf
        self.calls_made = 0
        self.start_time = 0.0
        self.deadline = 0.0

    def __call__(self, func):
        self.start_time = time.time()
        self.deadline = self.start_time + 280.0  # Safety margin before 300s limit

        def wrapped_func(x):
            if self.calls_made >= self.budget:
                raise StopIteration("Budget exhausted")
            if time.time() > self.deadline:
                raise StopIteration("Time limit exceeded")

            x = np.clip(x, -5.0, 5.0)
            self.calls_made += 1
            val = func(x)

            if val < self.best_f:
                self.best_f = val
                self.best_x = x.copy()

            return val

        # CMA-ES parameters
        dim = self.dim
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Step size (sigma)
        sigma = 2.0  # Initial step size, tuned for [-5,5]^5
        # Means
        mean = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        # Eigen decomposition for efficient sampling
        D = np.sqrt(np.diag(C))
        B = np.eye(dim)

        # Adaptation parameters
        alpha_mu = 1.0 / (lam + 4)
        c_sigma = (2 * (alpha_mu + 0.3)) / (dim + 4 + alpha_mu)
        c_c = (4 + dim / 4) / (dim + 4 + 4 + dim / 4)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * (alpha_mu - 2 + 1 / dim)) / ((dim + 2) ** 2 + lam)
        d_sigma = 1 + c_sigma + 2 * max(0, np.sqrt((alpha_mu - 1) / (alpha_mu + 1)) - 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        # Weights for recombination
        weights = np.log(lam / 2 + 0.5) - np.log(np.arange(1, lam + 1))
        weights /= np.sum(weights)
        mu = lam // 2 + 1  # Number of parents
        mu_eff = 1 / np.sum(weights[:mu] ** 2)
        # Normalized weights
        w = weights[:mu]

        # Track progress
        n_restarts = 0
        max_restarts = 3

        while self.calls_made < self.budget and time.time() < self.deadline:
            # Generate population
            offspring = []
            for i in range(lam):
                if self.calls_made >= self.budget or time.time() > self.deadline:
                    break
                # Sample from multivariate normal using eigen decomposition
                z = np.random.randn(dim)
                y = B @ (D * z)
                x = mean + sigma * y
                x = np.clip(x, -5.0, 5.0)
                offspring.append((x, wrapped_func(x)))

            if len(offspring) < mu:
                break

            # Sort offspring by fitness
            offspring.sort(key=lambda t: t[1])
            selected = offspring[:mu]

            # Weighted recombination
            new_mean = np.zeros(dim)
            for i, (x, f) in enumerate(selected):
                new_mean += w[i] * x
            mean = new_mean

            # Update evolution paths
            old_mean = mean - sigma * (B @ (D * np.mean([np.random.randn(dim) for _ in range(mu)] / mu)))
            # More standard approach:
            y_weighted = np.zeros(dim)
            for i, (x, f) in enumerate(selected):
                y_weighted += w[i] * (x - mean) / sigma

            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (B @ (D @ (y_weighted / sigma)))
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * y_weighted

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 10.0)

            # Update covariance matrix
            v = p_c / np.linalg.norm(p_c) if np.linalg.norm(p_c) > 0 else np.zeros(dim)
            delta_sigma = (np.linalg.norm(p_sigma) / chi_n - 1) * c_sigma * (2 - c_sigma)
            C = (1 + c_1 * delta_sigma - c_1 - c_mu) * C + \
                c_1 * np.outer(v, v) + \
                c_mu * sum(w[i] * np.outer((x - mean) / sigma, (x - mean) / sigma) for i, (x, f) in enumerate(selected))

            # Ensure C is symmetric and positive definite
            C = (C + C.T) / 2
            # Eigen decomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.sqrt(eigvals)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                B = np.eye(dim)
                D = np.ones(dim)

            # Check for stagnation or convergence
            if sigma < 1e-10 or np.linalg.norm(mean - self.best_x) < 1e-8:
                # Restart with larger sigma if we have budget
                if n_restarts < max_restarts and self.calls_made < self.budget - 50:
                    n_restarts += 1
                    sigma = max(sigma, 1.0)
                    mean = self.best_x.copy()
                    C = np.eye(dim)
                    B = np.eye(dim)
                    D = np.ones(dim)
                    p_sigma = np.zeros(dim)
                    p_c = np.zeros(dim)
                else:
                    break

        # Final refinement with (1+1)-ES if budget remains
        if self.best_x is not None and self.calls_made < self.budget and time.time() < self.deadline:
            self._refine_1plus1(func, wrapped_func)

        if self.best_x is None:
            self.best_x = np.zeros(self.dim)
            self.best_f = func(np.clip(self.best_x, -5, 5))

        return self.best_x, self.best_f

    def _refine_1plus1(self, func, wrapped_func):
        """(1+1)-ES refinement around the best point found."""
        if self.calls_made >= self.budget:
            return
        if time.time() > self.deadline:
            return

        rng = np.random.default_rng(42)
        x = self.best_x.copy()
        fx = self.best_f
        sigma = 0.5

        remaining = self.budget - self.calls_made
        for _ in range(remaining):
            if self.calls_made >= self.budget:
                break
            if time.time() > self.deadline:
                break

            y = np.clip(x + sigma * rng.standard_normal(self.dim), -5, 5)
            fy = wrapped_func(y)

            if fy < fx:
                x, fx = y, fy
                sigma *= 1.1
            else:
                sigma *= 0.9

            if sigma < 1e-8:
                break

        if fx < self.best_f:
            self.best_f = fx
            self.best_x = x.copy()
# EVOLVE-BLOCK-END
