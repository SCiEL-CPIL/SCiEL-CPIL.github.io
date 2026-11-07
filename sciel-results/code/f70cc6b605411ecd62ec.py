"""Combined optimizer: IPOP-CMA-ES with logarithmic weights and proper updates."""
import numpy as np
import time


class Optimizer:
    """IPOP-CMA-ES optimizer for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.deadline = None

    def __call__(self, func):
        self.deadline = time.time() + 290.0  # Safety margin before 330s limit
        rng = np.random.default_rng(12345)

        n = self.dim
        budget = self.budget

        # Initial point: center of the box
        x = np.zeros(n)
        fx = func(x)
        self.calls += 1
        self.best_x = x.copy()
        self.best_f = fx

        # IPOP-CMA-ES parameters
        # Population size: lambda = max(4 + 3*ln(n), 2*n + 1)
        lambda_ = max(4 + int(3 * np.log(n)), 2 * n + 1)
        lambda_ = min(lambda_, budget // 2)

        # Initial sigma: 2.0 for exploration of [-5,5] box
        sigma = 2.0

        # CMA-ES state
        x_mean = x.copy()
        C = np.eye(n)
        C_inv = np.eye(n)

        # Logarithmic weights
        mu = lambda_ // 2
        w = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mu_eff = 1.0 / np.sum(w ** 2)

        # Learning rates
        c_sigma = (mu_eff + 2) / (n + mu_eff + 5)
        d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (n + 1)) + c_sigma
        c_c = (4 + mu_eff / n) / (n + 4 + 2 * mu_eff / n)
        c_1 = 2 / ((n + 1.3) ** 2 + mu_eff)
        c_mu = min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((n + 2) ** 2 + mu_eff))

        chi_n = np.sqrt(n) * (1 - 1.0 / (4 * n) + 1.0 / (21 * n ** 2))

        p_sigma = np.zeros(n)
        p_c = np.zeros(n)
        count_eval = 1

        while self.calls < self.budget:
            # Check deadline
            if time.time() > self.deadline:
                break

            # Generate population
            ys = []
            fs = []
            for i in range(lambda_):
                if time.time() > self.deadline or self.calls >= self.budget:
                    break
                z = rng.standard_normal(n)
                # Sample from multivariate normal with covariance C
                # y = x_mean + sigma * C_sqrt @ z
                # We use the Cholesky-like approach via eigendecomposition
                # For efficiency, we'll use the fact that C is symmetric positive definite
                # and use np.linalg.cholesky or similar
                try:
                    L = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    # Fallback if C is not positive definite
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-12)
                    L = eigvecs @ np.diag(np.sqrt(eigvals))
                y = x_mean + sigma * (L @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                self.calls += 1
                ys.append(y)
                fs.append(fy)
                if fy < self.best_f:
                    self.best_f = fy
                    self.best_x = y.copy()
                count_eval += 1

            if len(ys) < 2:
                break

            ys = np.array(ys)
            fs = np.array(fs)

            # Sort by fitness
            idx = np.argsort(fs)
            ys = ys[idx]
            fs = fs[idx]

            # Update mean
            x_mean_new = np.sum(w[:, None] * ys[:mu], axis=0)

            # Update evolution paths
            delta_x = x_mean_new - x_mean
            y_sigma = delta_x / sigma

            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (C_inv @ y_sigma)

            # h_sigma check
            norm_p_sigma = np.linalg.norm(p_sigma)
            denom = np.sqrt(1 - (1 - c_sigma) ** (count_eval))
            h_sigma = 1.0 if (norm_p_sigma / denom) < (1.4 + 2 / (n + 1)) * chi_n else 0.0

            # p_c update
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (C_inv @ y_sigma)

            # Update covariance matrix
            delta_h = (1 - h_sigma) * c_c * (2 - c_c)
            C = ((1 + c_1 + c_mu - c_1 + delta_h) * C -
                 c_1 * (np.outer(p_c, p_c) + delta_h * C) +
                 c_mu * np.sum(w[:, None, None] * np.outer(ys[:mu] - x_mean, ys[:mu] - x_mean), axis=0))

            # Ensure positive semi-definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T
            C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (norm_p_sigma / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            x_mean = x_mean_new

            # IPOP: reduce population size if stagnating
            if count_eval % 10 == 0 and self.calls < self.budget:
                if len(fs) > mu and np.std(fs[:mu]) < 1e-6:
                    lambda_ = max(int(lambda_ * 0.5), 4)
                    sigma *= 2.0
                    # Reset some state for IPOP restart
                    p_sigma = np.zeros(n)
                    p_c = np.zeros(n)

            # Reset if stuck
            if self.calls > 100 and self.best_f > 0 and count_eval % 50 == 0:
                sigma = min(sigma * 2.0, 10.0)

        return self.best_x, self.best_f
