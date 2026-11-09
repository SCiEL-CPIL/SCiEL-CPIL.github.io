"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES style optimizer with adaptive step size and restarts."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        calls_made = 0

        def count_call(x):
            nonlocal calls_made
            calls_made += 1
            return func(x)

        # Use CMA-ES via scipy's minimize with Nelder-Mead as fallback,
        # but primarily implement a simple CMA-ES-like strategy.

        # Initial point: try center first, then random
        x0 = np.zeros(self.dim)
        f0 = count_call(x0)
        best_x = x0.copy()
        best_f = f0

        # Try a few random initializations to find a good starting point
        for i in range(5):
            xi = rng.uniform(-5, 5, self.dim)
            fi = count_call(xi)
            if fi < best_f:
                best_f = fi
                best_x = xi.copy()

        # Now run a population-based optimizer (simple CMA-ES)
        # Population size
        lam = max(4, int(4 + 3 * np.log(self.dim)))  # ~14 for dim=5
        sigma = 1.0
        x_mean = best_x.copy()
        f_mean = best_f

        # Covariance matrix (start as identity)
        C = np.eye(self.dim)
        # Evolution paths
        ps = np.zeros(self.dim)
        pc = np.zeros((self.dim, self.dim))

        # Adaptation parameters
        cs = (1 + 2 * max(0, (lam // 2 - 1) / (self.dim + 2))) / (self.dim + 1 + 2 * max(0, (lam // 2 - 1) / (self.dim + 2)))
        cm = (1 + 2 * max(0, (lam // 2 - 1) / (self.dim + 2))) / (self.dim + 1 + 2 * max(0, (lam // 2 - 1) / (self.dim + 2)))
        c1 = 2 / ((self.dim + 1.3) ** 2 + lam)
        cmu = min(0.5, c1 * (lam // 2 - 2) / (self.dim + 4))
        c_sigma = (lam // 2 + 2) / (self.dim + lam // 2 + 5)
        d_sigma = 1 + 2 * max(0, (lam // 2 - 1) / (self.dim + 2) - 1) + c_sigma
        chi_n = np.sqrt(self.dim) * (1 - 1 / (4 * self.dim) + 1 / (21 * self.dim ** 2))

        # Eigen decomposition cache
        D = np.sqrt(np.diag(C))
        B = np.eye(self.dim)

        # Main loop
        while calls_made < self.budget:
            # Generate population
            ys = []
            for i in range(lam):
                if calls_made >= self.budget:
                    break
                z = rng.standard_normal(self.dim)
                # Transform using B and D
                y = x_mean + sigma * (B @ (D * z))
                y = np.clip(y, -5, 5)
                fy = count_call(y)
                ys.append((fy, y))

            if not ys:
                break

            # Sort by fitness
            ys.sort(key=lambda t: t[0])

            # Update mean
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(1 + np.arange(mu))
            weights /= np.sum(weights)
            mueff = 1 / np.sum(weights ** 2)

            x_new_mean = np.sum([weights[i] * ys[i][1] for i in range(mu)], axis=0)
            x_old_mean = x_mean.copy()
            x_mean = x_new_mean

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ (D * ((x_mean - x_old_mean) / sigma)))
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls_made + 1))) / chi_n < (1.4 + 2 / (self.dim + 1))
            pc = (1 - cm) * pc + hsig * cm * (1 - cm) * np.outer(x_mean - x_old_mean, x_mean - x_old_mean) / sigma ** 2

            # Update covariance
            delta_d = hsig * (1 - cm) * (1 + cm) * (1 + cmu / cm)
            C = (1 + c1 * delta_d + cmu - c1) * C
            for i in range(mu):
                zi = (ys[i][1] - x_old_mean) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)

            # Update sigma
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update eigen decomposition periodically
            if calls_made % lam == 0:
                eigenvalues, B = np.linalg.eigh(C)
                eigenvalues = np.clip(eigenvalues, 1e-20, 1e10)
                D = np.sqrt(eigenvalues)
                # Ensure C is symmetric
                C = (C + C.T) / 2

            # Update best
            if ys[0][0] < best_f:
                best_f = ys[0][0]
                best_x = ys[0][1].copy()

        return best_x, best_f
# EVOLVE-BLOCK-END
