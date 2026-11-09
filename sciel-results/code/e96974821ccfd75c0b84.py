"""CMA-ES based optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """Population-based CMA-ES with full covariance adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        d = self.dim

        # Initialize
        x = rng.uniform(-4, 4, d)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        calls = 1

        # CMA-ES parameters
        lam = max(4, int(np.ceil(4 + 3 * np.log(d))))  # population size
        sigma = 2.0
        C = np.eye(d)
        C_sqrt = np.eye(d)  # Cholesky factor

        # Step sizes for covariance update
        cmu = 1 + 2 / ((d + 1) ** 2 - 2) - 1 / lam
        ccs = (4 + d / 4) / (d + 4 + 2 / d)
        cc = (2 + 2 / (d + 1.3)) / (d + 1 + cc + 2 * cmu)
        cs = (cs + 1) if False else (4 + d / 4) / (d + 4 + 2 / d)  # same as ccs
        c_sigma = (2 + 2 / (d + 1.3)) / (d + 1 + cc + 2 * cmu)

        # Eigenvalue bounds for covariance
        D = 1.0
        D_min = 1.0
        D_max = 1.0
        D_min = 1.0 / (1 + 1e-4)
        D_max = 1.0 + 1e-4

        ps = np.zeros(d)
        pc = np.zeros(d)
        chi_d = np.sqrt(d) * (1 - 1 / (4 * d) + 1 / (21 * d ** 2))

        # Main loop
        while calls < self.budget:
            # Generate population
            offspring = []
            offspring_f = []
            for i in range(lam):
                z = rng.standard_normal(d)
                y = x + sigma * C_sqrt @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls += 1
                offspring.append(y)
                offspring_f.append(fy)
                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()

                if calls >= self.budget:
                    break

            # Sort by fitness
            indices = np.argsort(offspring_f)
            offspring = [offspring[i] for i in indices]
            offspring_f = [offspring_f[i] for i in indices]
            mu = lam // 2  # number of parents

            # Weights
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1 / np.sum(weights ** 2)

            # Mean update
            mean_old = x.copy()
            x = sum(weights[i] * offspring[i] for i in range(mu))

            # Update evolution paths
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * C_sqrt.T @ ((x - mean_old) / sigma)
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (self.budget - calls))) <
                    (1.4 + 2 / (d + 1)) * chi_d)
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x - mean_old) / sigma

            # Update covariance
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 - cmu * delta_hsig) * C + cmu * (
                sum(weights[i] * np.outer(offspring[i] - mean_old, offspring[i] - mean_old) for i in range(mu)) / sigma ** 2 +
                delta_hsig * np.outer(pc, pc)
            )

            # Ensure positive definite: add regularization if needed
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                C = 0.5 * (C + C.T) + 1e-6 * np.eye(d)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C = np.eye(d)
                    C_sqrt = np.eye(d)

            # Update sigma
            sigma *= np.exp(c_sigma / chi_d * (np.linalg.norm(ps) / chi_d - 1))
            sigma = max(1e-10, min(sigma, 10.0))

            # Rescale C to keep expected step size ~1
            D = np.sqrt(np.trace(C) / d)
            C = C / (D * D)
            C_sqrt = C_sqrt / D

            # Check convergence
            if sigma < 1e-12 or np.all(np.abs(x - best_x) < 1e-10):
                # Exploit remaining budget around best
                while calls < self.budget:
                    y = np.clip(best_x + 0.1 * rng.standard_normal(d), -5, 5)
                    fy = func(y)
                    calls += 1
                    if fy < best_fx:
                        best_fx = fy
                        best_x = y.copy()
                break

        return best_x, best_fx
# EVOLVE-BLOCK-END
