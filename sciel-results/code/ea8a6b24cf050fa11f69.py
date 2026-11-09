"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard for BBOB f1 (Sphere)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin before 300s limit

        n = self.dim
        budget = self.budget

        # Initial point: center of the box (0,0,...,0) is a good prior
        # since the minimizer is in [-4,4]^5
        x = np.zeros(n)
        fx = func(x)
        calls = 1

        # CMA-ES parameters
        # For dim=5, use standard CMA-ES settings
        lambda_ = 4 * int(np.ceil(3 + np.log(n)))  # population size
        # lambda_ = 20 for n=5
        cc = (4 + n / (n + 4)) / ((n + 1.3) + 4 + n / (n + 4))  # evolution path for sigma
        cs = (n + 2) / (n + 2 + 4)  # evolution path for cm
        cmu = min(1 - 2 / (n + 2), (n - 2) / (n + n - 2)) / 2  # weight for covariance update

        # Weights
        mu = lambda_ // 2
        w = np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mu_eff = 1 / (w ** 2).sum()

        # Initial step size
        sigma = 2.0

        # Evolution paths
        ps = np.zeros(n)
        pc = np.zeros(n)

        # Covariance matrix
        C = np.eye(n)
        C_sqrt = np.eye(n)

        # Precompute eigen decomposition (will be updated)
        # For small n, we can recompute each iteration

        def sample_directions(k):
            """Sample k random directions from N(0, C)."""
            # Use Cholesky or eigendecomposition
            # For simplicity, use eigendecomposition
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                z = rng.standard_normal((k, n))
                return z @ (np.sqrt(eigvals)[:, np.newaxis] * eigvecs.T)
            except np.linalg.LinAlgError:
                # Fallback to isotropic
                return rng.standard_normal((k, n))

        # Main loop
        while calls < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Sample offspring
            directions = sample_directions(lambda_)
            offspring = np.array([x + sigma * d for d in directions])
            # Clip to box
            offspring = np.clip(offspring, -5, 5)

            # Evaluate offspring
            offspring_f = np.array([func(xi) for xi in offspring])
            calls += lambda_

            # Sort by fitness
            idx = np.argsort(offspring_f)
            sorted_offspring = offspring[idx]
            sorted_offspring_f = offspring_f[idx]

            # Update best if found
            if sorted_offspring_f[0] < fx:
                x = sorted_offspring[0].copy()
                fx = sorted_offspring_f[0]

            # Weighted mean of top mu
            m = np.sum(w[:, np.newaxis] * sorted_offspring[:mu], axis=0)

            # Update evolution path for sigma
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-10)
            C_inv_sqrt = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * C_inv_sqrt @ (m - x)

            # Update sigma
            sigma *= np.exp(cs / (n + 2) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** 2) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update evolution path for cm
            pc = (1 - cc) * pc + cc * (C_inv_sqrt @ (m - x)) / sigma

            # Update covariance
            delta = (1 + 2 * (mu_eff < (n + 1) / 2) * cc / (1 - cc)) * (np.outer(pc, pc) - C)
            C = (1 + cmu * (1 - (1 - cc) ** 2)) * C - cmu * (np.outer(pc, pc) - C)
            C = C + cmu * sum(w[i] * np.outer(sorted_offspring[i] - x, sorted_offspring[i] - x) for i in range(mu))
            # Ensure symmetry
            C = (C + C.T) / 2

            # Update x
            x = m.copy()

        return x, fx
# EVOLVE-BLOCK-END
