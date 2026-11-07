"""Pure CMA-ES optimizer for BBOB-style f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with population-based sampling for the attractive sector."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # Safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size: lambda = 4 + floor(3 * log(n))
        lam = 4 + int(3 * np.log(dim))
        lam = max(lam, 4)
        # For dim=5, lam = 4 + floor(3*1.609) = 4 + 4 = 8
        # Use lam=10 as suggested in directive for better exploration
        lam = 10

        # Step size adaptation parameters
        c_sigma = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((lam - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + lam / dim) / (dim + 4 + 3 * lam / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + lam)
        c_mu = min(1 - c_1, 2 * ((lam / 2 + 1) / (dim + 2) - 1))

        # Weights for recombination
        mu = lam // 2  # Number of best points used
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Initialize
        rng = np.random.default_rng(42)
        x_mean = rng.uniform(-2, 2, dim)
        sigma = 2.0

        # Covariance matrix and its decomposition
        C = np.eye(dim)
        C_eigval = np.ones(dim)
        C_eigvec = np.eye(dim)

        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        best_f = np.inf
        best_x = x_mean.copy()
        calls_made = 0

        def time_ok():
            return time.time() < deadline

        while calls_made < budget and time_ok():
            # Generate population
            # Compute D = sqrt(C_eigval)
            D = np.sqrt(C_eigval)
            # B = C_eigvec * diag(D)
            B = C_eigvec @ np.diag(D)
            # Inverse of B
            B_inv = np.diag(1.0 / D) @ C_eigvec.T

            # Generate lambda sample points
            Y = rng.standard_normal((lam, dim))
            X = np.array([x_mean + sigma * (B @ y.T).T for y in Y])
            # Clip to box
            X = np.clip(X, -5, 5)

            # Evaluate population
            F = np.array([func(x) for x in X])
            calls_made += lam

            if not time_ok():
                break

            # Update best
            for i in range(lam):
                if F[i] < best_f:
                    best_f = F[i]
                    best_x = X[i].copy()

            # Sort by fitness
            order = np.argsort(F)
            X_sorted = X[order]
            F_sorted = F[order]

            # Weighted recombination
            y_w = np.sum(weights[:, np.newaxis] * X_sorted[:mu], axis=0) - x_mean
            x_new = x_mean + y_w

            # Update evolution path p_sigma
            norm_y_w = np.linalg.norm(y_w)
            h_sig = 1.0 if norm_y_w / (1 - (1 / (1 + c_sigma) ** (calls_made / (c_sigma * dim)))) < (1.4 + 2 / (dim + 1)) * np.sqrt((mu_eff - 1) / (dim + 1) + 1) else 0.0

            p_sigma = (1 - c_sigma) * p_sigma + c_sigma * h_sig * y_w / sigma

            # Update evolution path p_c
            p_c = (1 - c_c) * p_c + c_c * h_sig * y_w / sigma

            # Update covariance matrix
            delta_hsig = (1 - h_sig) * c_sigma * (2 - c_sigma)
            C = ((1 + c_1 * delta_hsig - c_1 - c_mu) * C +
                 c_1 * np.outer(p_c, p_c) +
                 c_mu * np.sum(weights[:, np.newaxis, np.newaxis] *
                               np.outer(y_w / sigma - B @ B_inv @ p_sigma,
                                        (y_w / sigma - B @ B_inv @ p_sigma).T),
                               axis=0))

            # Enforce symmetry
            C = (C + C.T) / 2

            # Eigendecomposition for next iteration
            try:
                C_eigval, C_eigvec = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                C_eigval = np.maximum(C_eigval, 1e-12)
            except np.linalg.LinAlgError:
                C = np.eye(dim)
                C_eigval = np.ones(dim)
                C_eigvec = np.eye(dim)

            # Update step size
            sigma = sigma * np.exp((c_sigma / d_sigma) * (np.linalg.norm(p_sigma) / np.sqrt(2 * dim) - 1))
            sigma = max(sigma, 1e-6)
            sigma = min(sigma, 10.0)

            # Update mean
            x_mean = x_new
            x_mean = np.clip(x_mean, -5, 5)

        # Final check: ensure we have a valid result
        if best_x is None or calls_made == 0:
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = func(best_x)
                calls_made += 1
            else:
                best_f = 0.0

        return best_x, best_f
# EVOLVE-BLOCK-END
