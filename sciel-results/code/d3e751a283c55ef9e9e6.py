"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # 20s safety margin before 300s limit

        # CMA-ES parameters
        # Population size
        lam = max(4, int(4 + 3 * np.log(dim)))
        # Learning rates
        cc = (4 + lam / dim) / (dim + 4 + 2 * lam / dim)
        cmu = min(2, (2 + 2 / (dim + 1)) / (dim + 2 + 2 / (dim + 1)))
        # Initial step size
        sigma = 2.0

        # Initial point near origin (prior: minimizer in [-4,4]^5)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)
        # Covariance matrix
        C = np.eye(dim)
        Cinv = np.eye(dim)
        # Eigen decomposition for C (updated periodically)
        C_eigvals = np.ones(dim)
        C_eigvecs = np.eye(dim)
        # Update interval for eigen decomposition
        eig_interval = max(1, int(1 / (cc * cmu)))
        counter = 0
        # Normalization of C
        D = 1.0
        # Step size control
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Budget tracking
        calls_made = 1

        for gen in range(budget - 1):
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            Y = np.zeros((lam, dim))
            for i in range(lam):
                # Sample from N(0, C) using current eigen decomposition
                z = np.random.randn(dim)
                # Transform: x = xmean + sigma * C_eigvecs @ diag(sqrt(C_eigvals)) @ z
                Y[i] = sigma * (C_eigvecs @ (np.sqrt(C_eigvals) * z))

            # Evaluate population
            F = np.zeros(lam)
            for i in range(lam):
                y = np.clip(x + Y[i], -5, 5)
                F[i] = func(y)
                calls_made += 1
                if F[i] < best_fx:
                    best_fx = F[i]
                    best_x = y.copy()
                if time.time() > deadline:
                    # Emergency: break out of inner loop
                    F[i:] = np.inf
                    break

            if time.time() > deadline:
                break

            # Sort by fitness
            idx = np.argsort(F)
            Y_sorted = Y[idx]
            F_sorted = F[idx]

            # Update mean
            weights = np.zeros(lam)
            mu = lam // 2
            # Log-linear weights
            w = np.log(np.arange(1, mu + 1))
            w /= np.sum(w)
            weights[:mu] = w
            w_max = 2.0 / (np.sum(weights ** 2) ** 0.5)
            weights[:mu] = np.minimum(weights[:mu], w_max)
            weights[:mu] /= np.sum(weights[:mu])
            mueff = 1.0 / np.sum(weights[:mu] ** 2)

            x_new = x + np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update evolution paths
            ps = (1 - cc) * ps + np.sqrt(cc * (2 - cc) * mueff) * (
                Cinv @ (sigma * (x_new - x))
            )
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cc) ** (2 * (gen + 1)))) < (
                1.4 + 2 / (dim + 1)
            ) * chi_n

            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc)) * (x_new - x) / sigma
            x = x_new

            # Update covariance
            delta_hs = (1.0 if hsig else 0.0) - (1 - cc)
            C = (
                (1 + cmu * delta_hs - cmu) * C
                + cmu
                * np.sum(
                    weights[:, None, None]
                    * np.outer(Y_sorted[:mu], Y_sorted[:mu]),
                    axis=0,
                )
                + cmu
                * delta_hs
                * np.outer(pc, pc)
            )

            # Ensure symmetry
            C = 0.5 * (C + C.T)

            # Update eigen decomposition periodically
            counter += 1
            if counter >= eig_interval:
                C_eigvals, C_eigvecs = np.linalg.eigh(C)
                C_eigvals = np.maximum(C_eigvals, 1e-12)
                Cinv = (C_eigvecs @ np.diag(1.0 / C_eigvals)) @ C_eigvecs.T
                D = 1.0 / np.sqrt(np.prod(C_eigvals))
                counter = 0

            # Step size control (1/5 rule variant)
            sigma *= np.exp(
                (cc / (dim + 1)) * (np.linalg.norm(ps) / chi_n - 1)
            )
            sigma = np.clip(sigma, 1e-12, 1e6)

            # Track best
            if F_sorted[0] < best_fx:
                best_fx = F_sorted[0]
                best_x = np.clip(x + Y_sorted[0], -5, 5)

        return best_x, best_fx
# EVOLVE-BLOCK-END
