"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """(1+1)-ES with CMA-ES-style covariance adaptation and restarts."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        deadline = time.time() + 285.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        # Parameters for CMA-ES-like adaptation
        # We'll use a simplified version with covariance matrix adaptation
        # and restart mechanism for local minima escape.

        # Phase 1: Initial random sampling
        n_init = min(20, budget)
        for i in range(n_init):
            if time.time() > deadline:
                break
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            calls_made += 1

        # Phase 2: CMA-ES-like optimization with restarts
        # We'll implement a simplified CMA-ES with:
        # - Covariance matrix adaptation
        # - Step size control
        # - Restart when stagnation detected

        sigma = 1.0
        C = np.eye(dim)  # Covariance matrix
        C_inv_sqrt = np.eye(dim)  # Square root inverse of C
        pc = np.zeros(dim)  # Path for covariance adaptation
        ps = np.zeros(dim)  # Path for step size adaptation
        cc = (4 + dim / 3) / ((dim + 4) * 2)
        cs = (2 + 2 / (dim + 1)) / ((dim + 1) * 2 + cc)
        cmu = 2 / ((dim + 1.3) ** 2 + 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        x = best_x.copy()
        fx = best_f
        no_improve = 0
        max_no_improve = 50  # Restart after 50 evaluations without improvement

        while calls_made < budget:
            if time.time() > deadline:
                break

            # Generate offspring using multivariate normal with covariance C
            z = rng.standard_normal(dim)
            y = x + sigma * C_inv_sqrt @ z
            y = np.clip(y, -5, 5)

            fy = func(y)
            calls_made += 1

            if fy < fx:
                # Success: update paths and best
                y_minus_x = (y - x) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * cc) * y_minus_x
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (calls_made + 1))) < (1.4 + 2 / (dim + 1)) * chi_n
                if hsig:
                    pc = (1 - cc) * pc + cc * y_minus_x
                C = (1 + cmu * (2 - cmu) - cc * (1 - cc)) * C + cmu * np.outer(y_minus_x, y_minus_x) - (1 - cc * (2 - cc)) * cc * C
                # Update step size
                sigma *= np.exp(cs / chi_n * (np.linalg.norm(ps) / chi_n - 1))
                x = y
                fx = fy
                no_improve = 0

                # Update global best
                if fx < best_f:
                    best_f = fx
                    best_x = x.copy()
            else:
                # Failure
                ps = (1 - cs) * ps
                no_improve += 1

            # Ensure C remains positive definite
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update C_inv_sqrt
            eigvals_inv_sqrt = 1.0 / np.sqrt(eigvals)
            C_inv_sqrt = eigvecs @ np.diag(eigvals_inv_sqrt) @ eigvecs.T

            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 1e-8, 10.0)

            # Check for restart
            if no_improve >= max_no_improve:
                # Restart from current best with reduced sigma
                x = best_x.copy()
                fx = best_f
                sigma *= 0.5
                C = np.eye(dim)
                C_inv_sqrt = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                no_improve = 0

        return best_x, best_f
# EVOLVE-BLOCK-END
