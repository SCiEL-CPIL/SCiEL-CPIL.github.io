"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive population and time guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2  # number of selected individuals

        # Weights
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w /= np.sum(w)
        mueff = 1.0 / np.sum(w**2)

        # Strategy parameters
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        cs = (mueff + 2) / (dim + mueff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        sigma = 1.0

        # Initial point
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Track best value after each call for potential use
        # But we just need to make calls

        # CMA-ES state
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        C = np.eye(dim)
        # Cholesky of C
        C_chol = np.eye(dim)

        # Initial sigma for 5D in [-5,5]^5
        sigma = 2.0

        calls_made = 1

        for gen in range(budget):
            # Check time
            if time.time() > deadline:
                break

            # Generate population
            # Use CMA-ES step
            # For simplicity and speed, we'll do a simplified CMA-ES
            # Actually, let's do a proper but efficient CMA-ES

            # Generate lambda_ samples
            Y = np.random.standard_normal((lambda_, dim))
            # Transform: x_i = x + sigma * C_chol @ Y_i
            X = x + sigma * (C_chol @ Y.T).T
            X = np.clip(X, -5, 5)

            # Evaluate
            F = np.array([func(xi) for xi in X])
            calls_made += lambda_

            # Update best
            idx_min = np.argmin(F)
            if F[idx_min] < best_f:
                best_f = F[idx_min]
                best_x = X[idx_min].copy()

            # Sort
            order = np.argsort(F)
            Y_sorted = Y[order]
            X_sorted = X[order]
            F_sorted = F[order]

            # Mean update
            y_w = np.sum(w[:, None] * Y_sorted[:mu], axis=0)
            x_new = x + sigma * (C_chol @ y_w)
            x_new = np.clip(x_new, -5, 5)

            # Update path
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (C_chol @ y_w)
            hsig = (np.linalg.norm(ps) /
                    np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) /
                    np.linalg.norm(C_chol)) < (1 + 2 / (dim + 1)) * ccs if (ccs := cs) else 0
            # Simplified: hsig = True for now, or use standard formula
            hsig = np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1 + 2 / (dim + 1)) * (1 / np.sqrt(cs))
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * y_w

            # Covariance update
            delta_hsig = (1 - hsig) * cc * (2 - cc)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + \
                c1 * (np.outer(pc, pc) + delta_hsig - 1) + \
                cmu * np.sum(w[:, None, None] * np.outer(Y_sorted[:mu], Y_sorted[:mu]), axis=0)

            # Ensure positive semi-definite
            # Symmetrize
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))

            # Update C_chol
            try:
                C_chol = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small diagonal
                C += 1e-6 * np.eye(dim)
                try:
                    C_chol = np.linalg.cholesky(C)
                except:
                    C_chol = np.eye(dim)

            x = x_new

            # Check convergence
            if np.linalg.norm(sigma * C_chol) < 1e-10:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
