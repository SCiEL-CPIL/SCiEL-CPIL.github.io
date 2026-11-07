"""Rewritten program using CMA-ES (Covariance Matrix Adaptation Evolution Strategy)."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES optimizer for the BBOB-style f16 (Weierstrass) task."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        # Initial population size
        lambda_ = 4 + int(3 * np.log(dim))  # standard CMA-ES lambda
        # Ensure at least 4 and not too large for budget
        lambda_ = max(4, min(lambda_, budget // 2))

        # Initial point: center of the box
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_f = fx

        # Initial step size
        sigma = 1.0

        # Mean
        m = x.copy()

        # Covariance matrix
        C = np.eye(dim)

        # Weights (logarithmic)
        weights = np.log(lambda_ / 2) - np.log(np.arange(1, lambda_ + 1))
        weights /= weights.sum()
        mu = int(np.sum(weights > 0))
        weights = weights[:mu]
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        cs = (mueff + 2) / (dim + mueff + 5)
        ds = 1 + 2 * max(0, (mueff - 1) / (dim + 1)) - cs
        cc = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c1 = (2 - 1 / (dim + 1.3) + 2 / (dim + 1)) / ((dim + 2) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((dim + 2) ** 2 + mueff))

        # Evolution path
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Precompute Cholesky factor
        B = np.linalg.cholesky(C).T
        D = np.sqrt(np.diag(B @ B.T)) if dim > 1 else np.array([1.0])
        # Actually, for C=I, B=I, D=1
        D = np.sqrt(np.diag(C))
        B = np.linalg.cholesky(C).T

        calls_made = 1

        while calls_made < budget:
            # Generate lambda_ offspring
            pop = []
            pop_f = []
            for i in range(lambda_):
                # Check if we can afford this evaluation
                if calls_made >= budget:
                    break
                z = rng.standard_normal(dim)
                y = m + sigma * (B @ z)
                y = np.clip(y, -5, 5)
                fy = func(y)
                pop.append(y)
                pop_f.append(fy)
                calls_made += 1

            if not pop:
                break

            # Sort by fitness
            order = np.argsort(pop_f)
            pop_sorted = [pop[i] for i in order]
            pop_f_sorted = [pop_f[i] for i in order]

            # Update best
            if pop_f_sorted[0] < best_f:
                best_f = pop_f_sorted[0]
                best_x = pop_sorted[0].copy()

            # Update mean
            m_old = m.copy()
            m = sum(w * pop_sorted[i] for i, w in enumerate(weights))

            # Update evolution path ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B.T @ ((m - m_old) / (sigma * D)))

            # Hessian norm check (clamping sigma)
            hsig = np.sqrt((1 - (1 - cs) ** (2 * (calls_made // lambda_))) / (1 - (1 - cs) ** 2)) if calls_made // lambda_ > 0 else 0
            if np.linalg.norm(ps) / np.sqrt(dim) < (1.4 + 2 / (dim + 1)) * hsig:
                # Update pc
                pc = (1 - cs) * pc + cs * (m - m_old) / sigma
            else:
                # Reset pc effect
                pc = (1 - cs) * pc

            # Update covariance
            delta = 1 if np.linalg.norm(ps) / np.sqrt(dim) < (1.4 + 2 / (dim + 1)) * hsig else 0
            C = (1 - c1 - cmu + c1 * delta) * C + c1 * np.outer(pc, pc)
            for i, w in enumerate(weights):
                y_i = (pop_sorted[i] - m_old) / sigma
                C += cmu * w * np.outer(y_i, y_i)

            # Re-orthogonalize
            C = (C + C.T) / 2
            # Ensure positive definiteness
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            C = eigvecs @ np.diag(eigvals) @ eigvecs.T

            # Update sigma
            sigma *= np.exp(cs / ds * (np.linalg.norm(ps) / np.sqrt(dim) - 1))

            # Update B and D
            B = np.linalg.cholesky(C).T
            D = np.sqrt(np.diag(B @ B.T))

            # Clamp sigma
            sigma = np.clip(sigma, 1e-6, 10.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
