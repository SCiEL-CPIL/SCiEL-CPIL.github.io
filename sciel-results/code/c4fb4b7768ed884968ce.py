"""Rewritten optimizer using CMA-ES style strategy."""
# EVOLVE-BLOCK-START
import numpy as np


class Optimizer:
    """CMA-ES based continuous optimizer."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget

        # Set up CMA-ES parameters
        # Population size
        lambda_ = max(4, int(4 + 3 * np.log(dim)))
        # Step size
        sigma = 2.0
        # Initial mean
        m = np.zeros(dim)

        # Covariance matrix (initially identity)
        C = np.eye(dim)
        # Cholesky decomposition of C for sampling
        C_sqrt = np.eye(dim)

        # Evolution paths
        ps = np.zeros(dim)
        pc = np.zeros(dim)

        # Update rates
        cs = (lambda_ + 2) / (dim + lambda_ + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + lambda_)
        cmu = min(1 - c1, 2 * ((lambda_ * 0.5 - 1 + 1.0 / lambda_) / (dim + 2) - 1))
        # Damping for step size control
        d_sigma = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1) - 1)) + cs
        # Path length normalization
        chi_n = np.sqrt(dim) * (1 - 1.0 / (4 * dim) + 1.0 / (21 * dim ** 2))

        # Weights for recombination
        mu = lambda_ // 2
        weights = np.log(np.arange(lambda_, 0, -1) + 1e-10) - np.log(np.arange(mu, 0, -1) + 1e-10)
        weights = weights[:mu]
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        # Best solution tracking
        best_x = m.copy()
        best_f = func(m)

        # Evaluate initial population if needed, but we start with mean
        # Actually, let's evaluate the mean first
        # Then generate population each iteration

        # We need to be careful with budget
        # Each iteration costs lambda_ evaluations
        # Plus the initial evaluation

        calls = 1
        if calls >= budget:
            return best_x, best_f

        # Main loop
        while calls + lambda_ <= budget:
            # Generate population
            pop = np.zeros((lambda_, dim))
            for i in range(lambda_):
                z = np.random.standard_normal(dim)
                pop[i] = m + sigma * (C_sqrt @ z)

            # Clip to box
            pop = np.clip(pop, -5, 5)

            # Evaluate population
            fvals = np.zeros(lambda_)
            for i in range(lambda_):
                fvals[i] = func(pop[i])
                calls += 1

            # Track best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = pop[idx_min].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]

            # Update mean
            m_old = m.copy()
            m_new = np.sum(weights[:, np.newaxis] * pop_sorted[:mu], axis=0)
            m = m_new

            # Update evolution paths
            # ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (C_sqrt.T @ ((m_new - m_old) / sigma))
            # hsig = 1 if ||ps|| / sqrt(1 - (1-cs)^{2/gen}) < (1.4 + 2/(dim+1)) * chi_n
            # Approximate: gen counter
            gen = calls / lambda_
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * gen)) < (1.4 + 2 / (dim + 1)) * chi_n else 0

            # pc
            pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) * (m_new - m_old) / sigma

            # Update covariance
            delta_hsig = (1 - hsig) * cs * (2 - cs)
            C = (1 + c1 * delta_hsig - c1 - cmu) * C + c1 * np.outer(pc, pc)
            for i in range(mu):
                y_i = (pop_sorted[i] - m_old) / sigma
                C += cmu * weights[i] * np.outer(y_i, y_i)

            # Ensure positive semi-definiteness
            C = (C + C.T) / 2
            # Eigenvalue check
            eigvals = np.linalg.eigvalsh(C)
            if np.min(eigvals) < 1e-10:
                C += (1e-10 - np.min(eigvals)) * np.eye(dim)

            # Update sigma
            sigma *= np.exp((cs / d_sigma) * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-10, 100)

            # Update C_sqrt
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Add small diagonal
                C += 1e-8 * np.eye(dim)
                try:
                    C_sqrt = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C_sqrt = np.eye(dim)

            # Boundary handling: if too many points are clipped, shrink sigma
            clipped_count = np.sum(np.any((pop >= 4.99) | (pop <= -4.99), axis=1))
            if clipped_count > lambda_ * 0.5:
                sigma *= 0.5

            # Convergence check
            if sigma < 1e-12:
                break

        # Use remaining budget for local refinement
        remaining = budget - calls
        if remaining > 0:
            # Simple local search around best_x
            step = max(sigma, 0.1)
            for _ in range(remaining):
                if calls >= budget:
                    break
                direction = np.random.standard_normal(dim)
                candidate = best_x + step * direction
                candidate = np.clip(candidate, -5, 5)
                f_candidate = func(candidate)
                calls += 1
                if f_candidate < best_f:
                    best_f = f_candidate
                    best_x = candidate.copy()
                else:
                    step *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
