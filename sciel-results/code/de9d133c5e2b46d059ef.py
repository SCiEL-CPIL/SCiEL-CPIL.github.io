"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with 1/5 success rule for step-size adaptation."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit

        rng = np.random.default_rng(0)

        # Initial point
        x = rng.uniform(-5, 5, dim)
        fx = float(func(x))
        best_x = x.copy()
        best_f = fx

        # CMA-ES parameters
        sigma = 1.0
        # Success rate window
        window = 15
        successes = []
        # Step size adaptation with 1/5 rule
        c_sigma = 0.01 + 2 / (dim + 1) ** 1.5
        d_sigma = 1 + 2 * max(0, 1 / (dim + 3) - 1)

        # Population size
        lambda_ = 4 + int(3 * np.log(dim))
        mu = lambda_ // 2
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / (weights @ weights)

        # Covariance matrix update
        c1 = (2 * (mu_eff - 2 + 2 / (dim + 1))) / ((dim + 2) ** 2 + mu_eff)
        cmu = min(2 * (1 - c1) / ((dim + 2) ** 3 + mu_eff), 0.5)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 += (1 - c1) * cs
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chiN = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        # Generate initial population
        def generate_population(x_center, sigma, C, rng):
            B, D = np.linalg.eigh(C)
            D = np.sqrt(np.maximum(D, 0))
            # Generate samples
            z = rng.standard_normal((lambda_, dim))
            # Transform to have covariance C
            y = (z * D) @ B.T
            return x_center + sigma * y, z

        # Track best
        for gen in range(budget // lambda_ + 1):
            if time.time() > deadline:
                break

            # Generate population
            pop, z = generate_population(x, sigma, C, rng)
            # Clip to bounds
            pop = np.clip(pop, -5, 5)

            # Evaluate
            fvals = np.array([float(func(xi)) for xi in pop])

            # Update best
            idx_best = np.argmin(fvals)
            if fvals[idx_best] < best_f:
                best_f = fvals[idx_best]
                best_x = pop[idx_best].copy()

            # Sort by fitness
            order = np.argsort(fvals)
            pop_sorted = pop[order]
            fvals_sorted = fvals[order]
            z_sorted = z[order]

            # Weighted mean
            m_new = (weights[:, None] * pop_sorted[:mu]).sum(axis=0)

            # Update step size (1/5 rule via CMA-ES)
            diff = m_new - x
            if np.linalg.norm(diff) > 1e-14:
                ps = (1 - cs) * ps + cs * np.sqrt(mu_eff) / chiN * (C @ diff) / sigma
                hsig = int(np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) <
                            (1.4 + 2 / (dim + 1)) * chiN)
                pc = (1 - c1) * pc + hsig * c1 * diff / sigma

                # Update C
                delta_hsig = (1 - hsig) * c1 * (2 - c1)
                C = (1 + c1 * (c1 - 2 + cmu * mu_eff) - delta_hsig) * C
                C += c1 * np.outer(pc, pc)
                C += cmu * (weights[:, None, None] * np.einsum('ij,ik->jk',
                    (pop_sorted[:mu] - m_new) / sigma, (pop_sorted[:mu] - m_new) / sigma)).sum(axis=0)

                # Symmetrize
                C = (C + C.T) / 2

                # Update sigma
                sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chiN - 1))

                x = m_new
            else:
                x = m_new

            # Ensure C is positive definite
            eigvals = np.linalg.eigvalsh(C)
            if eigvals.min() < 1e-12:
                C = C + 1e-12 * np.eye(dim)

            # If sigma becomes too small, try a restart with larger sigma
            if sigma < 1e-6:
                # Try a few random restarts
                for _ in range(min(10, budget - (gen * lambda_ + 1))):
                    if time.time() > deadline:
                        break
                    x_try = rng.uniform(-5, 5, dim)
                    f_try = float(func(x_try))
                    if f_try < best_f:
                        best_f = f_try
                        best_x = x_try.copy()
                break

            # Check if we've used enough budget
            if (gen + 1) * lambda_ >= budget:
                break

        return best_x, best_f
# EVOLVE-BLOCK-END
