"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based CMA-ES for Rastrigin rotated landscape."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 10  # population size
        mu = lambda_ // 2  # number of parents

        # Initial mean: center of the box
        m = np.zeros(dim)
        # Initial sigma: large enough to explore the box
        sigma = 4.0

        # Covariance matrix
        C = np.eye(dim)
        # Inverse covariance
        C_inv = np.eye(dim)

        # Step sizes
        c_sigma = (mu + 0.5) / (dim + 2)
        c_c = (4 + mu / dim) / (dim + 4)
        d_sigma = 1 + 2 * max(0, (mu - 1) / (dim + 1) - 1) + c_sigma

        # Eigenvalues and eigenvectors of C
        eigvals, eigvecs = np.linalg.eigh(C)
        # Ensure positive eigenvalues
        eigvals = np.maximum(eigvals, 1e-12)

        # Precompute
        sqrt_eigvals = np.sqrt(eigvals)
        C_sqrt_inv = eigvecs @ np.diag(1.0 / sqrt_eigvals) @ eigvecs.T

        # State variables
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Evaluation counter
        eval_count = 0

        def clip(x):
            return np.clip(x, -5, 5)

        # Initial population: sample around origin with large sigma
        # Use a diverse initial spread
        rng = np.random.default_rng(42)

        # First, do a broad random search to find a good starting point
        # Sample several points to initialize CMA-ES better
        best_x = None
        best_f = np.inf
        n_init = min(50, budget // 10)
        for i in range(n_init):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            eval_count += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        if eval_count >= budget:
            return best_x, best_f

        # Initialize CMA-ES around the best point found
        m = best_x.copy()
        sigma = 2.0  # moderate initial sigma

        # Reset covariance
        C = np.eye(dim)
        C_inv = np.eye(dim)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-12)
        sqrt_eigvals = np.sqrt(eigvals)
        C_sqrt_inv = eigvecs @ np.diag(1.0 / sqrt_eigvals) @ eigvecs.T
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        while eval_count < budget:
            # Check deadline
            if time.time() > deadline:
                break

            # Generate population
            offspring = []
            fitnesses = []

            # Compute weights
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            # Generate lambda_ offspring
            for i in range(lambda_):
                # Transform normal sample
                z = rng.standard_normal(dim)
                # Apply C^{-1/2}
                x = m + sigma * (C_sqrt_inv @ z)
                x = clip(x)
                fx = func(x)
                eval_count += 1
                if eval_count >= budget:
                    offspring.append((fx, x))
                    break
                offspring.append((fx, x))

            if len(offspring) < 2:
                break

            # Sort by fitness
            offspring.sort(key=lambda t: t[0])

            # Update best
            best_fx, best_x = offspring[0]
            if best_fx < best_f:
                best_f = best_fx
                best_x = best_x.copy()

            # Take mu best
            mu_offspring = offspring[:mu]
            x_w = np.array([x for _, x in mu_offspring])

            # Weighted mean
            m_new = np.sum(weights[:, None] * x_w, axis=0)

            # Update covariance matrix
            # Compute difference vectors
            diff = x_w - m
            # Update C
            C = (1 - c_c) * C + c_c * (np.outer(m_new - m, m_new - m) / mu_eff +
                                          np.sum(weights[:, None] * np.outer(diff, diff), axis=0) / mu_eff -
                                          c_c * (1 - c_c) * C)

            # Ensure symmetry
            C = (C + C.T) / 2

            # Update eigen decomposition
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-12)
            sqrt_eigvals = np.sqrt(eigvals)
            C_sqrt_inv = eigvecs @ np.diag(1.0 / sqrt_eigvals) @ eigvecs.T

            # Update step size
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * \
                      (C_sqrt_inv @ (m_new - m) / sigma)
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * (m_new - m)

            # Adapt sigma
            sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma) / np.sqrt(d_sigma) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update mean
            m = m_new

            # If sigma is too small, we might be stuck; try a perturbation
            if sigma < 1e-5 and eval_count < budget:
                # Restart with larger sigma from current best
                m = best_x.copy()
                sigma = 1.0
                C = np.eye(dim)
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                sqrt_eigvals = np.sqrt(eigvals)
                C_sqrt_inv = eigvecs @ np.diag(1.0 / sqrt_eigvals) @ eigvecs.T
                p_sigma = np.zeros(dim)
                p_c = np.zeros(dim)

        return best_x, best_f
# EVOLVE-BLOCK-END
