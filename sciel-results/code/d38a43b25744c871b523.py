"""Optimizer for BBOB f14 (different powers) on [-5,5]^5."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """CMA-ES with population size 4 for non-separable optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 330s kill

        rng = np.random.default_rng(42)

        # Initialize at origin with large sigma to cover the box
        x = np.zeros(dim)
        fx = func(x)
        calls = 1

        # CMA-ES parameters
        sigma = 5.0  # large initial step to explore the box
        pop_size = 4
        # Covariance matrix (identity initially)
        C = np.eye(dim)
        # Evolution paths
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        
        # Learning rates
        c_sigma = (pop_size + 2) / (dim + 4)
        c_c = (4 + pop_size / dim) / (dim + 4 + 2 * pop_size / dim)
        c_1 = 2 / ((dim + 1.3) ** 2 + pop_size)
        c_mu = min(1 - c_1, 2 * ((pop_size / 2 + 1) - dim - 2) / ((dim + 2) ** 2 + pop_size))
        d_sigma = 1 + c_sigma + 2 / max(dim - 1, 1)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))
        
        # Weights for the mean update
        weights = np.log(pop_size / 2 + 1) - np.log(np.arange(1, pop_size + 1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)
        
        best_x = x.copy()
        best_f = fx

        while calls < budget:
            if time.time() > deadline:
                break

            # Generate population
            # Use eigendecomposition for sampling (or Cholesky if possible)
            try:
                # For small dim, eigendecomposition is fast enough
                eigvals, eigvecs = np.linalg.eigh(C)
                # Ensure positive eigenvalues
                eigvals = np.maximum(eigvals, 1e-12)
                # Sample: x + sigma * C^{1/2} * z
                # C^{1/2} = eigvecs * diag(sqrt(eigvals))
                z = rng.standard_normal((pop_size, dim))
                # Sample in transformed space
                samples = x[None, :] + sigma * (z @ (eigvecs * np.sqrt(eigvals)))
            except np.linalg.LinAlgError:
                # Fallback to diagonal sampling if matrix issues
                samples = x[None, :] + sigma * np.sqrt(np.diag(C))[None, :] * z

            # Evaluate population
            fvals = np.array([func(np.clip(s, -5, 5)) for s in samples])
            calls += pop_size

            # Update best
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = np.clip(samples[idx_min], -5, 5)

            # Sort samples by fitness
            sort_idx = np.argsort(fvals)
            samples_sorted = samples[sort_idx]
            fvals_sorted = fvals[sort_idx]

            # Compute weighted mean
            mean = np.sum(weights[:, None] * samples_sorted, axis=0)
            
            # Update evolution paths
            # p_sigma update
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * (
                (C @ (sigma * (mean - x))) / sigma
            )
            
            # p_c update
            norm_p_sigma = np.linalg.norm(p_sigma)
            h_sigma = (norm_p_sigma / np.sqrt(1 - (1 - c_sigma) ** (2 * (calls - dim)))) < (
                1.4 + 2 / (dim + 1)
            ) * chi_n
            p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mu_eff) * (
                (C @ (mean - x)) / sigma
            )
            
            # Update covariance matrix C
            diff = (samples_sorted - mean) / sigma
            C = (
                (1 + c_1 * h_sigma * (2 - h_sigma) - c_1 - c_mu) * C
                + c_1 * np.outer(p_c, p_c)
                + c_mu * np.sum(weights[:, None, None] * diff[:, None, :] * diff[:, None, :], axis=0)
            )
            
            # Symmetrize C
            C = (C + C.T) / 2
            
            # Update sigma
            sigma *= np.exp(
                c_sigma / d_sigma * (norm_p_sigma / chi_n - 1)
            )
            
            # Update mean
            x = mean
            
            # Clip x to box
            x = np.clip(x, -5, 5)

        return best_x, best_f
# EVOLVE-BLOCK-END
