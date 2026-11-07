"""Black-box optimizer for BBOB f11 (Discus) on [-5,5]^5 with 500 evaluations."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s kill

        # Start with a few diverse initial points to explore the space
        # Use a small set of starting points spread across the domain
        rng = np.random.default_rng(42)

        # Initial exploration: sample a few points
        n_init = min(10, budget // 5)
        init_points = np.array([rng.uniform(-5, 5, dim) for _ in range(n_init)])
        init_vals = np.array([func(p) for p in init_points])

        best_idx = np.argmin(init_vals)
        x = init_points[best_idx].copy()
        fx = init_vals[best_idx]
        calls_used = n_init

        if calls_used >= budget:
            return x, fx

        # Adaptation: use (1+1)-ES with step-size control
        # For discus functions, the landscape is highly anisotropic.
        # We'll use a covariance matrix adaptation approach.

        sigma = 1.0  # initial step size
        C = np.eye(dim)  # covariance matrix
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)  # square root of C
        C_sqrt_inv = np.eye(dim)

        # Evolution path for covariance adaptation
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)
        c_sigma = 1.0 / (4 * dim + 4)
        c_c = (2 * (dim + 1.7)) / ((dim + 2) * (4 * dim + 4))
        d_sigma = 1 + 2 * max(0, (dim - 1) / (dim + 2)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))

        # Step size adaptation
        h_sigma = 1 + 2 * max(0, (dim - 1) / (dim + 2)) + c_sigma

        # Remaining budget
        remaining = budget - calls_used

        # We'll do (1+1)-ES with CMA-like updates for the covariance
        # But to keep it simple and fast, we'll use a simplified version

        # Actually, let's use a more standard approach:
        # CMA-ES with population size 1 (i.e., (1+1)-ES with covariance adaptation)
        # This is essentially the same as the seed but with better step size control.

        # Let me implement a proper simplified CMA-ES:

        # Reset for CMA-ES
        x = x.copy()
        fx = fx
        sigma = 1.0
        C = np.eye(dim)
        C_inv = np.eye(dim)
        C_sqrt = np.eye(dim)
        C_sqrt_inv = np.eye(dim)
        p_sigma = np.zeros(dim)
        p_c = np.zeros(dim)

        # Precompute
        c_sigma = 1.0 / (4 * dim + 4)
        c_c = (2 * (dim + 1.7)) / ((dim + 2) * (4 * dim + 4))
        d_sigma = 1 + 2 * max(0, (dim - 1) / (dim + 2)) + c_sigma
        chi_n = np.sqrt(dim) * (1 - 1/(4*dim) + 1/(21*dim**2))
        h_sigma = 1 + 2 * max(0, (dim - 1) / (dim + 2)) + c_sigma

        # For (1+1)-ES with CMA updates, we sample one point per iteration
        # This is slow but with only 500 budget in 5D it should be fine

        for it in range(remaining):
            if time.time() > deadline:
                break

            # Sample
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * (C_sqrt @ z), -5, 5)
            fy = func(y)
            calls_used += 1

            # Compute delta
            delta = y - x

            # Update evolution paths
            p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * dim) * (C_sqrt_inv @ delta) / sigma
            p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * dim) * (C_sqrt_inv @ delta) / sigma

            # Update step size
            h = (1 - (1 - c_sigma)**(it + 1)) * chi_n
            if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma)**(2*(it+1))) < (1.4 + 2/(dim+1)) * h:
                sigma *= np.exp(c_sigma * (np.linalg.norm(p_sigma)/chi_n - 1))

            # Update covariance
            if fy < fx:
                # Accept new point
                x_new = y
                fx_new = fy
                # Update C
                v = (C_sqrt_inv @ delta) / sigma
                C = (1 + c_c * (2 - c_c) - c_c) * C + c_c * np.outer(v, v) + c_c * (2 - c_c) * (np.outer(p_c, p_c) - C)
                # Ensure C is positive definite
                C = (C + C.T) / 2
                # Update C_inv and C_sqrt via Cholesky
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt = L
                    C_sqrt_inv = np.linalg.inv(L)
                    C_inv = C_sqrt_inv.T @ C_sqrt_inv
                except np.linalg.LinAlgError:
                    # Fallback: use eigenvalue decomposition
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                    C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                    C_inv = C_sqrt_inv.T @ C_sqrt_inv
                x = x_new
                fx = fx_new
            else:
                # Reject, but still update C
                v = (C_sqrt_inv @ delta) / sigma
                C = (1 + c_c * (2 - c_c) - c_c) * C + c_c * (2 - c_c) * (np.outer(p_c, p_c) - C)
                C = (C + C.T) / 2
                try:
                    L = np.linalg.cholesky(C)
                    C_sqrt = L
                    C_sqrt_inv = np.linalg.inv(L)
                    C_inv = C_sqrt_inv.T @ C_sqrt_inv
                except np.linalg.LinAlgError:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.maximum(eigvals, 1e-10)
                    C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals))
                    C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals))
                    C_inv = C_sqrt_inv.T @ C_sqrt_inv

        return x, fx
# EVOLVE-BLOCK-END
