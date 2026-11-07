"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES style optimizer with deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s kill
        dim = self.dim
        budget = self.budget

        # Initialize at origin (center of box)
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx
        call_count = 1

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        sigma = 2.0
        c1 = (2 + 2 / (dim + 1)) / ((dim + 1.3) ** 2 + lambda_ + 2)
        cmu = min(1 - c1, (2 + 2 / (dim + 1)) / ((dim + 1.3) ** 2 + lambda_ + 8))
        cs = (lambda_ + 2) / (dim + lambda_ + 5)
        ds = 1 + 2 * max(0, np.sqrt((lambda_ - 1) / (dim + 1)) - 1) + cs
        cc = (4 + dim / lambda_) / (dim + 4 + 2 * lambda_ / dim)
        chi = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Precompute eigendecomposition helper
        def update_cma(x, sigma, C, pc, ps, lambda_, dim):
            """Generate offspring and update CMA-ES state."""
            # Eigendecompose C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Generate random directions
            z = np.random.default_rng().standard_normal((lambda_, dim))
            # Transform
            y = (z @ (B * D).T).T  # shape (lambda_, dim)
            # Actually: y_i = B * D * z_i
            y = (z * D[None, :]) @ B.T

            # Offspring
            offsprings = x[None, :] + sigma * y
            offsprings = np.clip(offsprings, -5, 5)

            # Evaluate
            fvals = np.array([func(op) for op in offsprings])

            # Sort by fitness
            indices = np.argsort(fvals)
            y_sorted = y[indices]
            f_sorted = fvals[indices]

            # Update mean
            weights = np.zeros(lambda_)
            weights[:lambda_ // 2] = np.log(lambda_ / 2 + 0.5) - np.log(1 + np.arange(lambda_ // 2))
            weights = weights / weights.sum()

            new_x = x + sigma * (weights @ y_sorted)
            new_x = np.clip(new_x, -5, 5)

            # Update evolution paths
            pc = (1 - cs) * pc + cs * np.sqrt(lambda_ * c1 * (2 - c1)) * (weights @ y_sorted)
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * lambda_ * cmu) * (B @ (D @ (B.T @ y_sorted)))

            # Update covariance
            delta_sig = 1.0 - np.exp(-np.linalg.norm(ps) / (chi * lambda_))
            C = (1 - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + (1 - (1 - cs) ** 2) * c1 * C) \
                + cmu * (y_sorted[:lambda_ // 2].T * weights[:lambda_ // 2] @ y_sorted[:lambda_ // 2])

            # Symmetrize
            C = (C + C.T) / 2

            # Clip eigenvalues
            eigvals2, eigvecs2 = np.linalg.eigh(C)
            eigvals2 = np.clip(eigvals2, 1e-20, 1e10)
            C = (eigvecs2 * eigvals2[None, :]) @ eigvecs2.T
            C = (C + C.T) / 2

            return new_x, C, pc, ps, f_sorted[0], indices[0]

        # Use a local RNG
        rng = np.random.default_rng(42)

        # Override the random generation to use our rng
        # We'll inline the CMA-ES loop for speed

        while call_count < budget:
            # Check deadline
            if time.time() > deadline - 5.0:
                break

            # Eigendecompose C
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.clip(eigvals, 1e-20, None)
            B = eigvecs
            D = np.sqrt(eigvals)

            # Generate random directions
            z = rng.standard_normal((lambda_, dim))
            y = (z * D[None, :]) @ B.T

            # Offspring
            offsprings = x[None, :] + sigma * y
            offsprings = np.clip(offsprings, -5, 5)

            # Evaluate all offspring
            fvals = np.empty(lambda_)
            for i in range(lambda_):
                if time.time() > deadline - 2.0:
                    break
                fvals[i] = func(offsprings[i])
                call_count += 1
            else:
                # All evaluated
                pass
            if time.time() > deadline - 2.0:
                # Check best so far
                valid = np.isfinite(fvals)
                if valid.any():
                    min_idx = np.argmin(np.where(valid, fvals, np.inf))
                    if fvals[min_idx] < best_fx:
                        best_fx = fvals[min_idx]
                        best_x = offsprings[min_idx].copy()
                break

            # Sort by fitness
            indices = np.argsort(fvals)
            y_sorted = y[indices]
            f_sorted = fvals[indices]

            # Update best
            if f_sorted[0] < best_fx:
                best_fx = f_sorted[0]
                best_x = offsprings[indices[0]].copy()

            # Update mean
            weights = np.zeros(lambda_)
            weights[:lambda_ // 2] = np.log(lambda_ / 2 + 0.5) - np.log(1 + np.arange(lambda_ // 2))
            weights = weights / weights.sum()

            new_x = x + sigma * (weights @ y_sorted)
            new_x = np.clip(new_x, -5, 5)

            # Update evolution paths
            pc = (1 - cs) * pc + cs * np.sqrt(lambda_ * c1 * (2 - c1)) * (weights @ y_sorted)
            ps = (1 - ds) * ps + np.sqrt(ds * (2 - ds) * lambda_ * cmu) * (B @ (D @ (B.T @ y_sorted)))

            # Update covariance
            C = (1 - c1 - cmu) * C \
                + c1 * (np.outer(pc, pc) + (1 - (1 - cs) ** 2) * c1 * C) \
                + cmu * (y_sorted[:lambda_ // 2].T * weights[:lambda_ // 2] @ y_sorted[:lambda_ // 2])

            # Symmetrize and clip
            C = (C + C.T) / 2
            eigvals2, eigvecs2 = np.linalg.eigh(C)
            eigvals2 = np.clip(eigvals2, 1e-20, 1e10)
            C = (eigvecs2 * eigvals2[None, :]) @ eigvecs2.T
            C = (C + C.T) / 2

            # Adaptive sigma
            delta_sig = 1.0 - np.exp(-np.linalg.norm(ps) / (chi * lambda_))
            sigma = sigma * np.exp(cs / ds * (np.linalg.norm(ps) / chi - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            x = new_x

        return best_x, best_fx
# EVOLVE-BLOCK-END
