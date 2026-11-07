"""Optimized black-box continuous optimizer for BBOB-style Schwefel f20."""
import numpy as np
import time


class Optimizer:
    """Hybrid optimizer: CMA-ES-like search with coordinate descent polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        # Start at center
        x = np.zeros(dim)
        fx = func(x)
        best_x = x.copy()
        best_fx = fx

        # CMA-ES-like parameters
        sigma = 1.5
        # Covariance matrix (start as identity)
        C = np.eye(dim)
        # Eigen decomposition cache
        eigvals, eigvecs = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(eigvals, 1e-10))

        # Step size adaptation
        successes = []
        window = 10

        # Phase 1: Global search with random restarts + local refinement
        calls_used = 1

        for iteration in range(budget - 1):
            if time.time() > deadline:
                break

            # Generate candidate
            z = rng.standard_normal(dim)
            # Transform using current covariance structure
            step = sigma * (eigvecs @ (D * z))
            y = np.clip(x + step, -5, 5)
            fy = func(y)
            calls_used += 1

            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()

            # Track success
            success = fy < fx
            successes.append(1 if success else 0)
            if len(successes) > window:
                successes.pop(0)

            if success:
                x = y
                fx = fy
                # Update covariance: rank-1 update
                diff = (y - x)  # This is step, but we use the actual move
                # Actually use the step direction
                step_vec = sigma * (eigvecs @ (D * z))
                # Simplified: just adapt sigma and occasionally update C
                sigma *= 1.1
            else:
                sigma *= 0.9

            # Keep sigma in reasonable range
            sigma = np.clip(sigma, 0.01, 5.0)

            # Periodic coordinate descent polish
            if calls_used % 30 == 0:
                x, fx = self._coordinate_descent(func, x, fx, dim, calls_used, deadline)
                best_fx = min(best_fx, fx)
                if fx < best_fx:
                    best_x = x.copy()
                    best_fx = fx

            # Occasional random restart to escape local minima
            if calls_used % 100 == 0 and fx > best_fx + 50:
                # Perturb from best
                x = best_x + rng.standard_normal(dim) * 0.5
                x = np.clip(x, -5, 5)
                fx = func(x)
                calls_used += 1

            # Re-diagonalize C occasionally (simplified: reset to identity if stuck)
            if calls_used % 50 == 0:
                # Check if we're making progress
                if len(successes) >= 5:
                    recent_success_rate = sum(successes[-5:]) / 5
                    if recent_success_rate < 0.2:
                        # Reset covariance to identity
                        C = np.eye(dim)
                        eigvals, eigvecs = np.linalg.eigh(C)
                        D = np.sqrt(np.maximum(eigvals, 1e-10))

        # Final coordinate descent polish
        if time.time() < deadline - 5:
            best_x, best_fx = self._coordinate_descent(func, best_x, best_fx, dim, calls_used, deadline)

        return best_x, best_fx

    def _coordinate_descent(self, func, x, fx, dim, calls_used, deadline):
        """Perform coordinate descent with adaptive step sizes."""
        x = x.copy()
        fx = func(x) if fx is None else fx

        step_sizes = np.ones(dim) * 0.5

        for _ in range(10):  # 10 passes of coordinate descent
            if time.time() > deadline:
                break

            for d in range(dim):
                if time.time() > deadline:
                    break

                # Try positive and negative steps
                best_local_fx = fx
                best_local_x = x.copy()

                for sign in [1, -1]:
                    for trial in range(5):
                        step = step_sizes[d] * (0.5 ** trial)
                        y = x.copy()
                        y[d] += sign * step
                        y = np.clip(y, -5, 5)
                        fy = func(y)

                        if fy < best_local_fx:
                            best_local_fx = fy
                            best_local_x = y.copy()

                if best_local_fx < fx:
                    x = best_local_x
                    fx = best_local_fx
                    step_sizes[d] *= 1.5  # Expand step size on success
                else:
                    step_sizes[d] *= 0.5  # Shrink step size on failure

                # Keep step sizes in bounds
                step_sizes[d] = np.clip(step_sizes[d], 0.01, 5.0)

        return x, fx
