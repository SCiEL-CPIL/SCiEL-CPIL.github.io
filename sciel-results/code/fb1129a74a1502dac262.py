"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES for Weierstrass f16."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        n = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin

        best_x = None
        best_f = np.inf

        # Phase 1: Initial exploration - sample random points to find good basins
        n_explore = min(20, budget // 5)
        candidates = []
        for i in range(n_explore):
            x = rng.uniform(-4, 4, n)
            fx = func(x)
            candidates.append((fx, x))
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        # Sort candidates by fitness
        candidates.sort(key=lambda t: t[0])

        # Phase 2: CMA-ES from the best starting points
        # Use remaining budget for CMA-ES
        remaining_budget = budget - n_explore

        # Try CMA-ES from top 3 starting points
        n_starts = min(3, len(candidates))
        budget_per_start = max(10, remaining_budget // n_starts)

        for start_idx in range(n_starts):
            if time.time() > deadline:
                break

            x0 = candidates[start_idx][1].copy()
            f0 = candidates[start_idx][0]

            # CMA-ES parameters
            pop_size = max(4, n + 2)
            sigma = 1.0  # initial step size
            mean = x0.copy()
            C = np.eye(n)  # covariance matrix
            C_inv = np.eye(n)
            step_size = 1.0  # for covariance update

            # CMA-ES state
            history = []  # (f, x) for successful steps
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)
            c_sigma = 1 / (n + 4)
            c_c = (2 + 2 / (n + 1)) / (n + 2)
            mu = pop_size // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1 / (weights @ weights)

            calls_in_this_start = 0
            stagnation = 0
            best_local_f = f0
            best_local_x = x0.copy()

            for gen in range(budget_per_start):
                if time.time() > deadline:
                    break

                # Generate population
                Y = np.zeros((pop_size, n))
                F = np.zeros(pop_size)

                for i in range(pop_size):
                    if time.time() > deadline:
                        break
                    # Sample from N(mean, sigma^2 * C)
                    z = rng.standard_normal(n)
                    # Cholesky decomposition of C
                    try:
                        L = np.linalg.cholesky(C)
                    except np.linalg.LinAlgError:
                        L = np.eye(n)
                    y = mean + sigma * (L @ z)
                    y = np.clip(y, -5, 5)
                    F[i] = func(y)
                    Y[i] = y
                    calls_in_this_start += 1

                    if F[i] < best_local_f:
                        best_local_f = F[i]
                        best_local_x = y.copy()
                        if F[i] < best_f:
                            best_f = F[i]
                            best_x = y.copy()

                if calls_in_this_start >= budget_per_start:
                    break

                # Sort by fitness
                order = np.argsort(F)
                Y = Y[order]
                F = F[order]

                # Update mean
                new_mean = Y[:mu] @ weights[:, np.newaxis]
                new_mean = np.clip(new_mean, -5, 5)

                # Update evolution paths
                diff = (new_mean - mean) / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff
                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff

                # Check for sigma adaptation
                if np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (gen + 1)) < (1.4 + 2 / (n + 1)) * np.sqrt(mu_eff / (n + 1)):
                    sigma *= np.exp(c_sigma / (n + 1) * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - np.sqrt(mu_eff)))

                # Update covariance
                delta_mu = (1 / sigma) * (new_mean - mean)
                C = (1 - c_c * (2 - c_c) * mu_eff) * C + c_c * (np.outer(delta_mu, delta_mu) + 
                     (1 - c_c) * (np.outer(p_c, p_c) - C))

                # Ensure C is positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T

                mean = new_mean

                # Check stagnation
                if F[0] >= best_local_f * (1 + 1e-10) and gen > 0:
                    stagnation += 1
                else:
                    stagnation = 0

                # Restart if stagnating too long
                if stagnation > 15:
                    break

            if best_local_f < best_f:
                best_f = best_local_f
                best_x = best_local_x.copy()

        # Phase 3: Final local refinement with (1+1)-ES from best point
        if best_x is not None and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.5
            calls_left = budget - (n_explore + calls_in_this_start if 'calls_in_this_start' in dir() else n_explore)

            # Count actual calls used
            total_calls = n_explore
            for start_idx in range(n_starts):
                if time.time() > deadline:
                    break
                # We already counted these in the loop above, so just do a small refinement
                pass

            # Simple (1+1)-ES refinement
            remaining = budget - n_explore - sum(min(budget_per_start, 100) for _ in range(n_starts))
            remaining = max(10, remaining)

            for _ in range(remaining):
                if time.time() > deadline:
                    break
                y = np.clip(x + sigma * rng.standard_normal(n), -5, 5)
                fy = func(y)
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
