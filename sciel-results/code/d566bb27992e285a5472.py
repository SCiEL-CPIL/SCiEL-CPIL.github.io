"""Black-box continuous optimizer for BBOB f4 (Büche-Rastrigin)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit
        dim = self.dim
        budget = self.budget

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        mu = lambda_ // 2  # number of parents
        # Weights for recombination
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1.0 / np.sum(weights ** 2)

        # Step size control parameters
        c_sigma = (mueff + 2) / (dim + mueff + 5)
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + c_sigma
        c_c = (4 + mueff / dim) / (dim + 4 + 2 * mueff / dim)
        c_1 = 2 / (dim ** 1.7 + 10)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 1 / mueff) / (dim ** 2 + 18))

        # Boundary handling: reflection
        def clip_and_reflect(x):
            # Reflect points that are outside [-5, 5]
            out = x.copy()
            for i in range(dim):
                if out[i] < -5:
                    out[i] = -5 + (-5 - out[i]) * 0.5
                elif out[i] > 5:
                    out[i] = 5 - (out[i] - 5) * 0.5
            # Clip to be safe
            return np.clip(out, -5, 5)

        best_x = None
        best_f = np.inf
        calls_used = 0

        # Try multiple restarts with different initializations
        n_restarts = 3
        calls_per_restart = budget // n_restarts

        for restart in range(n_restarts):
            if calls_used >= budget:
                break
            if time.time() > deadline:
                break

            # Initialize center
            if restart == 0:
                x_center = np.zeros(dim)
            else:
                # Start near the best found so far or random
                if best_x is not None and restart == 1:
                    x_center = best_x.copy()
                else:
                    x_center = np.random.uniform(-4, 4, dim)

            sigma = 2.0
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)
            C = np.eye(dim)
            C_sqrt_inv = np.eye(dim)  # Initially identity, updated via Cholesky

            # Evaluate initial center
            fx_center = func(x_center)
            calls_used += 1
            if fx_center < best_f:
                best_f = fx_center
                best_x = x_center.copy()

            # Track history of step sizes
            sigma_history = []

            for it in range(calls_per_restart):
                if calls_used >= budget:
                    break
                if time.time() > deadline:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    z = np.random.standard_normal(dim)
                    # Apply C matrix (approximate with current C_sqrt_inv)
                    y = x_center + sigma * C_sqrt_inv @ z
                    y = clip_and_reflect(y)
                    offspring.append(y)

                # Evaluate offspring
                f_offspring = []
                for y in offspring:
                    fy = func(y)
                    calls_used += 1
                    f_offspring.append(fy)
                    if fy < best_f:
                        best_f = fy
                        best_x = y.copy()

                f_offspring = np.array(f_offspring)
                offspring = np.array(offspring)

                # Sort by fitness
                order = np.argsort(f_offspring)
                offspring_sorted = offspring[order]
                f_sorted = f_offspring[order]

                # Weighted mean
                y_mean = np.sum(weights[:, None] * offspring_sorted[:mu], axis=0)

                # Update path for sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mueff) * C_sqrt_inv @ (y_mean - x_center) / sigma

                # Hessian-free update of C
                # Delta sigma
                delta_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (2 * (it + 1))) - 1
                sigma *= np.exp(c_sigma / d_sigma * delta_sigma)
                sigma = np.clip(sigma, 1e-10, 10.0)

                # Update path for C
                h_sigma = np.linalg.norm(p_sigma) / np.sqrt(1 - (1 - c_sigma) ** (it + 1)) < (1.4 + 2 / (dim + 1)) * c_sigma
                p_c = (1 - c_c) * p_c + h_sigma * np.sqrt(c_c * (2 - c_c) * mueff) * (y_mean - x_center) / sigma

                # Rank-mu update of C
                yw = (y_mean - x_center) / sigma
                C = (1 + c_1 + c_mu - c_1 / dim - c_mu / dim) * C \
                    + c_1 * (np.outer(p_c, p_c) + (1 - h_sigma) * c_c * (2 - c_c) * C) \
                    + c_mu * np.sum(weights[:, None, None] * np.outer(offspring_sorted[:mu] - x_center, offspring_sorted[:mu] - x_center) / sigma ** 2, axis=0)

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update inverse square root using eigen decomposition
                try:
                    eigvals, eigvecs = np.linalg.eigh(C)
                    eigvals = np.clip(eigvals, 1e-12, None)
                    C_sqrt_inv = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
                except np.linalg.LinAlgError:
                    C_sqrt_inv = np.eye(dim)

                # Update center
                x_center = y_mean
                x_center = clip_and_reflect(x_center)

                # Evaluate new center if it's different
                if not np.allclose(x_center, offspring_sorted[0]):
                    fx_new = func(x_center)
                    calls_used += 1
                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_center.copy()

            # After each restart, check if we should continue
            if calls_used >= budget:
                break

        # Use remaining budget for local search around best
        remaining = budget - calls_used
        if remaining > 0 and best_x is not None and time.time() < deadline:
            # Local refinement with decreasing step size
            x_local = best_x.copy()
            f_local = best_f
            step = 0.5
            for _ in range(remaining):
                if time.time() > deadline:
                    break
                for _ in range(dim):  # Try each dimension
                    if time.time() > deadline:
                        break
                    for direction in [1, -1]:
                        x_try = x_local.copy()
                        x_try[_] += direction * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls_used += 1
                        if f_try < f_local:
                            x_local = x_try
                            f_local = f_try
                            best_f = f_local
                            best_x = x_local.copy()
                            step *= 1.2
                        else:
                            step *= 0.7
                step *= 0.9

        return best_x, best_f
# EVOLVE-BLOCK-END
