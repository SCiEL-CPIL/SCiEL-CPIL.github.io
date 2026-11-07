"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """
    Hybrid optimizer: CMA-ES with multiple restarts, time guard, and
    aggressive early exploration for AOCC maximization.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 330s limit
        n = self.dim
        total_budget = self.budget

        best_x = None
        best_f = np.inf
        calls_made = 0

        rng = np.random.default_rng(42)

        def check_time():
            return time.time() > deadline

        def eval_point(x):
            nonlocal calls_made, best_f, best_x
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Initial exploration with a small set of random points
        n_init = min(20, total_budget)
        for i in range(n_init):
            if check_time() or calls_made >= total_budget:
                break
            x = rng.uniform(-5, 5, n)
            eval_point(x)

        if best_x is None:
            x0 = np.zeros(n)
            best_f = eval_point(x0)
            best_x = x0.copy()

        # Phase 2: CMA-ES with restarts
        # We'll do several CMA-ES runs, each starting from a perturbation of the best
        max_restarts = 8
        remaining = total_budget - calls_made
        budget_per_restart = max(10, remaining // max_restarts)

        for restart in range(max_restarts):
            if calls_made >= total_budget or check_time():
                break

            # Initialize CMA-ES
            if restart == 0:
                x_mean = best_x.copy()
                sigma = 1.0
            else:
                # Restart from best with random perturbation
                x_mean = best_x + rng.normal(0, 0.3, n)
                x_mean = np.clip(x_mean, -5, 5)
                sigma = 0.5 + 0.5 * rng.random()

            # CMA-ES parameters
            pop_size = max(4, int(4 + 2 * np.log(n)))  # standard CMA-ES population
            # Ensure we don't exceed budget
            if pop_size > budget_per_restart // 2:
                pop_size = max(4, budget_per_restart // 4)

            # Initialize covariance and step size
            C = np.eye(n)
            C_inv = np.eye(n)
            C_sqrt = np.eye(n)
            step_size = sigma

            # Adaptation parameters
            c_sigma = (pop_size + 2) / (n + 4)
            d_sigma = 1 + 2 * max(0, np.sqrt((pop_size - 1) / (n + 1)) - 1) + c_sigma
            c_c = (4 + pop_size / n) / (n + 4 + 2 * pop_size / n)
            c_1 = 2 / ((n + 1.3) ** 2 + pop_size)
            c_mu = min(1 - c_1, 2 * (mu_eff - 2) / ((n + 2) ** 2 + mu_eff))

            # Weights for selection
            weights = np.log(pop_size / 2 + 0.5) - np.log(np.arange(1, pop_size + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights[:pop_size // 2] ** 2)

            # Path accumulators
            p_sigma = np.zeros(n)
            p_c = np.zeros(n)

            stagnation = 0
            max_stagnation = 30
            prev_best = best_f

            for gen in range(budget_per_restart):
                if calls_made >= total_budget or check_time():
                    break

                # Generate population
                xs = []
                fs = []
                for i in range(pop_size):
                    if calls_made >= total_budget or check_time():
                        break
                    # Sample from multivariate normal
                    z = rng.standard_normal(n)
                    # Transform using C_sqrt approximation (simplified: use diagonal)
                    # For simplicity and speed, use diagonal covariance initially
                    x_new = x_mean + step_size * C_sqrt @ z
                    x_new = np.clip(x_new, -5, 5)
                    f_new = eval_point(x_new)
                    xs.append(x_new)
                    fs.append(f_new)

                if len(xs) < 2:
                    break

                xs = np.array(xs)
                fs = np.array(fs)

                # Sort by fitness
                order = np.argsort(fs)
                xs = xs[order]
                fs = fs[order]

                # Update best if improved
                if fs[0] < prev_best:
                    prev_best = fs[0]
                    stagnation = 0
                else:
                    stagnation += 1

                # Weighted mean
                w = weights[:len(xs)]
                x_new_mean = np.sum(w[:, None] * xs, axis=0)

                # Update sigma
                x_old_mean = x_mean
                diff = x_new_mean - x_old_mean
                norm_diff = np.linalg.norm(diff)
                if norm_diff > 1e-10:
                    p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * diff / (step_size * norm_diff)
                sigma_norm = np.linalg.norm(p_sigma)
                h_sigma = sigma_norm < (1 + 5 / (pop_size + 1)) * np.sqrt((1 - (1 - c_sigma) ** (gen + 1)))
                step_size *= np.exp(c_sigma / d_sigma * (sigma_norm / np.sqrt(mu_eff) - 1))
                step_size = np.clip(step_size, 1e-10, 5.0)

                # Update covariance
                if h_sigma:
                    p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * diff / step_size
                else:
                    p_c = (1 - c_c) * p_c

                # Covariance update (simplified: use diagonal approximation for speed)
                # For full CMA-ES, we'd update C properly, but for speed we use a simplified version
                diff_mean = xs - x_new_mean
                C = (1 - c_1 - c_mu) * C + c_1 * (np.outer(p_c, p_c) + (not h_sigma) * c_c * (2 - c_c) * np.eye(n))
                for i in range(len(xs)):
                    C += c_mu * weights[i] * np.outer(diff_mean[i], diff_mean[i])

                # Ensure C is positive definite (simplified: add small regularization)
                C = (C + C.T) / 2
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-10)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_inv = eigvecs @ np.diag(1.0 / eigvals) @ eigvecs.T
                C_sqrt = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T

                x_mean = x_new_mean

                # If stagnation is high, break early to restart
                if stagnation >= max_stagnation:
                    break

        # Final polish: local search around the best point
        if calls_made < total_budget and not check_time():
            # Use a fine-grained local search
            local_sigma = 0.1
            for _ in range(min(50, total_budget - calls_made)):
                if calls_made >= total_budget or check_time():
                    break
                x_try = best_x + local_sigma * rng.standard_normal(n)
                x_try = np.clip(x_try, -5, 5)
                f_try = func(x_try)
                calls_made += 1
                if f_try < best_f:
                    best_f = f_try
                    best_x = x_try.copy()
                    local_sigma *= 0.9
                else:
                    local_sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
