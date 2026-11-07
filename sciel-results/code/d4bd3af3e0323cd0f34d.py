"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for BBOB f8 (Rosenbrock)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # safety margin before 300s limit
        start_time = time.time()

        # Counting wrapper
        call_count = [0]

        def counting_func(x):
            call_count[0] += 1
            return func(x)

        best_x = None
        best_f = np.inf

        # CMA-ES parameters
        lambda_ = 4 + int(3 * np.log(dim))  # population size
        # Initial sigma based on problem scale
        sigma0 = 2.0

        # We'll run multiple restarts if budget allows
        # Each restart gets a portion of the budget
        # For Rosenbrock in 5D, CMA-ES typically needs ~200-500 evaluations

        # Strategy: Use most of budget for first run, if stagnating, restart
        # But since we have a fixed budget, let's do a single well-tuned CMA-ES run
        # with adaptive sigma and restarts within the budget

        # Actually, let's implement a proper CMA-ES with the standard algorithm
        # and use the remaining budget for potential restarts

        # Let's do: first 400 evaluations for main CMA-ES, remaining for refinement
        main_budget = min(budget - 20, 400)
        refine_budget = budget - main_budget

        # Initial point: try center first, then random
        rng = np.random.default_rng(42)

        # Start at origin (center of box) which is often good for Rosenbrock
        x0 = np.zeros(dim)
        fx0 = counting_func(x0)
        if fx0 < best_f:
            best_f = fx0
            best_x = x0.copy()

        # If origin is bad, try a few random starts
        for trial in range(5):
            if call_count[0] >= budget:
                break
            x_trial = rng.uniform(-5, 5, dim)
            fx_trial = counting_func(x_trial)
            if fx_trial < best_f:
                best_f = fx_trial
                best_x = x_trial.copy()
            if call_count[0] >= 20:
                break

        # Use the best starting point for CMA-ES
        x_mean = best_x.copy()
        sigma = sigma0

        # CMA-ES state
        # Covariance matrix C, step sizes
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        # Standard CMA-ES parameters
        cmu = 2 / ((dim + 4) ** 2 + lambda_)
        ccs = (4 + dim / 4) / (dim + 4 + 2 * cmu)
        cc = (2 + 2 / (dim + 1)) / (dim + 2 + 2 / (dim + 1))
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        damps = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + cs
        mu = lambda_ // 2  # number of weights

        # Weights
        weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mu_eff = 1 / np.sum(weights ** 2)

        # Evaluation budget for this CMA-ES run
        evals_in_run = 0
        max_evals_run = main_budget - call_count[0]
        if max_evals_run < lambda_:
            max_evals_run = max(lambda_, min(main_budget - call_count[0], budget - call_count[0]))

        # Track progress for stagnation detection
        prev_best = best_f
        stagnation_counter = 0
        stagnation_threshold = 50  # no improvement in 50 evals -> consider restart

        while call_count[0] < budget and evals_in_run < max_evals_run:
            if time.time() > deadline:
                break

            # Check for stagnation
            if best_f < prev_best - 1e-10:
                prev_best = best_f
                stagnation_counter = 0
            else:
                stagnation_counter += lambda_

            # If stagnating and we have budget for a restart, do it
            if stagnation_counter > stagnation_threshold and call_count[0] < budget - 50:
                # Try a restart with new random center
                if best_x is not None:
                    # Perturb the best point
                    x_new = np.clip(best_x + sigma * rng.standard_normal(dim), -5, 5)
                    fx_new = counting_func(x_new)
                    if fx_new < best_f:
                        best_f = fx_new
                        best_x = x_new.copy()
                        x_mean = best_x.copy()
                        sigma = sigma0
                        C = np.eye(dim)
                        pc = np.zeros(dim)
                        ps = np.zeros(dim)
                        stagnation_counter = 0
                        prev_best = best_f
                        evals_in_run = 0
                        continue

            # Generate population
            B = rng.standard_normal((lambda_, dim))
            # Apply C^0.5 via Cholesky (recompute periodically for efficiency)
            try:
                L = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                L = np.eye(dim)

            # Transform and generate samples
            Y = np.array([x_mean + sigma * (B[i] @ L) for i in range(lambda_)])
            Y = np.clip(Y, -5, 5)

            # Evaluate population
            F = np.array([counting_func(Y[i]) for i in range(lambda_)])
            evals_in_run += lambda_

            # Update best
            idx_best = np.argmin(F)
            if F[idx_best] < best_f:
                best_f = F[idx_best]
                best_x = Y[idx_best].copy()

            # Sort by fitness
            order = np.argsort(F)
            Y_sorted = Y[order]
            F_sorted = F[order]

            # Update mean
            x_new_mean = np.sum(weights[:, None] * Y_sorted[:mu], axis=0)

            # Update covariance
            ys = (Y_sorted[:mu] - x_mean) / sigma
            C_new = (1 - cmu * cc * (2 - cc)) * C
            C_new += cmu * np.sum(weights[:, None, None] * (ys[:, :, None] * ys[:, None, :]), axis=0)

            # Update evolution paths
            pc = (1 - cc) * pc + cc * np.sqrt(mu_eff) * sigma * np.sum(weights[:, None] * ys, axis=0)
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) * (x_new_mean - x_mean) / sigma

            # Check for termination conditions
            # Sigma adaptation
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / np.sqrt(mu_eff) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            # Update state
            x_mean = x_new_mean
            C = C_new
            pc = pc
            ps = ps

            # Ensure C remains PSD
            try:
                eigvals, eigvecs = np.linalg.eigh(C)
                if np.min(eigvals) < 1e-12:
                    C = eigvecs @ np.diag(np.maximum(eigvals, 1e-12)) @ eigvecs.T
            except:
                C = np.eye(dim)

        # Refinement phase: use remaining budget with smaller steps around best
        if best_x is not None and call_count[0] < budget:
            refine_sigma = max(sigma * 0.1, 0.01)
            for _ in range(budget - call_count[0]):
                if time.time() > deadline:
                    break
                x_refine = np.clip(best_x + refine_sigma * rng.standard_normal(dim), -5, 5)
                fx_refine = counting_func(x_refine)
                if fx_refine < best_f:
                    best_f = fx_refine
                    best_x = x_refine.copy()
                    # Slightly increase step if improving
                    refine_sigma *= 1.1
                else:
                    refine_sigma *= 0.95

        return best_x, best_f
# EVOLVE-BLOCK-END
