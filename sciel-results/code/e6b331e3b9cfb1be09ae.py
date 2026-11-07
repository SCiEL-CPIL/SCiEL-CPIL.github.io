"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with time guard for BBOB-style functions."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)

        def time_ok():
            return time.time() < deadline

        # --- Phase 1: Coarse multi-start exploration to find promising basins ---
        # Use a small number of random starts to identify good regions
        n_starts = 8
        start_points = []
        start_fvals = []
        for _ in range(n_starts):
            if not time_ok():
                break
            x = rng.uniform(-5, 5, n)
            fx = func(x)
            start_points.append(x)
            start_fvals.append(fx)

        # Sort by fitness, take the best few as CMA-ES seeds
        order = np.argsort(start_fvals)
        best_starts = [start_points[i] for i in order[:3]]
        best_fval = start_fvals[order[0]]
        best_x = start_points[order[0]].copy()

        # --- Phase 2: CMA-ES from the best start points ---
        # CMA-ES parameters
        pop_size = max(4, int(4 + 3 * np.log(n)))  # lambda
        # Initial sigma: based on spread of good starts or a reasonable default
        if len(best_starts) > 1:
            spread = np.std([np.linalg.norm(s - best_starts[0]) for s in best_starts[1:]], axis=0)
            sigma = max(0.5, min(2.0, float(np.mean(spread)) * 0.5))
        else:
            sigma = 1.5

        # CMA-ES state
        mean = best_starts[0].copy()
        C = np.eye(n)
        C_inv = np.eye(n)
        C_sqrt = np.eye(n)
        C_sqrt_inv = np.eye(n)
        ps = np.zeros(n)
        pc = np.zeros(n)
        cs = 1 + 2 * max(0, (np.sqrt((pop_size - 2) / (n + 2)) - 1) / (pop_size + 2))
        cmu = min(0.5, 2 / ((n + 2)**2 + pop_size))
        c1 = (4 + pop_size / (n + 4)) / ((n + 2)**2 + 4)
        cmu = min(c1 * 2 / 3, 0.5)
        cc = (4 + pop_size / (n + 4)) / (n + 4)
        damps = 1 + 2 * max(0, (np.sqrt((pop_size - 2) / (n + 2)) - 1) / (pop_size + 2)) + cs
        # Actually let me use standard CMA-ES formulas
        mu = pop_size // 2
        weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        weights /= weights.sum()
        mueff = 1 / np.sum(weights**2)
        c_sigma = (mueff + 2) / (n + mueff + 5)
        c_c = (4 + mueff / n) / (n + 4 + mueff / n)
        c_1 = 2 / ((n + 1.3)**2 + mueff)
        c_mu = min(1 - c_1, 2 * (mueff - 2 + 2 / mueff) / ((n + 2)**2 + mueff))
        d_sigma = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1) - 1)) + c_sigma

        # Cholesky decomposition helper
        def chol_update(C, C_sqrt, step):
            """Update C and its Cholesky factor."""
            C = C + step
            # Ensure symmetry
            C = (C + C.T) / 2
            try:
                C_sqrt = np.linalg.cholesky(C)
            except np.linalg.LinAlgError:
                # Regularize if not positive definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = eigvecs @ np.diag(eigvals) @ eigvecs.T
                C_sqrt = np.linalg.cholesky(C)
            return C, C_sqrt

        # Generate population using CMA-ES
        def generate_population(mean, sigma, C_sqrt, n, pop_size, rng):
            z = rng.standard_normal((pop_size, n))
            # Transform z using C_sqrt
            y = (z @ C_sqrt.T) * sigma + mean
            return y

        # Evaluate and update
        evals_used = n_starts
        for gen in range(budget - n_starts):
            if not time_ok():
                break
            if evals_used >= budget:
                break

            # Generate population
            z = rng.standard_normal((pop_size, n))
            pop = (z @ C_sqrt.T) * sigma + mean
            # Clip to bounds
            pop = np.clip(pop, -5, 5)

            # Evaluate
            fvals = np.zeros(pop_size)
            for i in range(pop_size):
                if not time_ok() or evals_used >= budget:
                    break
                fvals[i] = func(pop[i])
                evals_used += 1
                if fvals[i] < best_fval:
                    best_fval = fvals[i]
                    best_x = pop[i].copy()

            # Only proceed if we evaluated enough
            valid = np.isfinite(fvals)
            if valid.sum() < mu:
                break

            # Sort by fitness
            idx = np.argsort(fvals)
            y_sorted = pop[idx]
            f_sorted = fvals[idx]

            # Update mean
            w = weights[:mu]
            mean_new = np.sum(w[:, None] * y_sorted[:mu], axis=0)

            # Update evolution paths
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mueff) / sigma * (C_sqrt_inv @ (mean_new - mean))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)):
                pc = (1 - c_c) * pc + c_c / sigma * (mean_new - mean)

            # Update covariance
            delta_mu = 0
            if np.linalg.norm(ps)**2 / d_sigma > (1 + 2 / (n + 1)) * (1 - (1 - c_sigma)**(2 * (gen + 1))):
                delta_mu = c_mu * (1 + c_sigma) * np.outer(ps, ps)

            hsig = 1.0
            if np.linalg.norm(pc) / np.sqrt(1 - (1 - c_c)**(2 * (gen + 1))) < (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)):
                hsig = 1.0

            rank_1 = np.outer(mean_new - mean, mean_new - mean)
            rank_mu = np.zeros((n, n))
            for i in range(mu):
                yi = (y_sorted[i] - mean) / sigma
                rank_mu += w[i] * np.outer(yi, yi)

            C = (1 + c_1 * hsig + c_mu - c_1 - c_mu) * C + c_1 * (rank_1 + (1 - hsig + delta_mu) * np.outer(pc, pc)) + c_mu * rank_mu
            C, C_sqrt = chol_update(C, C_sqrt, 0)

            # Update sigma
            sigma = sigma * np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / np.sqrt(2 * n) - 1))
            sigma = np.clip(sigma, 1e-10, 10.0)

            mean = mean_new

            # Update C_sqrt_inv
            try:
                C_sqrt_inv = np.linalg.inv(C_sqrt)
            except np.linalg.LinAlgError:
                C_sqrt_inv = np.linalg.pinv(C_sqrt)

            # If sigma is too small, try a restart from another good start
            if sigma < 0.01 and gen > 20:
                # Restart from second-best start if available
                if len(best_starts) > 1:
                    mean = best_starts[1].copy()
                    sigma = 1.0
                    C = np.eye(n)
                    C_sqrt = np.eye(n)
                    C_sqrt_inv = np.eye(n)
                    ps = np.zeros(n)
                    pc = np.zeros(n)
                else:
                    break

        return best_x, best_fval
# EVOLVE-BLOCK-END
