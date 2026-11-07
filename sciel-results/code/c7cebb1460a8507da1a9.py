"""Optimizer for BBOB-style f17 (Schaffers F7) with hidden shift/rotation/offset."""
import numpy as np
import time


class Optimizer:
    """CMA-ES with deadline guard and adaptive restart."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 320.0  # safety margin before 330s limit

        best_x = None
        best_f = np.inf
        total_budget = self.budget
        dim = self.dim

        # Use multiple restarts of CMA-ES with different initializations
        # to handle multimodality of Schaffers F7

        rng = np.random.default_rng(42)

        # First, do a quick random search to find a good starting point
        # This helps with the hidden shift/rotation
        n_random = 20
        best_rand_x = None
        best_rand_f = np.inf

        for _ in range(n_random):
            x = rng.uniform(-5, 5, dim)
            fx = func(x)
            if fx < best_rand_f:
                best_rand_f = fx
                best_rand_x = x.copy()

        if best_rand_x is not None:
            best_x = best_rand_x
            best_f = best_rand_f

        # Now run CMA-ES from the best random point
        # We'll do a simple CMA-ES implementation

        def run_cmaes(x0, fx, remaining_budget, rng):
            nonlocal best_x, best_f

            n = dim
            # CMA-ES parameters
            lam = max(4, int(4 + 3 * np.log(n)))  # population size
            sigma = 1.0  # initial step size

            # CMA-ES state
            mean = x0.copy()
            C = np.eye(n)
            Cinv = np.eye(n)
            Csqrt = np.eye(n)
            Cinv_sqrt = np.eye(n)
            Cs = 1.0  # step size for C
            Cs_inv = 1.0
            step_size = 1.0

            # Update rates
            mu = lam // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            # Path variables
            ps = np.zeros(n)
            pc = np.zeros(n)

            def update_cmaes(y, fy):
                nonlocal mean, C, Cinv, Csqrt, Cinv_sqrt, sigma, ps, pc
                # Sort by fitness
                idx = np.argsort(fy)
                y = y[idx]
                fy = fy[idx]

                # Update mean
                yw = y[:mu]
                mean = mean + sigma * np.sum(weights[:, None] * yw, axis=0)

                # Update paths
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / sigma * Csqrt @ (np.sum(weights[:, None] * yw, axis=0))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (lam + 1))) <
                        (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1)))
                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) / sigma * C @ (np.sum(weights[:, None] * yw, axis=0))

                # Update covariance
                delta_h = (1 - hsig) * cc * (2 - cc)
                rank_mu = np.sum(weights[:, None, None] * yw[:, :, None] * yw[:, None, :], axis=0)
                rank_one = np.outer(pc, pc)
                C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * (rank_one + delta_h * np.outer(mean, mean) - np.outer(mean, mean)) + cmu * rank_mu

                # Ensure symmetry
                C = (C + C.T) / 2

                # Update Cholesky decomposition
                try:
                    Csqrt = np.linalg.cholesky(C)
                    Cinv = np.linalg.inv(C)
                    Cinv_sqrt = np.linalg.cholesky(Cinv)
                except np.linalg.LinAlgError:
                    # If Cholesky fails, reset covariance
                    C = np.eye(n)
                    Cinv = np.eye(n)
                    Csqrt = np.eye(n)
                    Cinv_sqrt = np.eye(n)

                # Update step size
                sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (lam + 1))) - 1))

            # Main CMA-ES loop
            calls_made = 0
            while calls_made < remaining_budget:
                if time.time() > deadline - 5:
                    break

                # Generate population
                y = np.array([Csqrt @ rng.standard_normal(n) for _ in range(lam)])
                fy = np.array([func(np.clip(mean + sigma * yi, -5, 5)) for yi in y])
                calls_made += lam

                # Update best
                for xi, fxi in zip(mean + sigma * y, fy):
                    xi = np.clip(xi, -5, 5)
                    if fxi < best_f:
                        best_f = fxi
                        best_x = xi.copy()

                # Update CMA-ES state
                update_cmaes(y, fy)

                # Termination criteria
                if sigma < 1e-10 or np.max(C) / np.min(C) > 1e14:
                    break

            return calls_made

        # Run CMA-ES with remaining budget
        remaining = total_budget - n_random
        if remaining > 10:
            run_cmaes(best_x, best_f, remaining, rng)

        # If we have time and budget left, try another restart from a different point
        elapsed = time.time() - start_time
        if elapsed < deadline - 30 and calls_made < total_budget:
            # Try another CMA-ES run from a different random point
            x2 = rng.uniform(-5, 5, dim)
            fx2 = func(x2)
            if fx2 < best_f:
                best_x = x2.copy()
                best_f = fx2

            remaining2 = total_budget - calls_made - 1
            if remaining2 > 10 and time.time() < deadline - 10:
                run_cmaes(best_x, best_f, remaining2, rng)

        return best_x, best_f
