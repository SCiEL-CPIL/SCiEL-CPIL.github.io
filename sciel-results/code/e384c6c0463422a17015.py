"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with restarts for BBOB f7 (Step Ellipsoidal)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280  # safety margin before 300s limit

        best_x = None
        best_f = np.inf
        calls_made = 0

        def count_call():
            nonlocal calls_made
            calls_made += 1

        def safe_func(x):
            x = np.asarray(x, dtype=np.float64)
            x = np.clip(x, -5.0, 5.0)
            val = func(x)
            count_call()
            return float(val)

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_iter):
            n = dim
            # Parameters
            mu = max(1, int(np.floor(3.0 * n / 2)))  # number of parents
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 - cs) / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = np.array(x0, dtype=np.float64)
            sigma = sigma0
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            evolution_path_h = 0.0
            damps = 1 + 2 * max(0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
            condD = 1.0

            for it in range(max_iter):
                if time.time() > deadline:
                    break

                # Generate offspring
                z = np.random.randn(mu, n)
                offspring = x_mean + sigma * (z @ (B * D))
                offspring = np.clip(offspring, -5.0, 5.0)

                # Evaluate offspring
                offspring_f = np.array([safe_func(offspring[i]) for i in range(mu)])

                # Sort by fitness
                order = np.argsort(offspring_f)
                offspring = offspring[order]
                offspring_f = offspring_f[order]

                # Update best
                if offspring_f[0] < best_f:
                    best_f = offspring_f[0]
                    best_x = offspring[0].copy()

                # Update mean
                x_mean = np.sum(weights[:, None] * offspring[:mu], axis=0)

                # Update covariance matrix adaptation
                y = (x_mean - x_mean_prev) / sigma
                pc = (1 - cc) * pc + cc * y
                hsig = (np.sqrt((1 - (1 - cc) ** (it + 1))) / np.sqrt(1 - (1 - cc) ** (it + 1)))
                # Simplified: check if pc is too large
                if np.linalg.norm(pc) / np.sqrt(1 - (1 - cc) ** (it + 1)) < (1.4 + 1 / cc) * chi_n:
                    ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y
                    delta_sig = 1 + cs * (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1)
                    delta_sig = np.clip(delta_sig, 1e-5, 1e5)
                    sigma *= delta_sig

                # Rank-mu update
                yw = np.sum(weights[:, None] * offspring[:mu], axis=0) / sigma
                delta_h = 1 - (1 - cc) ** (it + 1)
                rank_mu = (cmu / delta_h) * (np.outer(yw, yw) +
                                              (cmu / mueff) * np.sum(weights ** 2) * np.eye(n))
                # For simplicity, we'll use a simplified covariance update
                # Full CMA-ES would track C matrix, but let's use a simpler approach

                # Actually, let's implement proper CMA-ES with covariance matrix
                # Reset and do it properly below

                x_mean_prev = x_mean.copy()

                # Check for stagnation
                if it > 0 and it % 50 == 0:
                    if np.std(offspring_f[:mu]) < 1e-10:
                        break

            return best_x, best_f, calls_made

        # Let's implement a cleaner CMA-ES
        def cma_es_proper(x0, sigma0, max_iter):
            n = dim
            mu = max(1, int(np.floor(3.0 * n / 2)))
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = (2 - cs) / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
            chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))

            x_mean = np.array(x0, dtype=np.float64)
            sigma = sigma0
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)
            pc = np.zeros(n)
            ps = np.zeros(n)
            it = 0

            while it < max_iter:
                if time.time() > deadline:
                    break

                # Generate offspring
                z = np.random.randn(mu, n)
                offspring = x_mean[None, :] + sigma * (z @ (B * D))
                offspring = np.clip(offspring, -5.0, 5.0)

                # Evaluate offspring
                offspring_f = np.array([safe_func(offspring[i]) for i in range(mu)])

                # Sort by fitness
                order = np.argsort(offspring_f)
                offspring = offspring[order]
                offspring_f = offspring_f[order]

                # Update best
                if offspring_f[0] < best_f:
                    best_f = offspring_f[0]
                    best_x = offspring[0].copy()

                # Save old mean
                x_mean_old = x_mean.copy()

                # Update mean
                x_mean = np.sum(weights[:, None] * offspring[:mu], axis=0)

                # Update evolution paths
                y = (x_mean - x_mean_old) / sigma
                pc = (1 - cc) * pc + cc * y
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * y

                # Update sigma
                delta_sig = 1 + cs * (np.linalg.norm(ps) ** 2 / chi_n ** 2 - 1)
                delta_sig = np.clip(delta_sig, 1e-5, 1e5)
                sigma *= delta_sig

                # Update covariance matrix
                yw = np.sum(weights[:, None] * offspring[:mu], axis=0) / sigma
                delta_h = 1 - (1 - cc) ** (it + 1)
                rank_one = np.outer(pc, pc)
                rank_mu = (cmu / delta_h) * (np.outer(yw, yw) +
                                             (cmu / mueff) * np.sum(weights ** 2) * np.eye(n))
                C = (1 + c1 * (rank_one - C) + cmu * (rank_mu - C)) * (1 + 2 * cs * (np.dot(ps, ps) / chi_n ** 2 - 1))

                # Ensure positive semi-definite
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                C = (eigvecs * eigvals) @ eigvecs.T

                # Update B and D
                B, D = np.linalg.eigh(C)
                D = np.sqrt(np.diag(C))  # This isn't quite right, let me fix

                # Proper eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.maximum(eigvals, 1e-12)
                B = eigvecs
                D = np.sqrt(eigvals)

                # Check for convergence
                if sigma < 1e-15 or np.all(np.abs(eigvals) < 1e-15):
                    break

                it += 1

                # Check stagnation
                if it % 100 == 0 and np.std(offspring_f[:mu]) < 1e-10:
                    break

            return best_x, best_f

        # Strategy: multiple restarts with different initializations
        # Use a mix of random starts and guided starts

        # First, do a quick random search to get a good starting point
        n_initial = min(20, budget // 5)
        candidates = []
        for _ in range(n_initial):
            if time.time() > deadline or calls_made >= budget:
                break
            x = np.random.uniform(-5, 5, dim)
            f = safe_func(x)
            candidates.append((f, x))
            if f < best_f:
                best_f = f
                best_x = x.copy()

        candidates.sort()
        top_candidates = [c[1] for c in candidates[:min(5, len(candidates))]]

        # Now run CMA-ES from each top candidate
        # Allocate remaining budget among restarts
        remaining = budget - calls_made
        n_restarts = min(5, len(top_candidates))
        if n_restarts == 0:
            n_restarts = 1
            top_candidates = [np.random.uniform(-5, 5, dim)]

        budget_per_restart = max(10, remaining // n_restarts)

        for i in range(n_restarts):
            if time.time() > deadline or calls_made >= budget - 5:
                break

            x0 = top_candidates[i] if i < len(top_candidates) else np.random.uniform(-5, 5, dim)
            sigma0 = 1.5 if i == 0 else 1.0

            # Run CMA-ES
            local_best_x, local_best_f = cma_es_proper(x0, sigma0, budget_per_restart)

            if local_best_f < best_f:
                best_f = local_best_f
                best_x = local_best_x.copy()

        # Final polish with a simple local search if budget remains
        if time.time() < deadline - 10 and calls_made < budget - 10 and best_x is not None:
            x = best_x.copy()
            f = best_f
            step = 0.5
            for _ in range(min(30, budget - calls_made)):
                if time.time() > deadline or calls_made >= budget:
                    break
                improved = False
                for d in range(dim):
                    if time.time() > deadline or calls_made >= budget:
                        break
                    for direction in [-1, 1]:
                        y = x.copy()
                        y[d] = np.clip(y[d] + direction * step, -5, 5)
                        fy = safe_func(y)
                        if fy < f:
                            x, f = y, fy
                            improved = True
                            if f < best_f:
                                best_f = f
                                best_x = x.copy()
                if not improved:
                    step *= 0.5
                    if step < 1e-4:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
