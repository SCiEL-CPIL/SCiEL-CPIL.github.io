"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with adaptive restarts for Büche-Rastrigin optimization."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s kill
        n = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def check_deadline():
            return time.time() > deadline

        def eval_point(x):
            nonlocal calls_used
            if calls_used >= budget:
                return None
            fx = func(x)
            calls_used += 1
            return fx

        # CMA-ES implementation
        def cma_es(x0, sigma0, max_calls, deadline):
            nonlocal calls_used, best_x, best_f

            if max_calls < 10:
                return

            x0 = np.array(x0, dtype=float)
            n = len(x0)

            # CMA-ES parameters
            lambda_ = max(4, int(4 + 3 * np.log(n)))
            mu = lambda_ // 2
            weights = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            weights /= np.sum(weights)
            mueff = 1.0 / np.sum(weights ** 2)
            cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
            cs = (mueff + 2) / (n + mueff + 5)
            c1 = 2 / ((n + 1.3) ** 2 + mueff)
            cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))

            x = x0.copy()
            sigma = sigma0
            ps = np.zeros(n)
            pc = np.zeros(n)
            C = np.eye(n)
            B = np.eye(n)
            D = np.ones(n)

            # Evaluate initial point
            fx = eval_point(x)
            if fx is None:
                return
            local_best_x = x.copy()
            local_best_f = fx

            local_calls = 1
            max_iter = max(1, max_calls // lambda_)

            for it in range(max_iter):
                if check_deadline() or local_calls >= max_calls:
                    break

                # Generate population
                offspring = []
                for i in range(lambda_):
                    if local_calls >= max_calls or check_deadline():
                        break
                    z = np.random.standard_normal(n)
                    y = x + sigma * (B @ z)
                    y = np.clip(y, -5, 5)
                    fy = eval_point(y)
                    if fy is None:
                        break
                    local_calls += 1
                    offspring.append((fy, y))

                if len(offspring) < mu:
                    break

                # Sort
                offspring.sort(key=lambda t: t[0])

                # Update best
                if offspring[0][0] < local_best_f:
                    local_best_f = offspring[0][0]
                    local_best_x = offspring[0][1].copy()
                    if local_best_f < best_f:
                        best_f = local_best_f
                        best_x = local_best_x.copy()

                # Weighted mean
                xnew = np.zeros(n)
                for i in range(mu):
                    xnew += weights[i] * offspring[i][1]

                # Step size control
                ydiff = (xnew - x) / sigma
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) / D * ydiff
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (it + 1))) < \
                       (1.4 + 2 / (n + 1)) * np.sqrt((n + 1) / (n - 1))
                hs = float(hsig)
                pc = (1 - cc) * pc + hs * cc / D * ydiff

                # Covariance matrix adaptation
                rank1 = np.outer(pc, pc)
                delta_h = (1 - hs) * cc * (1 - cc)
                rankmu = np.zeros((n, n))
                for i in range(mu):
                    zi = (offspring[i][1] - x) / sigma
                    rankmu += np.outer(weights[i] * zi, weights[i] * zi)
                C = (1 + c1 * delta_h - c1) * C + c1 * rank1 + cmu * rankmu

                # Eigen decomposition
                eigvals, eigvecs = np.linalg.eigh(C)
                eigvals = np.clip(eigvals, 1e-12, None)
                D = np.sqrt(eigvals)
                B = eigvecs

                # Update
                x = xnew
                sigma *= np.exp(cs / (n + 4) * (np.linalg.norm(ydiff) / np.sqrt(2 * n) - 1))
                sigma = max(sigma, 1e-12)
                sigma = min(sigma, 10.0)

        # Strategy: Multiple CMA-ES restarts with different initializations
        # Büche-Rastrigin has a single global minimum, so we need good exploration
        # first, then exploitation.

        # Phase 1: Broad exploration with a few random restarts
        # Use about 40% of budget for exploration
        explore_budget = int(budget * 0.4)
        n_explore = 4

        for r in range(n_explore):
            if check_deadline() or calls_used >= budget:
                break

            # Different starting points
            if r == 0:
                x0 = np.zeros(n)
                sigma0 = 3.0
            elif r == 1:
                x0 = np.random.uniform(-4, 4, n)
                sigma0 = 2.5
            elif r == 2:
                x0 = np.random.uniform(-3, 3, n)
                sigma0 = 2.0
            else:
                x0 = np.random.uniform(-4, 4, n)
                sigma0 = 3.5

            alloc = explore_budget // n_explore
            cma_es(x0, sigma0, alloc, deadline)

        # Phase 2: Exploitation around best found point
        if best_x is not None and not check_deadline() and calls_used < budget:
            remaining = budget - calls_used
            # Run CMA-ES centered at best point with smaller sigma
            cma_es(best_x, 0.5, remaining, deadline)

        # Phase 3: Fine-tuning if we have budget left
        if best_x is not None and not check_deadline() and calls_used < budget:
            remaining = budget - calls_used
            if remaining > 50:
                cma_es(best_x, 0.1, remaining, deadline)
            else:
                # Simple local search
                for _ in range(remaining):
                    if check_deadline() or calls_used >= budget:
                        break
                    y = np.clip(best_x + 0.05 * np.random.standard_normal(n), -5, 5)
                    fy = eval_point(y)
                    if fy is not None and fy < best_f:
                        best_f = fy
                        best_x = y.copy()

        if best_x is None:
            # Should not happen, but just in case
            best_x = np.zeros(n)
            best_f = func(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
