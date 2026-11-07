"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-start CMA-ES with adaptive population sizing for BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        rng = np.random.default_rng(12345)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_fx = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline - 2.0 and calls_used < budget

        def safe_call(x):
            nonlocal calls_used
            if not time_ok():
                return None
            x = np.clip(np.asarray(x, dtype=np.float64), -5, 5)
            try:
                fx = func(x)
                fx = float(fx)
            except Exception:
                return None
            calls_used += 1
            return x.copy(), fx

        # Phase 1: Dense global exploration - 60 samples
        n_global = min(60, budget)
        candidates = []
        for _ in range(n_global):
            x = rng.uniform(-5, 5, dim)
            result = safe_call(x)
            if result is None:
                break
            x_c, fx = result
            candidates.append((fx, x_c))

        if not candidates:
            # Fallback: just return origin
            result = safe_call(np.zeros(dim))
            if result:
                return result[0], result[1]
            return np.zeros(dim), 0.0

        candidates.sort(key=lambda t: t[0])
        best_x = candidates[0][1].copy()
        best_fx = candidates[0][0]

        # Phase 2: Multi-start CMA-ES
        # Use top 3 starting points from global phase
        n_starts = min(3, len(candidates))
        start_points = [candidates[i][1].copy() for i in range(n_starts)]

        # Allocate budget: give more to better starting points
        # Start 1 gets 50%, Start 2 gets 30%, Start 3 gets 20%
        if n_starts == 3:
            budgets = [int(budget * 0.5), int(budget * 0.3), int(budget * 0.2)]
        elif n_starts == 2:
            budgets = [int(budget * 0.6), int(budget * 0.4)]
        else:
            budgets = [budget]

        for start_idx in range(n_starts):
            if not time_ok():
                break

            x = start_points[start_idx].copy()
            sigma = 1.0  # initial step size
            start_budget = budgets[start_idx]
            start_calls = calls_used

            # CMA-ES parameters
            lambda_pop = 12
            mu = lambda_pop // 2
            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs

            c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma)
            cmu = min(1 - c1, 2 * (weights[0] - 0.5)) * c_sigma * c_sigma

            # State variables
            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            C = np.eye(dim)
            csqrt = np.eye(dim)

            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            no_improve = 0
            best_local_fx = best_fx
            best_local_x = best_x.copy()

            while time_ok() and (calls_used - start_calls) < start_budget:
                # Generate population
                pop = []
                for i in range(lambda_pop):
                    if not time_ok() or (calls_used - start_calls) >= start_budget:
                        break
                    zi = rng.standard_normal(dim)
                    xi = x + sigma * (csqrt @ zi)
                    result = safe_call(xi)
                    if result is None:
                        break
                    xi_c, fi = result
                    pop.append((fi, xi_c))

                if len(pop) < mu:
                    break

                # Sort by fitness
                pop.sort(key=lambda t: t[0])

                # Update best
                if pop[0][0] < best_local_fx:
                    best_local_fx = pop[0][0]
                    best_local_x = pop[0][1].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Check for stagnation
                if no_improve > 40:
                    # Restart locally with smaller sigma
                    x = best_local_x + rng.normal(0, 0.2, dim)
                    x = np.clip(x, -5, 5)
                    sigma = max(0.1, sigma * 0.5)
                    ps = np.zeros(dim)
                    pc = np.zeros((dim, dim))
                    C = np.eye(dim)
                    csqrt = np.eye(dim)
                    no_improve = 0
                    continue

                # Update mean
                x_new = np.sum([weights[i] * pop[i][1] for i in range(min(mu, len(pop)))], axis=0)

                # Update evolution paths
                n_gen = calls_used - start_calls
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (csqrt.T @ (x_new - x))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, n_gen)))) < \
                       (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x_new - x)

                # Update covariance matrix
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(pop))):
                    y = (pop[i][1] - x) / sigma
                    rankmu += weights[i] * np.outer(y, y)

                C = (1 + c1 + cmu - c1 - c_sigma) * C + c1 * rank1 + cmu * rankmu

                # Ensure C is symmetric and PSD
                C = (C + C.T) / 2
                try:
                    w, V = np.linalg.eigh(C)
                    w = np.maximum(w, 1e-14)
                    C = V @ np.diag(w) @ V.T
                    csqrt = V @ np.diag(np.sqrt(w)) @ V.T
                except Exception:
                    csqrt = np.eye(dim)
                    C = np.eye(dim)

                # Update step size
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 0.001, 15.0)

                x = x_new

            # Update global best
            if best_local_fx < best_fx:
                best_fx = best_local_fx
                best_x = best_local_x.copy()

        # Phase 3: Final local refinement with Nelder-Mead
        remaining = budget - calls_used
        if remaining > 10 and time_ok():
            from scipy.optimize import minimize
            try:
                res = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': min(remaining, 100), 'xatol': 1e-6, 'fatol': 1e-8}
                )
                if res.fun < best_fx:
                    best_x = res.x.copy()
                    best_fx = res.fun
            except Exception:
                pass

        return best_x, best_fx
# EVOLVE-BLOCK-END
