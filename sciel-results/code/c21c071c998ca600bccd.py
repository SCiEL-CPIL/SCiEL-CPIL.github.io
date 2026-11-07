"""Multi-restart CMA-ES for BBOB f6 (Attractive sector)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-restart CMA-ES for BBOB f6 (Attractive sector)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = np.zeros(dim)
        best_fx = np.inf
        calls_used = 0

        def time_ok():
            return time.time() < deadline - 3.0 and calls_used < budget

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

        # Phase 1: Global exploration with 50 uniform random samples
        n_global = min(50, budget)
        candidates = []
        for _ in range(n_global):
            x = rng.uniform(-5, 5, dim)
            result = safe_call(x)
            if result is None:
                break
            x_c, fx = result
            candidates.append((fx, x_c))

        if not candidates:
            result = safe_call(np.zeros(dim))
            if result:
                best_x = result[0]
                best_fx = result[1]
            return best_x, best_fx

        candidates.sort(key=lambda t: t[0])
        best_x = candidates[0][1].copy()
        best_fx = candidates[0][0]

        # Phase 2: CMA-ES with restarts
        # We'll run multiple CMA-ES restarts, each with a limited budget
        # The first restart starts from the best point found in Phase 1
        # Subsequent restarts start from random points near the best

        remaining = budget - calls_used
        if remaining <= 10:
            return best_x, best_fx

        # CMA-ES parameters
        lambda_pop = 12
        mu = lambda_pop // 2

        def run_cma_es(x0, sigma0, max_calls):
            """Run CMA-ES for at most max_calls evaluations."""
            nonlocal calls_used, best_x, best_fx

            x = x0.copy()
            sigma = sigma0

            weights = np.log(mu + 1) - np.log(np.arange(1, mu + 1))
            weights /= weights.sum()
            mu_eff = 1.0 / np.sum(weights ** 2)

            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + c_sigma
            cs = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            ds = 1 + 2 * max(0, (mu_eff - 1) / (dim + 1)) + cs

            c1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (1 - c_sigma)
            cmu = min(1 - c1, 2 * (weights[0] - 0.5)) * c_sigma * c_sigma

            ps = np.zeros(dim)
            pc = np.zeros((dim, dim))
            C = np.eye(dim)
            csqrt = np.eye(dim)
            chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

            no_improve = 0
            best_local_fx = best_fx
            best_local_x = best_x.copy()
            start_calls = calls_used
            gen = 0

            while time_ok() and (calls_used - start_calls) < max_calls:
                pop = []
                for i in range(lambda_pop):
                    if not time_ok() or (calls_used - start_calls) >= max_calls:
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

                pop.sort(key=lambda t: t[0])

                if pop[0][0] < best_local_fx:
                    best_local_fx = pop[0][0]
                    best_local_x = pop[0][1].copy()
                    no_improve = 0
                else:
                    no_improve += 1

                # Check for stagnation
                if no_improve > 25:
                    # Restart with smaller sigma near best
                    if gen > 5:
                        break
                    x = best_local_x + rng.normal(0, 0.1, dim)
                    x = np.clip(x, -5, 5)
                    sigma = max(0.05, sigma * 0.5)
                    ps = np.zeros(dim)
                    pc = np.zeros((dim, dim))
                    C = np.eye(dim)
                    csqrt = np.eye(dim)
                    no_improve = 0
                    gen += 1
                    continue

                # Update mean
                x_new = np.sum([weights[i] * pop[i][1] for i in range(min(mu, len(pop)))], axis=0)

                # Update path
                ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) / sigma * (csqrt.T @ (x_new - x))
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * max(1, gen + 1)))) < \
                       (1 + 2 / (dim + 1)) * np.sqrt(mu_eff)
                pc = (1 - cs) * pc + hsig * np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (x_new - x)

                # Update covariance
                rank1 = np.outer(pc, pc)
                rankmu = np.zeros((dim, dim))
                for i in range(min(mu, len(pop))):
                    y = (pop[i][1] - x) / sigma
                    rankmu += weights[i] * np.outer(y, y)

                C = (1 + c1 + cmu - c1 - c_sigma) * C + c1 * rank1 + cmu * rankmu
                C = (C + C.T) / 2
                try:
                    w, V = np.linalg.eigh(C)
                    w = np.maximum(w, 1e-14)
                    C = V @ np.diag(w) @ V.T
                    csqrt = V @ np.diag(np.sqrt(w)) @ V.T
                except Exception:
                    csqrt = np.eye(dim)
                    C = np.eye(dim)

                # Update sigma
                norm_ps = np.linalg.norm(ps)
                sigma *= np.exp(c_sigma / d_sigma * (norm_ps / chi_n - 1))
                sigma = np.clip(sigma, 0.001, 10.0)
                x = x_new
                gen += 1

            if best_local_fx < best_fx:
                best_fx = best_local_fx
                best_x = best_local_x.copy()

        # Run CMA-ES with restarts
        # First restart: from best point, sigma=1.0
        max_calls_per_restart = min(150, remaining)
        run_cma_es(best_x.copy(), 1.0, max_calls_per_restart)

        # If we still have budget, do another restart from a slightly perturbed best
        remaining = budget - calls_used
        if remaining > 20 and time_ok():
            x_restart = best_x + rng.normal(0, 0.3, dim)
            x_restart = np.clip(x_restart, -5, 5)
            max_calls2 = min(100, remaining)
            run_cma_es(x_restart, 0.5, max_calls2)

        # Final: if we have a few more calls, do a quick local search
        remaining = budget - calls_used
        if remaining > 5 and time_ok():
            for _ in range(min(remaining, 10)):
                if not time_ok():
                    break
                x_try = best_x + rng.normal(0, 0.1, dim)
                x_try = np.clip(x_try, -5, 5)
                result = safe_call(x_try)
                if result is None:
                    break
                x_c, fx = result
                if fx < best_fx:
                    best_fx = fx
                    best_x = x_c.copy()

        return best_x, best_fx
# EVOLVE-BLOCK-END
