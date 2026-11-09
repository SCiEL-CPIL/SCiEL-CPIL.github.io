"""Combined optimizer: CMA-ES for global exploration + (1+1)-ES polish for Rastrigin."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """CMA-ES with multi-restart structure and final (1+1)-ES polish."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # safety margin before 330s limit

        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = np.inf
        calls_used = 0

        def safe_func(x):
            nonlocal calls_used
            if calls_used >= budget:
                return np.inf
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5, 5)
            try:
                val = func(x)
                calls_used += 1
                if np.isfinite(val):
                    return float(val)
                return 1e10
            except Exception:
                calls_used += 1
                return 1e10

        def run_cma_es(x0, sigma0, max_calls):
            nonlocal best_x, best_f, calls_used
            
            if calls_used >= budget:
                return

            # CMA-ES parameters
            lam = 10
            mu = lam // 2
            
            x_mean = x0.copy()
            sigma = sigma0
            C = np.eye(dim)
            ps = np.zeros(dim)
            pc = np.zeros(dim)
            
            cmu = 2.0 / ((dim + 1.7) ** 2 + lam)
            cc = (4 + dim / lam) / ((dim + 2) ** 2 + lam)
            cs = (1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1)) / ((dim + 2) + lam)
            c1 = (2 - cmu + cc) / ((dim + 1.3) ** 2 + lam)
            cmu = 2 * (mu - 1 + 1.0 / mu) / ((dim + 2) ** 2 + lam)
            dsigma = 1 + 2 * max(0, np.sqrt((mu - 1) / (dim + 1)) - 1) + cc
            
            # Weights
            w = np.zeros(lam)
            w[:mu] = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
            w /= w.sum()
            mueff = 1.0 / np.sum(w ** 2)

            cholC = np.linalg.cholesky(C)

            # Initial population
            pop = []
            for i in range(lam):
                if calls_used >= budget or time.time() > deadline:
                    break
                zi = rng.standard_normal(dim)
                xi = x_mean + sigma * (cholC @ zi)
                fi = safe_func(xi)
                pop.append((fi, xi))
                if fi < best_f:
                    best_f = fi
                    best_x = xi.copy()

            if len(pop) < lam:
                return

            pop.sort(key=lambda t: t[0])

            while calls_used < budget and time.time() < deadline:
                # Update mean
                x_mean_old = x_mean.copy()
                x_mean = np.sum([w[i] * pop[i][1] for i in range(mu)], axis=0)

                # Update step size control variables
                ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (cholC.T @ (x_mean - x_mean_old)) / sigma
                hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (calls // lam + 1))) <
                        (1.4 + 2 / (dim + 1)) * np.trace(C) / dim)

                pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * (x_mean - x_mean_old) / sigma

                # Update covariance matrix
                rank1 = np.outer(pc, pc)
                delta_h = (1.0 if hsig else 0.0) * (2 - cc)
                rankmu = np.zeros((dim, dim))
                for i in range(mu):
                    zi = (pop[i][1] - x_mean_old) / sigma
                    rankmu += w[i] * np.outer(zi, zi)
                C = (1 + c1 * delta_h - c1 - cmu) * C + c1 * rank1 + cmu * rankmu

                C = (C + C.T) / 2.0
                try:
                    cholC = np.linalg.cholesky(C)
                except np.linalg.LinAlgError:
                    C += 1e-4 * np.eye(dim)
                    C = (C + C.T) / 2.0
                    cholC = np.linalg.cholesky(C)

                # Update sigma
                sigma *= np.exp(cs / dsigma * (np.linalg.norm(ps) / np.sqrt(2 * dim) - 1))

                # Generate new population
                new_pop = []
                for i in range(lam):
                    if calls_used >= budget or time.time() > deadline:
                        break
                    zi = rng.standard_normal(dim)
                    xi = x_mean + sigma * (cholC @ zi)
                    fi = safe_func(xi)
                    new_pop.append((fi, xi))
                    if fi < best_f:
                        best_f = fi
                        best_x = xi.copy()

                if len(new_pop) < lam:
                    break

                new_pop.sort(key=lambda t: t[0])
                pop = new_pop

        def run_es(start_x, max_calls):
            nonlocal best_x, best_f, calls_used
            
            if calls_used >= budget or best_x is None:
                return

            x = start_x.copy()
            fx = safe_func(x)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            
            sigma = 0.5
            used = 1
            stagnation = 0
            max_stagnation = 30

            while used < max_calls and calls_used < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = safe_func(y)
                used += 1
                
                if fy < fx:
                    x, fx = y, fy
                    sigma *= 1.2
                    stagnation = 0
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                else:
                    sigma *= 0.9
                    stagnation += 1
                    if stagnation >= max_stagnation:
                        break

        # Phase 1: CMA-ES from origin
        if calls_used < budget and time.time() < deadline:
            x0 = np.zeros(dim)
            sigma0 = 2.0
            max_calls = int(budget * 0.5)
            run_cma_es(x0, sigma0, max_calls)

        # Phase 2: CMA-ES from random point
        if calls_used < budget and time.time() < deadline:
            x0 = rng.uniform(-3, 3, dim)
            sigma0 = 1.5
            max_calls = int(budget * 0.3)
            run_cma_es(x0, sigma0, max_calls)

        # Phase 3: Polish with (1+1)-ES
        if calls_used < budget and time.time() < deadline and best_x is not None:
            start_x = best_x.copy()
            max_calls = budget - calls_used
            run_es(start_x, max_calls)

        return best_x, best_f
# EVOLVE-BLOCK-END
