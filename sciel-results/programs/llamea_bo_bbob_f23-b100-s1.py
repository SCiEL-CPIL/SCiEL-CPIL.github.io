import numpy as np
import time
import scipy.stats.qmc as qmc
from scipy.optimize import differential_evolution

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 285.0
        dim = self.dim
        budget = self.budget
        calls = 0
        best_x = None
        best_f = np.inf
        rng = np.random.default_rng(42)

        def safe_func(x):
            nonlocal calls, best_x, best_f
            if time.time() > deadline:
                raise TimeoutError
            if calls >= budget:
                raise StopIteration
            calls += 1
            x = np.asarray(x, dtype=float).flatten()
            if len(x) != dim:
                x = x[:dim]
            x = np.clip(x, -5, 5)
            val = func(x)
            if np.isfinite(val) and val < best_f:
                best_f = val
                best_x = x.copy()
            return val

        # Phase 1: Sobol sequence exploration
        n_explore = min(80, budget)
        sampler = qmc.Sobol(d=dim, scramble=True, seed=123)
        sobol_points = sampler.random(n_explore) * 10.0 - 5.0

        for x in sobol_points:
            if calls >= budget or time.time() > deadline:
                break
            try:
                safe_func(x)
            except (TimeoutError, StopIteration):
                break
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            try:
                safe_func(best_x)
            except Exception:
                pass

        if best_x is None:
            return np.zeros(dim), np.inf

        # Phase 2: Multi-start local search using Nelder-Mead
        if calls < budget and time.time() < deadline:
            n_starts = min(5, max(1, (budget - calls) // 20))
            starts = [best_x.copy()]
            # Add some perturbed starts
            for i in range(n_starts - 1):
                if time.time() > deadline or calls >= budget:
                    break
                x_try = best_x + rng.normal(0, 0.5, dim)
                x_try = np.clip(x_try, -5, 5)
                try:
                    f_try = safe_func(x_try)
                    if f_try < best_f + 1.0:  # Only keep if not too bad
                        starts.append(x_try.copy())
                except (TimeoutError, StopIteration):
                    break
                except Exception:
                    pass

            for s in starts:
                if calls >= budget or time.time() > deadline:
                    break
                try:
                    # Use Nelder-Mead locally
                    from scipy.optimize import minimize
                    result = minimize(
                        lambda x: safe_func(x),
                        s.copy(),
                        method='Nelder-Mead',
                        options={'maxiter': 50, 'xatol': 1e-6, 'fatol': 1e-8}
                    )
                    if result.success or np.isfinite(result.fun):
                        if result.fun < best_f:
                            best_f = result.fun
                            best_x = result.x.copy()
                except (TimeoutError, StopIteration):
                    break
                except Exception:
                    pass

        # Phase 3: CMA-ES for final exploitation
        if calls < budget - 20 and time.time() < deadline:
            try:
                self._cma_es(safe_func, best_x.copy(), calls, budget, deadline)
            except Exception:
                pass

        return best_x, best_f

    def _cma_es(self, safe_func, x0, calls, budget, deadline):
        dim = self.dim
        lam = max(4, int(4 + 3 * np.log(dim)))
        mu = lam // 2

        # Weights
        w = np.log(np.arange(1, mu + 1))
        w /= w.sum()
        mueff = 1.0 / np.sum(w ** 2)

        # CMA-ES parameters
        cs = (mueff + 1) / (dim + mueff + 3)
        ds = 1 + 2 * max(0, np.sqrt((mueff - 1) / (dim + 1)) - 1) + cs
        c1 = 2 / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, (2 * (mueff - 2 + 1 / mueff)) / ((dim + 2) ** 2 + mueff))

        m = x0.copy()
        sigma = 0.3
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)

        D, B = np.linalg.eigh(C)
        D = np.sqrt(np.maximum(D, 1e-12))

        no_improve = 0
        best_local_f = safe_func(m.copy())

        while calls < budget and time.time() < deadline:
            # Generate population
            Z = np.random.randn(lam, dim)
            X = m[None, :] + sigma * (Z @ (B / D).T)

            # Evaluate
            fvals = []
            for x in X:
                try:
                    fvals.append(safe_func(x))
                except (TimeoutError, StopIteration):
                    break
                except Exception:
                    fvals.append(np.inf)

            if len(fvals) < lam:
                break

            # Sort
            idx = np.argsort(fvals)
            X = X[idx]
            fvals = np.array(fvals)[idx]

            # Update mean
            y_w = np.sum(w[:, None] * (X[:mu] - m) / sigma, axis=0)
            m_new = m + sigma * y_w

            # Update ps
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mueff) * (B / D) @ y_w

            # Update pc
            norm_ps = np.linalg.norm(ps)
            hsig = (norm_ps / np.sqrt(1 - (1 - cs) ** 2) < (1.4 + 2 / (dim + 1)) * np.sqrt((dim + 1) / (dim - 1)))
            pc = (1 - c1) * pc + hsig * np.sqrt(c1 * (2 - c1) * mueff) * y_w

            # Update C
            C = (1 - c1 - cmu) * C + c1 * (pc @ pc.T + (1 - hsig) * c1 * C) + \
                cmu * np.sum(w[:, None, None] * (y_w[:, None] * y_w[None, :]), axis=0)
            C = (C + C.T) / 2

            # Update sigma
            sigma *= np.exp(cs / ds * (norm_ps / np.sqrt(dim) - 1))

            # Ensure C is positive definite
            try:
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
            except:
                C = np.eye(dim)
                D = np.ones(dim)
                B = np.eye(dim)

            m = m_new

            # Check for improvement
            if fvals[0] < best_local_f - 1e-10:
                best_local_f = fvals[0]
                no_improve = 0
            else:
                no_improve += 1

            # Restart if stuck
            if no_improve > 10 and calls < budget - 10:
                m = m + np.random.randn(dim) * 0.3
                m = np.clip(m, -5, 5)
                sigma = 0.3
                C = np.eye(dim)
                pc = np.zeros(dim)
                ps = np.zeros(dim)
                D, B = np.linalg.eigh(C)
                D = np.sqrt(np.maximum(D, 1e-12))
                no_improve = 0

            # Convergence check
            if sigma < 1e-10:
                break

        return m, best_local_f
