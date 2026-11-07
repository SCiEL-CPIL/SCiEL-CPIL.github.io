"""Combined optimizer: sparse grid global search + CMA-ES local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import itertools


class Optimizer:
    """Hybrid optimizer: coarse 3^5 grid for global basin identification,
    then CMA-ES for adaptive local refinement of top basins."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _cmaes(self, func, x0, sigma0, max_calls, calls_ref, best_ref, deadline):
        """Run CMA-ES with shared call counter and best tracking.
        
        Returns (best_x, best_fx, calls_used).
        """
        dim = self.dim
        lam = 4 + int(3 * np.log(dim))  # ~12 for dim=5
        mu = lam // 2
        weights = np.log(np.arange(mu, 0, -1))
        weights /= weights.sum()
        mu_eff = 1.0 / np.sum(weights ** 2)

        cc = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        cs = (mu_eff + 2) / (dim + mu_eff + 5)
        c1 = 2 / ((dim + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        damps = 1 + 2 * max(0, (mu_eff - 2) / (dim + 2) - 1) + cs

        x = np.array(x0, dtype=float)
        sigma = sigma0
        C = np.eye(dim)
        eigvals, eigvecs = np.linalg.eigh(C)
        eigvals = np.maximum(eigvals, 1e-20)
        B = eigvecs
        D = np.sqrt(eigvals)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        chi_n = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        best_x = np.array(x, dtype=float)
        best_fx = func(x)
        calls_used = 1
        calls_ref['n'] += 1
        if best_fx < best_ref['f']:
            best_ref['f'] = best_fx
            best_ref['x'] = best_x.copy()

        no_improve = 0

        for gen in range(1000):
            if time.time() > deadline:
                break
            if calls_ref['n'] + lam > max_calls:
                break

            pop = []
            fvals = []
            for i in range(lam):
                if time.time() > deadline:
                    break
                if calls_ref['n'] >= max_calls:
                    break

                z = np.random.standard_normal(dim)
                y = x + sigma * (B * D) @ z
                y = np.clip(y, -5, 5)
                fy = func(y)
                calls_ref['n'] += 1
                calls_used += 1
                pop.append(y)
                fvals.append(fy)

                if fy < best_fx:
                    best_fx = fy
                    best_x = y.copy()
                    if best_fx < best_ref['f']:
                        best_ref['f'] = best_fx
                        best_ref['x'] = best_x.copy()

            if len(pop) == 0:
                break

            idx = np.argsort(fvals)
            pop = [pop[i] for i in idx]
            fvals = np.array(fvals)[idx]

            if fvals[0] < best_fx - 1e-12:
                no_improve = 0
            else:
                no_improve += 1

            if no_improve > 30 and sigma < 1e-6:
                break

            mu_use = min(mu, len(pop))
            m = np.sum(weights[:mu_use, None] * np.array(pop[:mu_use]), axis=0)

            # Update step size
            ps = (1 - cs) * ps + np.sqrt(cs * (2 - cs) * mu_eff) / sigma * (B.T @ (m - x))
            hsig = 1 if np.linalg.norm(ps) / np.sqrt(1 - (1 - cs) ** (2 * (gen + 1))) < (1.4 + 2 / (dim + 1)) * chi_n else 0
            sigma *= np.exp(cs / damps * (np.linalg.norm(ps) / chi_n - 1))
            sigma = np.clip(sigma, 1e-12, 50.0)

            # Update evolution path
            pc = (1 - cc) * pc + hsig * np.sqrt(cc * (2 - cc) * mu_eff) / sigma * (B.T @ (m - x))

            # Update covariance
            delta_hs = (1 - hsig) * cc * (2 - cc)
            outer = np.outer(pc, pc)
            C = (1 + c1 * delta_hs - c1 - cmu) * C + c1 * outer
            for i in range(mu_use):
                zi = (np.array(pop[i]) - x) / sigma
                C += cmu * weights[i] * np.outer(zi, zi)

            # Ensure symmetry and positive definite
            C = (C + C.T) / 2
            eigvals, eigvecs = np.linalg.eigh(C)
            eigvals = np.maximum(eigvals, 1e-20)
            B = eigvecs
            D = np.sqrt(eigvals)

            x = m

        return best_x, best_fx, calls_used

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 280.0  # safety margin before 300s limit

        calls_ref = {'n': 0}
        best_ref = {'f': float('inf'), 'x': None}

        def evaluate(x):
            nonlocal deadline
            if time.time() > deadline:
                raise TimeoutError()
            if calls_ref['n'] >= budget:
                raise RuntimeError()

            calls_ref['n'] += 1
            try:
                f_val = float(func(x))
            except Exception:
                return best_ref['f'] if best_ref['f'] != float('inf') else 0.0

            if f_val < best_ref['f']:
                best_ref['f'] = f_val
                best_ref['x'] = np.array(x, dtype=float)
            return f_val

        N = dim

        # Phase 1: Sparse 3^5 grid search for global basin identification
        # Minimizer is in [-4,4]^5, so use levels -4, 0, 4
        grid_points = []
        levels = [-4.0, 0.0, 4.0]
        for combo in itertools.product(levels, repeat=N):
            grid_points.append(np.array(combo, dtype=float))

        # Evaluate grid points in a deterministic order that gives early progress
        # Sort by distance from origin to get reasonable early progress
        grid_points.sort(key=lambda p: np.linalg.norm(p))

        grid_results = []
        try:
            for x in grid_points:
                f_val = evaluate(x)
                grid_results.append((f_val, x))
        except (TimeoutError, RuntimeError):
            pass

        # Sort grid results by fitness
        grid_results.sort(key=lambda x: x[0])

        # Select top K basins for CMA-ES refinement
        K = min(5, len(grid_results))
        top_basins = []
        for f_val, x in grid_results[:K]:
            top_basins.append(x)

        # If grid search used all budget, we are done
        if calls_ref['n'] >= budget:
            return best_ref['x'], best_ref['f']

        # Phase 2: CMA-ES local refinement from each top basin
        # Allocate remaining budget among basins, prioritizing best basins
        remaining = budget - calls_ref['n']
        if remaining < 10:
            return best_ref['x'], best_ref['f']

        # Give more budget to better basins
        # Basin 0 gets 40%, basin 1 gets 25%, basin 2 gets 15%, basin 3 gets 10%, basin 4 gets 10%
        allocations = [0.40, 0.25, 0.15, 0.10, 0.10]
        allocations = allocations[:K]
        total_alloc = sum(allocations)
        allocations = [a / total_alloc for a in allocations]

        for i, start_x in enumerate(top_basins):
            if calls_ref['n'] >= budget - 5:
                break
            if time.time() > deadline:
                break

            alloc_calls = int(remaining * allocations[i])
            if alloc_calls < 10:
                alloc_calls = 10

            try:
                # Use different initial sigma for diversity
                sigma0 = 1.5 if i == 0 else 1.0 if i == 1 else 0.5
                cma_x, cma_fx, cma_calls = self._cmaes(
                    func, start_x, sigma0, budget, calls_ref, best_ref, deadline
                )
            except (TimeoutError, RuntimeError):
                break
            except Exception:
                pass

        return best_ref['x'], best_ref['f']
# EVOLVE-BLOCK-END
