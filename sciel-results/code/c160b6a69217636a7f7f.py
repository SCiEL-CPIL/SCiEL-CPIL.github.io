# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 295.0
        budget = self.budget
        dim = self.dim

        best_x = np.zeros(dim)
        best_f = np.inf
        calls = 0

        def _time_ok():
            return time.time() < deadline

        def _call(x):
            nonlocal best_x, best_f, calls
            x = np.asarray(x, dtype=float).reshape(dim)
            x = np.clip(x, -5.0, 5.0)
            fx = float(func(x))
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # ---------- Phase 1: stratified 3x3^5 lattice + a few random points ----------
        n_coarse = min(budget, 243)
        rng = np.random.default_rng(1234)

        # 3x3x3x3x3 grid over [-5,5]
        grid_vals = np.linspace(-4.5, 4.5, 3)
        coarse_pts = list(np.meshgrid(*[grid_vals] * dim, indexing='ij'))
        coarse_pts = np.array(coarse_pts).transpose(*range(dim), dim).reshape(-1, dim)

        # Add some random points to break up grid structure
        extra = rng.uniform(-5, 5, size=(min(20, n_coarse - coarse_pts.shape[0]), dim))
        if extra.shape[0] > 0:
            coarse_pts = np.vstack([coarse_pts, extra])

        coarse_pts = coarse_pts[:n_coarse]
        for x in coarse_pts:
            if not _time_ok() or calls >= budget:
                break
            _call(x)

        # Sort coarse points by function value
        if best_f < np.inf:
            # We don't have all function values stored, so re-evaluate would waste budget.
            # Instead, we'll use a simpler approach: pick the best few random points
            # from the grid vicinity.
            pass

        # For basin selection, we'll use a few promising starting points:
        # 1. The global best
        # 2. A few random points near the best
        basin_starts = [best_x.copy()]
        for _ in range(4):
            if not _time_ok() or calls >= budget:
                break
            perturbed = best_x + rng.normal(0, 1.5, dim)
            perturbed = np.clip(perturbed, -5, 5)
            fx = _call(perturbed)
            basin_starts.append(perturbed)

        # ---------- Phase 2: CMA-ES from multiple basins ----------
        remaining = budget - calls
        if remaining <= 0 or not _time_ok():
            return best_x, best_f

        n_basins = min(5, len(basin_starts))
        # Allocate budget: more to better basins
        # Weight by inverse rank
        weights = np.array([1.0 / (i + 1) for i in range(n_basins)])
        weights /= weights.sum()
        basin_budgets = [max(10, int(remaining * w)) for w in weights[:n_basins]]
        # Adjust to not exceed remaining
        total_alloc = sum(basin_budgets)
        if total_alloc > remaining:
            scale = remaining / total_alloc
            basin_budgets = [int(b * scale) for b in basin_budgets]

        # CMA-ES implementation
        def cma_es(start, n_eval, sigma0=1.0):
            nonlocal best_x, best_f, calls
            if n_eval < 4:
                return

            x = start.copy()
            sigma = sigma0
            popsize = max(6, dim + 2 + int(3 * np.log(dim)))
            mu = popsize // 2

            # Weights
            w = np.log(np.arange(1, mu + 1))
            w -= w[-1]
            w /= w.sum()
            mu_eff = 1.0 / np.sum(w ** 2)

            # Parameters
            c_sigma = (mu_eff + 2) / (dim + mu_eff + 5)
            d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
            c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
            c_mu = (2 + (mu_eff - 2) / dim) / ((dim + 2 + 2 * mu_eff / dim) * (c_c + 1))

            C = np.eye(dim)
            C_sqrt = np.eye(dim)
            C_inv_sqrt = np.eye(dim)
            p_sigma = np.zeros(dim)
            p_c = np.zeros(dim)

            n_gen = 0
            while n_gen < n_eval and _time_ok() and calls < budget:
                if calls + popsize > budget:
                    break

                # Generate population
                offspring = []
                for _ in range(popsize):
                    z = rng.standard_normal(dim)
                    y = x + sigma * (C_sqrt @ z)
                    y = np.clip(y, -5, 5)
                    offspring.append(y)

                # Evaluate
                offspring_f = np.array([_call(y) for y in offspring])
                if not _time_ok() or calls >= budget:
                    break

                # Sort
                idx = np.argsort(offspring_f)
                offspring = np.array([offspring[i] for i in idx])
                offspring_f = offspring_f[idx]

                # Update mean
                x_new = np.sum(w[:, None] * offspring[:mu], axis=0)

                # Update paths
                diff = x_new - x
                C_inv_sqrt_diff = C_inv_sqrt @ diff / sigma
                p_sigma = (1 - c_sigma) * p_sigma + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * C_inv_sqrt_diff
                sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(p_sigma) / np.sqrt(mu_eff) - 1))
                sigma = np.clip(sigma, 1e-8, 10.0)

                p_c = (1 - c_c) * p_c + np.sqrt(c_c * (2 - c_c) * mu_eff) * C_inv_sqrt_diff

                # Update covariance
                delta_heav = (1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1)) * c_c * (2 - c_c)
                rank1 = np.outer(p_c, p_c)
                rankmu = np.sum(w[:, None, None] * np.outer(offspring[:mu] - x, offspring[:mu] - x) / sigma ** 2, axis=0)
                C = (1 - c_mu) * C + c_mu * (rank1 + (1 if np.linalg.norm(p_c) ** 2 / (c_c * (1 - c_c)) < (1 + 2 * delta_heav) * dim else 0) * rank1 + rankmu - C)

                # Periodically recompute matrix square roots
                if n_gen % 15 == 0:
                    try:
                        C_inv = np.linalg.inv(C)
                        C_inv_sqrt = np.linalg.cholesky(C_inv).T
                        C_sqrt = np.linalg.cholesky(C)
                    except:
                        C = np.eye(dim)
                        C_inv_sqrt = np.eye(dim)
                        C_sqrt = np.eye(dim)

                x = x_new
                n_gen += 1

        # Run CMA-ES from each basin
        for i, (start, nb) in enumerate(zip(basin_starts[:n_basins], basin_budgets)):
            if not _time_ok() or calls >= budget:
                break
            # Scale sigma based on basin quality (first basin gets larger sigma)
            sigma0 = 2.0 if i == 0 else 1.0
            cma_es(start, nb, sigma0)

        # ---------- Phase 3: Final polish with (1+1)-ES ----------
        if _time_ok() and calls < budget:
            remaining = budget - calls
            if remaining > 5:
                sigma = 0.1
                for _ in range(remaining):
                    if not _time_ok() or calls >= budget:
                        break
                    x = best_x + sigma * rng.standard_normal(dim)
                    x = np.clip(x, -5, 5)
                    fx = _call(x)
                    if fx < best_f:
                        sigma *= 1.2
                    else:
                        sigma *= 0.8
                    sigma = np.clip(sigma, 1e-4, 5.0)

        return best_x, best_f
# EVOLVE-BLOCK-END
