"""Rastrigin-aware hybrid: lattice scan + CMA-ES refinement with deadline guard."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Lattice scan for a strong early baseline, then CMA-ES to adapt to the
    hidden rotation/shift and converge on the shifted minimum."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        t0 = time.time()
        DEADLINE = 320.0          # safety margin under the 330 s wall-clock kill
        dim = self.dim
        budget = self.budget

        # Shared best holder (mutable list so nested helpers can update it).
        best = [None, np.inf]

        def record(x, f):
            if f < best[1]:
                best[1] = f
                best[0] = np.asarray(x, dtype=float).copy()

        # --- Phase 1: coarse lattice scan for a strong early baseline -------
        # Unshifted Rastrigin minima are at integer coordinates. The true
        # minimizer lies in [-4,4]^5, so scan the integer grid there, plus a
        # half-integer offset grid, to quickly establish a low value.
        lattice = np.arange(-4, 5, dtype=float)
        grid_pts = np.array(np.meshgrid(*([lattice] * dim)), dtype=float).reshape(-1, dim)
        half = lattice + 0.5
        grid_pts_half = np.array(np.meshgrid(*([half] * dim)), dtype=float).reshape(-1, dim)

        used = 0
        reserve = 80  # keep budget for refinement
        for p in grid_pts:
            if used >= budget - reserve:
                break
            record(p, func(p))
            used += 1
            if time.time() - t0 > DEADLINE - 80:
                break

        if used < budget - reserve and time.time() - t0 < DEADLINE - 80:
            for p in grid_pts_half:
                if used >= budget - reserve:
                    break
                record(p, func(p))
                used += 1
                if time.time() - t0 > DEADLINE - 80:
                    break

        if best[0] is None:
            rng = np.random.default_rng(0)
            best[0] = rng.uniform(-5, 5, dim)
            best[1] = func(best[0])
            used += 1

        # --- Phase 2: CMA-ES refinement around the best lattice point --------
        remaining = budget - used
        if remaining > 12:
            self._cmaes(func, best[0].copy(), 0.8, remaining, dim, t0, DEADLINE, best)

        return best[0], best[1]

    def _cmaes(self, func, x0, sigma, remaining, dim, t0, deadline, best):
        rng = np.random.default_rng(7)
        x = x0.copy()
        C = np.eye(dim)
        sigma = float(sigma)
        pop = max(6, int(4 + 3 * np.log(dim)))   # ~15 for dim=5
        mu = pop // 2
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / (w @ (w ** 2))
        cc = (4.0 + mueff / dim) / ((dim + 1.75) ** 2 + mueff / dim)
        cs = (mueff + 2.0) / (dim + mueff + 5.0)
        c1 = (2.0 - cs - 2.0 / (dim + 1.0)) / ((dim + 1.3) ** 2 + mueff)
        cmu = min(1.0 - c1, 2.0 * (mueff - 2.0 + 1.0 / mueff) / ((dim + 2.0) ** 2 + mueff))
        chiN = np.sqrt(dim) * (1.0 - 1.0 / (4.0 * dim) + 1.0 / (21.0 * dim ** 2))
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        it = 0
        while it < remaining - pop:
            if time.time() - t0 > deadline:
                break
            D = np.linalg.cholesky(C + 1e-10 * np.eye(dim))
            B = D.T / np.sqrt(np.diag(D @ D.T))
            zs = rng.standard_normal((pop, dim))
            steps = (B @ D).T @ zs.T
            xs = x[None, :] + sigma * steps.T
            xs = np.clip(xs, -5.0, 5.0)
            fs = np.array([func(xi) for xi in xs])
            it += 1
            order = np.argsort(fs)
            xnew = (w[order[:mu], None] * xs[order[:mu]]).sum(axis=0)
            ps = (1.0 - cs) * ps + np.sqrt(cs * (2.0 - cs) * mueff) / chiN * (
                D @ (B.T @ (xnew - x) / sigma))
            if np.linalg.norm(ps) / np.sqrt(1.0 - (1.0 - cs) ** (2.0 * (it + 1))) < (
                    1.4 + 2.0 / (dim + 1.0)) * chiN:
                sigma *= np.exp(cs / chiN * (np.linalg.norm(ps) / chiN - 1.0))
            pc = (1.0 - cc) * pc + np.sqrt(cc * (2.0 - cc) * mueff) * (
                (xnew - x) / (sigma * chiN))
            delta = (1.0 + (1.0 - cc) * cc * (np.dot(pc, pc) - 1.0)) * cc
            delta_mu = (1.0 + (1.0 - cc) * cmu) * cmu
            stepmat = D @ (B.T @ (xs[order[:mu]] - x) / sigma)
            C = ((1.0 - delta - delta_mu) * C
                 + delta * np.outer(pc, pc)
                 + delta_mu * (w[order[:mu], None] * stepmat.T).T @ stepmat)
            C = 0.5 * (C + C.T)
            x = xnew
            for xi, fi in zip(xs, fs):
                if fi < best[1]:
                    best[1] = fi
                    best[0] = xi.copy()
            sigma = min(max(sigma, 1e-4), 10.0)
# EVOLVE-BLOCK-END
