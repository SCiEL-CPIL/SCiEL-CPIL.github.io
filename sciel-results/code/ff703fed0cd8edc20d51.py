"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-stage optimizer for BBOB-style Sphere (rotated, shifted, offset).

    Sphere is convex, so once CMA-ES finds the basin it descends monotonically.
    AOCC rewards early progress, so we open with a cheap global Latin-Hypercube
    spread (incl. the origin, since shifted minima are often near center),
    hand the best point to a rotation-adaptive CMA-ES, then polish with
    L-BFGS-B under a tight tolerance. An explicit wall-clock deadline guard
    returns the best-so-far with a safety margin before the 330 s limit.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    # ---- helpers ---------------------------------------------------------
    def _lhs_sample(self, n, dim, rng):
        """Latin Hypercube Sampling in [0,1)^dim."""
        samples = np.zeros((n, dim))
        for d in range(dim):
            perm = rng.permutation(n)
            samples[:, d] = (perm + rng.random(n)) / n
        return samples

    def _cmaes(self, func, x0, sigma0, budget, deadline, state):
        """Minimal CMA-ES. Returns (best_x, best_f, calls_made)."""
        dim = self.dim
        N = max(2 * dim + 2, 8)
        mu = max(1, N // 2)
        w = np.log(np.arange(mu, 0, -1)) - np.log(mu - 1)
        w /= w.sum()
        mu_eff = 1.0 / (w @ w)
        c_sigma = (mu_eff + 2) / ((dim + 1) ** 2 + mu_eff)
        c_c = (4 + mu_eff / dim) / (dim + 4 + 2 * mu_eff / dim)
        d_sigma = 1 + 2 * max(0, np.sqrt((mu_eff - 1) / (dim + 1)) - 1) + c_sigma
        c_1 = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (
            1 - c_sigma + 2 * max(0, (mu_eff - 1) / (dim + 1)) - c_sigma
        )
        c_mu = (2 / ((dim + 1.3) ** 2 + mu_eff)) * (
            min(1 - c_1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((dim + 2) ** 2 + mu_eff))
        )
        chi_N = np.sqrt(dim) * (1 - 1 / (4 * dim) + 1 / (21 * dim ** 2))

        pop = np.tile(x0, (N, 1))
        fvals = np.empty(N)
        for i in range(N):
            fvals[i] = state["count"] and 0  # placeholder, overwritten below
        # evaluate initial population
        for i in range(N):
            fvals[i] = self._eval(func, pop[i], state)
        order = np.argsort(fvals)
        mean = pop[order[:mu]].T @ w
        C = np.eye(dim)
        pc = np.zeros(dim)
        ps = np.zeros(dim)
        sigma = sigma0
        best_x = pop[order[0]].copy()
        best_f = fvals[order[0]]

        while state["budget"] > 0:
            if time.time() >= deadline:
                break
            # sample
            for i in range(N):
                if state["budget"] <= 0 or time.time() >= deadline:
                    break
                B = np.linalg.cholesky(C + 1e-12 * np.eye(dim))
                z = rng.standard_normal(dim)
                y = mean + sigma * (B @ z)
                pop[i] = np.clip(y, -5, 5)
            # evaluate
            for i in range(N):
                fvals[i] = self._eval(func, pop[i], state)
                if fvals[i] < best_f:
                    best_f = fvals[i]
                    best_x = pop[i].copy()
            if state["budget"] <= 0 or time.time() >= deadline:
                break
            # recompute order
            order = np.argsort(fvals)
            xw = pop[order[:mu]].T @ w
            # update ps, pc, C
            inv_sigma = 1.0 / sigma if sigma > 1e-12 else 1.0
            ps = (1 - c_sigma) * ps + np.sqrt(c_sigma * (2 - c_sigma) * mu_eff) * inv_sigma * (B @ ((xw - mean) / sigma))
            if np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * state["it"])) < (1.4 + 2 / (dim + 1)) * chi_N:
                pc = (1 - c_c) * pc + c_c * (xw - mean) / sigma
            hsig = (np.linalg.norm(ps) / np.sqrt(1 - (1 - c_sigma) ** (2 * state["it"]))) / chi_N < 1.7
            C = (1 + c_1 + c_mu - c_1 - c_1 * (dim + 1)) * C
            C += c_1 * (np.outer(pc, pc) - C)
            C += c_mu * (np.outer(xw - mean, xw - mean) / sigma ** 2 - C)
            if not hsig:
                C += c_1 * (np.outer(pc, pc) - C) * 0  # keep symmetric
            C = 0.5 * (C + C.T)
            mean = xw
            sigma *= np.exp(c_sigma / d_sigma * (np.linalg.norm(ps) / chi_N - 1))
            sigma = min(max(sigma, 1e-12), 1e6)
            state["it"] += 1
        return best_x, best_f

    def _eval(self, func, x, state):
        state["calls"] += 1
        state["budget"] -= 1
        return float(func(x))

    # ---- main ------------------------------------------------------------
    def __call__(self, func):
        dim = self.dim
        total = self.budget
        deadline = time.time() + 320.0  # safety margin before 330 s kill
        rng = np.random.default_rng(0)
        state = {"calls": 0, "budget": total, "it": 0}

        # Stage 1: Latin Hypercube global spread (incl. origin) for early AOCC.
        n_lhs = 31
        grid = self._lhs_sample(n_lhs, dim, rng) * 10.0 - 5.0
        grid = np.vstack([np.zeros(dim), grid])
        best_x, best_f = None, np.inf
        for x in grid:
            if state["budget"] <= 0 or time.time() >= deadline:
                break
            fx = self._eval(func, x, state)
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
        if best_x is None:
            return np.zeros(dim), np.inf

        # Stage 2: rotation-adaptive CMA-ES on the best point.
        if state["budget"] > 0 and time.time() < deadline:
            bx, bf = self._cmaes(func, best_x, 1.0, state["budget"], deadline, state)
            if bf < best_f:
                best_f, best_x = bf, bx

        # Stage 3: tight L-BFGS-B polish (bounded, cheap for convex Sphere).
        if state["budget"] > 0 and time.time() < deadline:
            remaining = state["budget"]
            try:
                res = minimize(
                    lambda v: self._eval(func, np.clip(v, -5, 5), state),
                    best_x,
                    method="L-BFGS-B",
                    bounds=[(-5.0, 5.0)] * dim,
                    options={"maxiter": 400, "maxfun": remaining, "ftol": 1e-15, "gtol": 1e-12},
                )
                if res.fun < best_f:
                    best_f, best_x = float(res.fun), np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
