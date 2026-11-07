"""Model-based lattice search exploiting Rastrigin 2π-periodicity."""
# EVOLVE-BLOCK-START
import time
import numpy as np

_BUDGET = 500
_LO, _HI = -5.0, 5.0
_TOL = 1e-10
_EPS = 1e-6


class Optimizer:
    """Rastrigin lattice search: estimate rotation, snap to minima, refine."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def _estimate_rotation(self, func, rng):
        d = self.dim
        x0 = np.zeros(d)
        fx0 = func(x0)
        pts = [x0]
        vals = [fx0]
        for _ in range(3 * d):
            x = rng.uniform(_LO, _HI, d)
            pts.append(x)
            vals.append(func(x))
        pts = np.asarray(pts)
        vals = np.asarray(vals)
        G = np.zeros((d, d))
        for k in range(d):
            for j in range(d):
                a = np.zeros(d)
                b = np.zeros(d)
                a[k] = _EPS
                b[j] = _EPS
                G[k, j] = (func(pts[0] + a) - fx0) * (func(pts[0] + b) - fx0)
        U, _, Vt = np.linalg.svd(G)
        dsgn = np.sign(np.linalg.det(Vt.T) * np.linalg.det(U))
        R = U @ (dsgn * np.diag(np.ones(d))) @ Vt
        return R, pts, vals

    def _snap(self, R, x):
        u = R @ x
        u = np.round(u / (2.0 * np.pi)) * (2.0 * np.pi)
        return R.T @ u

    def _refine(self, func, x, fx, budget_left, rng, deadline):
        d = self.dim
        sigma = 1.0
        for _ in range(budget_left):
            if time.time() > deadline:
                break
            y = np.clip(x + sigma * rng.standard_normal(d), _LO, _HI)
            fy = func(y)
            if fy < fx:
                x, fx = y, fy
                sigma *= 1.3
            else:
                sigma *= 0.85
            if sigma < _TOL:
                break
        return x, fx

    def __call__(self, func):
        start = time.time()
        deadline = start + 250.0
        rng = np.random.default_rng(0)
        d = self.dim

        R, pts, vals = self._estimate_rotation(func, rng)
        base = min(vals)
        base_idx = int(np.argmin(vals))
        best_x = pts[base_idx].copy()
        best_f = base

        cands = []
        for p in pts:
            c = self._snap(R, p)
            c = np.clip(c, _LO, _HI)
            cands.append(c)
        seen = {}
        for c in cands:
            key = np.round(c, 6).tobytes()
            if key not in seen:
                seen[key] = c
        cands = list(seen.values())
        cands.sort(key=lambda c: float(np.linalg.norm(c - best_x)))
        cands = cands[:8]

        for c in cands:
            if time.time() > deadline:
                break
            fc = func(c)
            if fc < best_f:
                best_f, best_x = fc, c.copy()

        used = _BUDGET
        left = max(0, _BUDGET - used)
        if left > 0:
            best_x, best_f = self._refine(func, best_x, best_f, left, rng, deadline)

        return best_x, best_f
# EVOLVE-BLOCK-END
