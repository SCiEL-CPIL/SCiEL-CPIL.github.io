# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Grid search + multi-start Nelder-Mead for the Discus (f11) function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = np.inf
        self.t0 = None
        self.deadline = None

    def _eval(self, func, x):
        x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
        if x.shape != (self.dim,):
            x = x.reshape(self.dim)
        f = float(func(x))
        self.calls += 1
        if not np.isfinite(f):
            f = np.inf
        if f < self.best_f:
            self.best_f = f
            self.best_x = x.copy()
        return f

    def _nm(self, func, x0, budget):
        d = self.dim
        simplex = [x0.copy()]
        f0 = self._eval(func, x0)
        for j in range(d):
            v = x0.copy()
            v[j] += 0.5 if v[j] < 0 else -0.5
            simplex.append(v)
            self._eval(func, v)
        order = np.argsort([self._eval(func, s) if False else 0 for _ in range(d + 1)])
        # recompute objective values (fresh) to be safe
        fvals = np.array([self._eval(func, s) for s in simplex])
        order = np.argsort(fvals)
        simplex = [simplex[i] for i in order]
        fvals = fvals[order]
        max_iter = 200
        for _ in range(max_iter):
            if self.calls >= budget:
                break
            if time.time() > self.deadline:
                break
            centroid = np.mean(simplex[:-1], axis=0)
            alpha = 1.0
            xr = centroid + alpha * (centroid - simplex[-1])
            fr = self._eval(func, xr)
            if fvals[0] <= fr < fvals[-2]:
                simplex[-1] = xr
                fvals[-1] = fr
            elif fr < fvals[0]:
                gamma = 2.0
                xe = centroid + gamma * (centroid - simplex[-1])
                fe = self._eval(func, xe)
                if fe < fr:
                    simplex[-1] = xe
                    fvals[-1] = fe
                else:
                    simplex[-1] = xr
                    fvals[-1] = fr
            else:
                delta = 0.5
                xc = centroid + delta * (simplex[-1] - centroid)
                fc = self._eval(func, xc)
                if fc < fvals[-1]:
                    simplex[-1] = xc
                    fvals[-1] = fc
                else:
                    for i in range(1, d + 1):
                        simplex[i] = (simplex[i] + simplex[0]) / 2.0
                        fvals[i] = self._eval(func, simplex[i])
            order = np.argsort(fvals)
            simplex = [simplex[i] for i in order]
            fvals = fvals[order]
            if np.ptp(fvals) < 1e-9:
                break
        return self.best_x.copy(), self.best_f

    def __call__(self, func):
        self.t0 = time.time()
        self.deadline = self.t0 + 300.0  # safety margin before 330s kill
        d = self.dim

        # --- Global coarse grid search ---
        # 6 points per axis; 6^5 = 7776 evaluations
        if self.budget > 7776:
            n_pts = 6
        else:
            n_pts = max(3, int(round(self.budget ** (1.0 / d))))
        grid = np.linspace(-5.0, 5.0, n_pts)
        pts = np.array(np.meshgrid(*([grid] * d)), dtype=float).reshape(-1, d)
        # shuffle for better early progress (AOCC rewards early improvement)
        idx = np.random.default_rng(0).permutation(pts.shape[0])
        grid_vals = np.full(pts.shape[0], np.inf)
        for i in idx:
            if self.calls >= self.budget:
                break
            if time.time() > self.deadline:
                break
            grid_vals[i] = self._eval(func, pts[i])
        # sort grid points by value
        gorder = np.argsort(grid_vals)

        # --- Local refinement: multi-start Nelder-Mead ---
        # Use remaining budget for local search
        remaining = self.budget - self.calls
        if remaining > 20:
            # Try up to a few distinct starts
            starts = []
            for gi in gorder:
                if len(starts) >= 3:
                    break
                starts.append(pts[gi])
            # Split remaining budget among starts
            per_start = max(20, remaining // max(1, len(starts)))
            for s in starts:
                if self.calls >= self.budget:
                    break
                if time.time() > self.deadline:
                    break
                self._nm(func, s, self.budget)

        if self.best_x is None:
            self.best_x = np.zeros(d)
            self.best_f = 0.0
        return self.best_x, self.best_f
# EVOLVE-BLOCK-END
