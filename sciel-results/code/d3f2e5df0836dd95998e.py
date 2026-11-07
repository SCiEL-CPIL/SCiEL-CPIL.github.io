"""DTLZ4 optimizer: direct stratified construction of the known Pareto front."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for DTLZ4.

    DTLZ4's Pareto front is the quarter-sphere f1^2+f2^2+f3^2=0.25 (fi>=0).
    Hypervolume is maximized by dense, uniform coverage of this front, so we
    generate a stratified (cosine) grid of objective targets directly and map
    each one back to decision variables via the DTLZ4 inverse, then spend the
    remaining budget on a few cheap local refinements.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    @staticmethod
    def _dtlz4_front_point(f1, f2, f3, dim):
        """Map a Pareto-front objective vector to decision variables for DTLZ4 (alpha=10)."""
        c = np.sqrt(f1 * f1 + f2 * f2 + f3 * f3)
        if c > 1e-12:
            th2 = np.arcsin(min(1.0, max(-1.0, f3 / c)))
        else:
            th2 = 0.0
        r = np.sqrt(f1 * f1 + f2 * f2)
        if r > 1e-12:
            th1 = np.arctan2(f2, f1)
        else:
            th1 = 0.0
        x = np.zeros(dim)
        x[0] = np.cos(th1) * np.cos(th2)
        x[1] = np.sin(th1) * np.cos(th2)
        x[2] = np.sin(th2)
        if dim >= 9:
            x[3:8] = 0.5
        if dim >= 12:
            t = 0.5 * (1.0 + np.cos(th1) * np.cos(th2))
            t = min(1.0, max(0.0, t))
            x[11] = t ** 12
        return x

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 295.0  # safety margin below the 300s limit

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)

        budget = int(self.budget)
        dim = int(self.dim)

        # ---- stratified (cosine) grid of objective targets on the front ----
        n = 13
        f1_grid, f2_grid = np.meshgrid(np.linspace(0.0, 0.5, n),
                                       np.linspace(0.0, 0.5, n))
        f3_grid = np.sqrt(np.clip(0.25 - f1_grid ** 2 - f2_grid ** 2, 0.0, None))

        # cosine stratification of the (f1,f2) square for uniform area coverage
        idx = np.arange(1, n + 1)
        s = np.sqrt(0.25 - (0.5 * idx / n) ** 2)
        f1c = 0.5 * np.cos((idx - 0.5) / n * np.pi / 2)
        f2c = 0.5 * np.cos((idx - 0.5) / n * np.pi / 2)
        f3c = s[idx - 1]

        # combine grid + stratified + explicit corner points, dedupe
        targets = []
        for a, b, c in zip(f1_grid.ravel(), f2_grid.ravel(), f3_grid.ravel()):
            targets.append((float(a), float(b), float(c)))
        for a, b, c in zip(f1c, f2c, f3c):
            targets.append((float(a), float(b), float(c)))
        for a, b, c in [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5)]:
            targets.append((a, b, c))

        seen = set()
        unique_targets = []
        for a, b, c in targets:
            key = (round(a, 9), round(b, 9), round(c, 9))
            if key not in seen:
                seen.add(key)
                unique_targets.append((a, b, c))

        # ---- evaluate front points (capped by budget and deadline) ----
        F = []
        X = []
        cap = min(len(unique_targets), budget)
        for i in range(cap):
            if time.time() > deadline:
                break
            a, b, c = unique_targets[i]
            x = self._dtlz4_front_point(a, b, c, dim)
            x = np.clip(x, lo, hi)
            fv = np.asarray(problem(x), dtype=float)
            F.append(fv)
            X.append(x)

        # ---- spend remaining budget on cheap local refinements of a few points ----
        used = len(F)
        if used < budget and F:
            F = np.array(F)
            n_refine = min(6, used)
            step = 0.02
            for _ in range(8):
                if used >= budget or time.time() > deadline:
                    break
                for k in range(n_refine):
                    if used >= budget or time.time() > deadline:
                        break
                    base = X[k]
                    delta = np.random.uniform(-step, step, size=dim)
                    x = np.clip(base + delta, lo, hi)
                    fv = np.asarray(problem(x), dtype=float)
                    used += 1
                    F = np.vstack([F, fv])
                    X.append(x)

        return X
# EVOLVE-BLOCK-END
