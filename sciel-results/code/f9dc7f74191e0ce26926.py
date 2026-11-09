# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Random-restart Nelder-Mead with a hard wall-clock deadline guard."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        t0 = time.perf_counter()
        deadline = t0 + 285.0          # safety margin under the 330s kill
        calls = [0]

        def feval(x):
            calls[0] += 1
            return float(func(x))

        def nm(x0, f0):
            # Build an initial simplex around x0 (unit-sphere directions, scaled).
            pts = [x0.copy()]
            for k in range(dim):
                d = np.zeros(dim)
                d[k] = 1.0
                pts.append(np.clip(x0 + 0.5 * d, -5.0, 5.0))
            vals = [f0] + [feval(p) for p in pts[1:]]

            for _ in range(600):
                if calls[0] >= budget:
                    break
                if time.perf_counter() > deadline:
                    break
                order = np.argsort(vals)
                pts = [pts[i] for i in order]
                vals = [vals[i] for i in order]
                worst = vals[-1]
                centroid = np.mean(pts[:-1], axis=0)

                # Reflection
                xr = np.clip(centroid + (centroid - pts[-1]), -5.0, 5.0)
                fr = feval(xr)
                if vals[0] <= fr < worst:
                    pts[-1], vals[-1] = xr, fr
                    continue
                # Expansion
                if fr < vals[0]:
                    xe = np.clip(centroid + 2.0 * (xr - centroid), -5.0, 5.0)
                    fe = feval(xe)
                    if fe < fr:
                        pts[-1], vals[-1] = xe, fe
                    else:
                        pts[-1], vals[-1] = xr, fr
                    continue
                # Contractions
                done = False
                if fr < worst:  # outside contraction
                    xc = np.clip(centroid + 0.5 * (xr - centroid), -5.0, 5.0)
                    fc = feval(xc)
                    if fc <= min(fr, worst):
                        pts[-1], vals[-1] = xc, fc
                        done = True
                if not done:  # inside contraction
                    xc = np.clip(centroid - 0.5 * (pts[-1] - centroid), -5.0, 5.0)
                    fc = feval(xc)
                    if fc < worst:
                        pts[-1], vals[-1] = xc, fc
                        done = True
                if done:
                    continue
                # Shrink
                for i in range(1, dim + 1):
                    pts[i] = np.clip(pts[0] + 0.5 * (pts[i] - pts[0]), -5.0, 5.0)
                    vals[i] = feval(pts[i])

            return min(vals)

        # Phase 1: cheap random sampling to locate promising regions.
        x_best = None
        f_best = np.inf
        for _ in range(25):
            if calls[0] >= budget:
                break
            if time.perf_counter() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = feval(x)
            if fx < f_best:
                f_best, x_best = fx, x.copy()

        # Phase 2: Nelder-Mead restarts from the best samples.
        if x_best is not None:
            f_best = nm(x_best, f_best)

        for _ in range(40):
            if calls[0] >= budget:
                break
            if time.perf_counter() > deadline:
                break
            x = rng.uniform(-5.0, 5.0, dim)
            fx = feval(x)
            if fx < f_best:
                f_best = fx
                f_best = nm(x, fx)

        return x_best, f_best
# EVOLVE-BLOCK-END
