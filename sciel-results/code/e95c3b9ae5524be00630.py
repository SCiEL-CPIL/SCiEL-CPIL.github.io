"""Fresh design for BBOB-style f12 (bent cigar): global sampling + coordinate search + local polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Global population sampler + dimension-wise coordinate search + local polish."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    @staticmethod
    def _goldmin(func, lo, hi, x0, fx0, deadline, state):
        """Golden-section 1-D minimization; returns (x_best, f_best) using state budget."""
        g = (np.sqrt(5.0) - 1.0) / 2.0
        inv_g = 1.0 / g
        a, b = lo, hi
        x1 = a + (1.0 - g) * (b - a)
        x2 = a + g * (b - a)
        x1 = min(max(x1, x0), b)
        x2 = min(max(x2, x0), b)
        if abs(x1 - x0) < 1e-12:
            f1 = fx0
        else:
            f1 = func(x1)
            state["used"] += 1
        if abs(x2 - x0) < 1e-12:
            f2 = fx0
        else:
            f2 = func(x2)
            state["used"] += 1
        for _ in range(28):
            if state["used"] >= state["budget"] or time.time() > deadline:
                break
            if f1 <= f2:
                b, x2, f2 = x2, x1, f1
                x1 = a + (1.0 - g) * (b - a)
                x1 = min(max(x1, x0), b)
                if abs(x1 - x0) < 1e-12:
                    f1 = fx0
                else:
                    f1 = func(x1)
                    state["used"] += 1
            else:
                a, x1, f1 = x1, x2, f2
                x2 = a + g * (b - a)
                x2 = min(max(x2, x0), b)
                if abs(x2 - x0) < 1e-12:
                    f2 = fx0
                else:
                    f2 = func(x2)
                    state["used"] += 1
        if f1 <= f2:
            return x1, f1
        return x2, f2

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285.0
        state = {"used": 0, "budget": budget}
        t0 = time.time()

        def fvec(v):
            if state["used"] >= state["budget"]:
                return None
            state["used"] += 1
            return func(v)

        # 1) Global Latin-hypercube sampling for strong early progress.
        m = 40
        perm = np.arange(m)
        idx = np.arange(m)
        for d in range(dim):
            np.random.shuffle(perm)
            idx[perm] = np.arange(m)
        grid = idx.astype(float) / m * 10.0 - 5.0
        best_x, best_f = None, np.inf
        for i in range(m):
            fv = fvec(grid[i])
            if fv is None:
                break
            if fv < best_f:
                best_f, best_x = fv, grid[i].copy()
        if best_x is None:
            return None

        # 2) Dimension-wise coordinate search (walks the cigar valley cheaply).
        for _ in range(6):
            if state["used"] >= state["budget"] or time.time() > deadline:
                break
            improved = False
            for d in range(dim):
                if state["used"] >= state["budget"] or time.time() > deadline:
                    break
                x0 = best_x.copy()
                fx0 = best_f
                lo, hi = -5.0, 5.0
                xb, fb = self._goldmin(fvec, lo, hi, x0, fx0, deadline, state)
                if fb < fx0 - 1e-12:
                    x0[d] = xb
                    best_x[d] = xb
                    best_f = fb
                    improved = True
            if not improved:
                break

        # 3) Local polish with a tiny adaptive step.
        step = 0.25
        rng = np.random.default_rng(12345)
        for _ in range(120):
            if state["used"] >= state["budget"] or time.time() > deadline:
                break
            cand = best_x + step * rng.standard_normal(dim)
            cand = np.clip(cand, -5.0, 5.0)
            fv = fvec(cand)
            if fv is None:
                break
            if fv < best_f:
                best_x, best_f = cand, fv
            else:
                step *= 0.8
        return best_x, best_f
# EVOLVE-BLOCK-END
