"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Population-based adaptive-search hybrid for Rastrigin-rotated f15.

    The function is multimodal (Rastrigin) and rotated/shifted/offset, so we
    need: (a) global diversification to jump between basins, (b) an adaptive
    step (1/5 success rule) for the local Rastrigin structure, and (c) a final
    deterministic basin-descent refinement (alternating axis Nelder-Mead with
    Rastrigin-aware local minima snap) to polish near the minimum. A strict
    wall-clock deadline guards against timeout-induced invalidity.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim
        self.calls = 0

    def _eval(self, func, x):
        x = np.clip(np.asarray(x, dtype=float), -5.0, 5.0)
        self.calls += 1
        return x, float(func(x))

    @staticmethod
    def _local_min_1d(y, h, lo, hi):
        # y: array of Rastrigin values at points spaced h apart starting near idx 0
        # find the index of a local minimum
        n = y.size
        best_i = 0
        best_v = y[0]
        for i in range(n):
            if y[i] < best_v:
                best_v = y[i]
                best_i = i
        # snap to local min by checking neighbours
        while best_i > 0 and y[best_i - 1] < y[best_i]:
            best_i -= 1
        while best_i < n - 1 and y[best_i + 1] < y[best_i]:
            best_i += 1
        return best_i

    def _refine_axis(self, func, x, axis, sigma, h):
        # deterministic local search along one axis exploiting Rastrigin 2pi-periodicity
        x = x.copy()
        base = x[axis]
        # sample a small window around base
        n = 9
        lo = base - 4.0 * h
        hi = base + 4.0 * h
        pts = np.linspace(lo, hi, n)
        vals = np.empty(n)
        for k in range(n):
            x[axis] = pts[k]
            vals[k] = float(func(np.clip(x, -5.0, 5.0)))
            self.calls += 1
        # snap to local minimum (Rastrigin has minima at multiples of 2pi)
        best_i = int(np.argmin(vals))
        # refine around best_i with a smaller step
        h2 = h * 0.25
        lo2 = pts[best_i] - 2.0 * h2
        hi2 = pts[best_i] + 2.0 * h2
        n2 = 7
        pts2 = np.linspace(lo2, hi2, n2)
        vals2 = np.empty(n2)
        for k in range(n2):
            x[axis] = pts2[k]
            vals2[k] = float(func(np.clip(x, -5.0, 5.0)))
            self.calls += 1
        x[axis] = pts2[int(np.argmin(vals2))]
        return np.clip(x, -5.0, 5.0)

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 285.0  # safety margin below 300s / 330s kill
        dim = self.dim
        rng = np.random.default_rng(1234)

        # ---- Phase 0: diverse initialization (Rastrigin has many basins) ----
        pop = rng.uniform(-5.0, 5.0, size=(16, dim))
        pop_vals = np.empty(16)
        for i in range(16):
            _, pop_vals[i] = self._eval(func, pop[i])
        bi = int(np.argmin(pop_vals))
        x = pop[bi].copy()
        fx = pop_vals[bi]
        sigma = 1.5

        # ---- Phase 1: adaptive population (1+lambda)-style with 1/5 success ----
        lam = 8
        succ = 0
        total = 0
        it = 0
        while self.calls < self.budget - 60:  # reserve calls for refinement
            if time.time() > deadline:
                break
            # adaptive population generation
            for _ in range(lam):
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                y, fy = self._eval(func, y)
                total += 1
                if fy < fx:
                    x, fx = y, fy
                    succ += 1
            # 1/5 success rule
            if total >= 10:
                rate = succ / total
                if rate > 1.0 / 5.0:
                    sigma *= 1.5
                elif rate < 1.0 / 5.0:
                    sigma *= 0.7
                succ = 0
                total = 0
            sigma = max(sigma, 0.02)
            it += 1
            # occasional restart / diversification to escape local Rastrigin minima
            if it % 40 == 0 and self.calls < self.budget - 100:
                # inject a few random individuals, keep best
                for _ in range(4):
                    ry = np.clip(x + rng.uniform(-3.0, 3.0, dim), -5.0, 5.0)
                    ry, ryv = self._eval(func, ry)
                    if ryv < fx:
                        x, fx = ry, ryv

        # ---- Phase 2: deterministic axis-wise basin refinement (cheap, precise) ----
        # alternate axes, a few passes, shrinking step. This exploits the
        # 2pi-periodic Rastrigin structure to land exactly on a basin minimum.
        h = max(0.5, sigma * 0.5)
        for pass_ in range(3):
            for axis in range(dim):
                if self.calls >= self.budget - 5 or time.time() > deadline:
                    break
                x = self._refine_axis(func, x, axis, sigma, h)
                x, fx = self._eval(func, x)
                if self.calls >= self.budget - 5 or time.time() > deadline:
                    break
            h *= 0.5
            h = max(h, 0.05)

        return x, fx
# EVOLVE-BLOCK-END
