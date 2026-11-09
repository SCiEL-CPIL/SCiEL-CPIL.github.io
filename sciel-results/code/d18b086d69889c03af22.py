"""Black-box continuous optimizer on BBOB-style f6 (Attractive sector).

Adaptive-step evolutionary strategy with a lightweight rotation/axis
adaptation (rank-1 "memory" of the search direction) plus a final polish
pass.  Stays in the (1+1)-ES family (cheap, many early evaluations, good
for AOCC) while aligning exploration with the function's anisotropic
geometry.  No full CMA-ES population (observed to be too slow / risky here).
"""
# EVOLVE-BLOCK-START
import time

import numpy as np


class Optimizer:
    """Adaptive-step ES with rank-1 direction memory + polish + deadline guard."""

    def __init__(self, budget, dim):
        self.budget = int(budget)
        self.dim = int(dim)

    # ------------------------------------------------------------------ #
    # helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _clip(x):
        return np.clip(x, -5.0, 5.0)

    # ------------------------------------------------------------------ #
    # main
    # ------------------------------------------------------------------ #
    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 310.0          # safety margin before the 330 s kill
        dim = self.dim
        B = int(self.budget)

        calls = 0
        best_x = None
        best_f = np.inf

        def f(x):
            nonlocal calls, best_x, best_f
            calls += 1
            v = func(x)
            if v < best_f:
                best_f = v
                best_x = x
            return v

        rng = np.random.default_rng(20240501)

        # ---- seed: box centre + a 5-point star to initialise the memory ---- #
        c = np.zeros(dim)
        fc = f(c)
        base = 2.5
        pts = [c.copy()]
        for i in range(dim):
            p = c.copy(); p[i] = base
            pts.append(f(self._clip(p)))
            p = c.copy(); p[i] = -base
            pts.append(f(self._clip(p)))

        # pick the best star point as the start
        star = [self._clip(c + base * np.eye(dim)[i]) for i in range(dim)]
        star += [self._clip(c - base * np.eye(dim)[i]) for i in range(dim)]
        vals = [f(p) for p in star]
        k = int(np.argmin(vals))
        x = star[k].copy()
        fx = vals[k]

        # ---- rank-1 direction memory (rotation adaptation, no eigendecomp) ---- #
        M = np.eye(dim)
        alpha = 0.5          # memory strength  (0 -> pure isotropic)
        beta = 0.95          # memory decay

        sigma = 1.5
        sigma_min = 1e-4
        sigma_max = 4.5

        # polish: switch to a fine local step for the last ~20% of the budget
        polish_at = int(0.80 * B)
        polish_sigma = 0.03

        it = 0
        while calls < B:
            # deadline guard
            if time.time() > deadline:
                break

            if it >= polish_at:
                s = polish_sigma
            else:
                s = sigma

            # sample from an anisotropic (rank-1 biased) normal, clipped to box
            u = rng.standard_normal(dim)
            d = (1.0 - alpha) * u + alpha * (M @ u)
            y = self._clip(x + s * d)
            fy = f(y)

            it += 1

            if fy < fx:
                # productive step: reinforce the direction in memory
                step = (y - x) / (s + 1e-12)
                M = beta * M + (1.0 - beta) * np.outer(step, step)
                x, fx = y, fy
                if it < polish_at:
                    sigma = min(sigma * 1.25, sigma_max)
            else:
                if it < polish_at:
                    sigma = max(sigma * 0.87, sigma_min)

        return best_x, best_f
# EVOLVE-BLOCK-END
