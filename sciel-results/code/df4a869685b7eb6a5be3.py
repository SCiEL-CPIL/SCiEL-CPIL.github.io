"""Differential-evolution + Nelder-Mead polish for BBOB-style 5-D box minimization."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """DE/rand/1 with adaptive F, then a Nelder-Mead polish, under a wall-clock guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        t0 = time.time()
        DEADLINE = t0 + 280.0          # safety margin well under the 300 s cap

        def alive():
            return time.time() < DEADLINE

        def clip(v):
            return np.clip(v, -5.0, 5.0)

        # --- Phase 0: cheap random sweep to seed early progress -------------------
        N = 8
        rng = np.random.default_rng(0)
        X = rng.uniform(-5.0, 5.0, (N, dim))
        X[0] = np.zeros(dim)
        F = np.empty(N)
        for i in range(N):
            F[i] = func(X[i])
        best = int(np.argmin(F))
        bestx = X[best].copy()
        bestf = F[best]

        calls = N
        if not alive() or budget <= N:
            return bestx, bestf

        # --- Phase 1: DE/rand/1 with adaptive F -----------------------------------
        remaining = budget - N
        pop = 10
        maxgen = max(1, remaining // pop)
        Fm = 0.8
        Fmin, Fmax = 0.4, 1.2
        Fhist = []

        for gen in range(maxgen):
            if not alive():
                break
            for i in range(pop):
                if not alive() or calls >= budget:
                    break
                # pick three distinct indices different from i
                a, b, c = rng.choice(np.delete(np.arange(pop), i), size=3, replace=False)
                v = X[i] + Fm * (X[b] - X[c])
                v = clip(v)
                fv = func(v)
                calls += 1
                if fv < F[i]:
                    X[i], F[i] = v, fv
                    if fv < bestf:
                        bestf, bestx = fv, v.copy()
                Fhist.append(Fm)
                if len(Fhist) > 50:
                    Fhist.pop(0)
                # adapt mutation factor toward the recent successful value
                if len(Fhist) >= 2:
                    Fm = np.clip(Fm + 0.05 * (Fm - float(np.mean(Fhist))), Fmin, Fmax)

        # --- Phase 2: Nelder-Mead polish from the best point -----------------------
        if alive() and calls < budget:
            try:
                res = minimize(
                    func, bestx, method="Nelder-Mead",
                    options={
                        "maxiter": max(1, (budget - calls) // 2),
                        "xatol": 1e-4,
                        "fatol": 1e-7,
                        "adaptive": True,
                    },
                )
                if res.x is not None:
                    rx = clip(np.asarray(res.x, dtype=float))
                    rf = float(func(rx))
                    calls += 1
                    if rf < bestf:
                        bestx, bestf = rx, rf
            except Exception:
                pass

        return bestx, bestf
# EVOLVE-BLOCK-END
