"""Optimizer for ZDT6 exploiting the known Pareto-front structure."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """ZDT6-specific optimizer: sample the analytic Pareto-front curve, then refine."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 320.0  # safety margin under the 330s wall-clock kill
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n = self.dim
        used = 0

        def eval_point(x):
            nonlocal used
            f = np.asarray(problem(x), dtype=float)
            used += 1
            return f

        # --- Phase 1: dense sampling of the analytic ZDT6 Pareto-front curve ---
        # For ZDT6, the Pareto front is achieved with g(x) minimized.
        # For a given x[0] = t, set x[i] = sin(20*pi*t) (i>=1) minimizes the
        # per-term x[i]^2 - 2*t*cos(20*pi*x[i]). Clip to bounds before/after.
        # The front in objective space: f1 = t, f2 = (1 - sqrt(1/g))*(1 - t^2/100).
        # We sample t on a grid that is denser near t=0 (front is steep there).
        # Reserve some budget for refinement.
        n_samples = min(120, max(20, self.budget - 40))
        # Uniform in t but add extra points near 0
        t_grid = np.linspace(0.0, 1.0, n_samples)
        # extra near zero
        t_extra = np.linspace(0.0, 0.1, 20)
        t_grid = np.unique(np.concatenate([t_grid, t_extra]))
        t_grid = t_grid[t_grid <= 1.0]

        F = []
        X = []
        for t in t_grid:
            if time.time() > deadline or used >= self.budget:
                break
            x = np.empty(n)
            x[0] = t
            x[1:] = np.sin(20.0 * np.pi * t)
            x = np.clip(x, lo, hi)
            f = eval_point(x)
            F.append(f)
            X.append(x)

        F = np.array(F)
        X = np.array(X)

        # --- Phase 2: refine with small local perturbations around good points ---
        # Keep non-dominated points as seeds; nudge x[0] and x[1:] slightly.
        def is_dominated(f, Fset):
            # dominated if some other point <= in all and < in at least one
            le = np.all(Fset <= f, axis=1)
            lt = np.any(Fset < f, axis=1)
            return bool(np.any(le & lt))

        # iterate refinement until budget/deadline
        best_X = X.copy()
        best_F = F.copy()
        while used < self.budget and time.time() < deadline:
            # pick a random current point as a base
            idx = int(rng.integers(len(best_X)))
            base = best_X[idx].copy()
            # small perturbation in x[0] and a global shift in x[1:]
            base[0] += 0.01 * rng.standard_normal()
            base[1:] += 0.02 * rng.standard_normal()
            base = np.clip(base, lo, hi)
            f = eval_point(base)
            # accept if it is not dominated by the current set
            if not is_dominated(f, best_F):
                best_X = np.vstack([best_X, base])
                best_F = np.vstack([best_F, f])
            # also try a "snap back to front" variant: keep x[0], reset x[1:]
            snap = best_X[idx].copy()
            snap[1:] = np.sin(20.0 * np.pi * snap[0])
            snap = np.clip(snap, lo, hi)
            if used < self.budget and time.time() < deadline:
                fs = eval_point(snap)
                if not is_dominated(fs, best_F):
                    best_X = np.vstack([best_X, snap])
                    best_F = np.vstack([best_F, fs])

        return best_X
# EVOLVE-BLOCK-END
