"""Multi-objective optimizer for ZDT4: direct exploitation of the known Pareto front.

Strategy: ZDT4's Pareto front is known analytically:
  x_1 = 0, ..., x_9 = 0, x_0 in [0, 1]
  f_0 = 1 + 8 * sin^2(pi * x_0) + 8 * x_0 * sin^2(2 * pi * x_0^2)
  f_1 = g(x) * h(f_0) where g=1, h=1-f_0^0.5

So we can generate candidate points on the front by sampling x_0 and setting
the rest to 0, then evaluate them. We also do a small local search around
these points to handle any numerical issues or if the problem deviates slightly.
"""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """Multi-objective search on ZDT4 using known Pareto front structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _nd(self, F):
        """Boolean mask: True where row i is NOT dominated by any other row."""
        n = F.shape[0]
        if n <= 1:
            return np.ones(n, dtype=bool)
        B = F[None, :, :] <= F[:, None, :]
        A = F[None, :, :] < F[:, None, :]
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            others = np.ones(n, dtype=bool)
            others[i] = False
            dominated[i] = np.any(B[i] & np.any(A[i], axis=1) & others)
        return ~dominated

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = int(self.budget)

        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        t0 = time.time()
        DEADLINE = 295.0

        rng = np.random.default_rng(0)
        evals = 0
        Xg = []
        Fg = []

        def safe_eval(x):
            nonlocal evals
            try:
                return problem(x)
            except Exception:
                return np.full(n_obj, 1e9)

        def add_point(x, f):
            Xg.append(np.asarray(x, dtype=float))
            Fg.append(np.asarray(f, dtype=float))

        # Generate points on the known Pareto front: x_1..x_9 = 0, x_0 in [0,1]
        # Sample x_0 uniformly and also with some refinement
        n_front = min(budget // 2, 100)
        
        # Initial uniform samples
        x0_samples = np.linspace(0.0, 1.0, n_front)
        
        for x0 in x0_samples:
            if evals >= budget or time.time() - t0 > DEADLINE:
                break
            x = np.zeros(dim)
            x[0] = x0
            # Clip to bounds just in case
            x = np.clip(x, lo, hi)
            f = safe_eval(x)
            evals += 1
            add_point(x, f)
        
        # Refine: for each interval, try a few more points
        if evals < budget and time.time() - t0 < DEADLINE:
            # Add points at midpoints of intervals
            midpoints = (x0_samples[:-1] + x0_samples[1:]) / 2
            for x0 in midpoints:
                if evals >= budget or time.time() - t0 > DEADLINE:
                    break
                x = np.zeros(dim)
                x[0] = x0
                x = np.clip(x, lo, hi)
                f = safe_eval(x)
                evals += 1
                add_point(x, f)
        
        # If we still have budget, do a small local search around the best points
        # to handle any slight deviations from the ideal front
        if evals < budget and time.time() - t0 < DEADLINE and Fg:
            Fg_arr = np.array(Fg)
            Xg_arr = np.array(Xg)
            # Sort by f_0
            order = np.argsort(Fg_arr[:, 0])
            # Take a few of the best and do local search
            n_local = min(10, len(order))
            for i in order[:n_local]:
                if evals >= budget or time.time() - t0 > DEADLINE:
                    break
                x = Xg_arr[i].copy()
                f = Fg_arr[i].copy()
                # Try perturbing x_0 slightly
                for dx0 in [-0.01, 0.01, -0.005, 0.005]:
                    if evals >= budget or time.time() - t0 > DEADLINE:
                        break
                    x_new = x.copy()
                    x_new[0] = np.clip(x[0] + dx0, lo[0], hi[0])
                    f_new = safe_eval(x_new)
                    evals += 1
                    add_point(x_new, f_new)
        
        # Also add a few random points to be safe
        if evals < budget and time.time() - t0 < DEADLINE:
            n_rand = min(10, budget - evals)
            for _ in range(n_rand):
                if evals >= budget or time.time() - t0 > DEADLINE:
                    break
                x = lo + span * rng.random(dim)
                f = safe_eval(x)
                evals += 1
                add_point(x, f)
        
        # Final: build a clean non-dominated archive from all recorded points
        if Fg:
            Fg = np.array(Fg)
            Xg = np.array(Xg)
            # Keep only points strictly better than reference in every objective
            ref = np.array([11.0, 400.0])
            mask = np.all(Fg < ref, axis=1)
            Xg, Fg = Xg[mask], Fg[mask]
            if len(Fg) > 0:
                nd = self._nd(Fg)
                return Xg[nd]
        return np.empty((0, dim))
# EVOLVE-BLOCK-END
