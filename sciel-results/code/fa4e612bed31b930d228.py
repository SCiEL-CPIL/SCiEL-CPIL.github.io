"""Hybrid multi-start optimizer: grid + random exploration with Nelder-Mead and coordinate descent refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Hybrid optimizer combining Nelder-Mead with coordinate descent for BBOB f22."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        calls = 0
        best_f = float('inf')
        best_x = np.zeros(dim)

        def eval_and_track(x):
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return best_f
            calls += 1
            try:
                val = func(np.asarray(x, dtype=float))
                if not np.isfinite(val):
                    val = best_f
                if val < best_f:
                    best_f = val
                    best_x = np.asarray(x, dtype=float).copy()
                return val
            except Exception:
                return best_f

        rng = np.random.default_rng(42)

        def golden_section_1d(x, axis, lo, hi, max_iter=15):
            """Optimize a single coordinate using golden-section search."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return x

            gr = (1.0 + math.sqrt(5.0)) / 2.0
            inv_gr = 1.0 / gr
            inv_gr2 = 1.0 / (gr * gr)

            x1 = x.copy()
            x1[axis] = lo + inv_gr2 * (hi - lo)
            x2 = x.copy()
            x2[axis] = lo + inv_gr * (hi - lo)

            f1 = eval_and_track(x1)
            f2 = eval_and_track(x2)

            for _ in range(max_iter):
                if calls >= budget or time.time() > deadline:
                    break
                if f1 < f2:
                    hi = x2[axis]
                    x2 = x1.copy()
                    f2 = f1
                    x1 = x.copy()
                    x1[axis] = lo + inv_gr2 * (hi - lo)
                    f1 = eval_and_track(x1)
                else:
                    lo = x1[axis]
                    x1 = x2.copy()
                    f1 = f2
                    x2 = x.copy()
                    x2[axis] = lo + inv_gr * (hi - lo)
                    f2 = eval_and_track(x2)

            if f1 < f2:
                return x1
            else:
                return x2

        def coordinate_descent(x, n_sweeps=3, search_radius=1.0, max_iter_per_dim=12):
            """Perform coordinate descent with golden-section on each dimension."""
            nonlocal calls, best_f, best_x
            for sweep in range(n_sweeps):
                if calls >= budget or time.time() > deadline:
                    break
                order = rng.permutation(dim)
                for axis in order:
                    if calls >= budget or time.time() > deadline:
                        break
                    lo = max(-5.0, x[axis] - search_radius)
                    hi = min(5.0, x[axis] + search_radius)
                    x = golden_section_1d(x, axis, lo, hi, max_iter=max_iter_per_dim)
            return x

        def nelder_mead(x0, max_evals):
            """Run Nelder-Mead from x0 with a limited evaluation budget."""
            nonlocal calls, best_f, best_x
            if calls >= budget or time.time() > deadline:
                return
            try:
                from scipy.optimize import minimize

                res = minimize(
                    eval_and_track,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': max(30, max_evals // 2),
                        'maxfev': max_evals,
                        'xatol': 1e-5,
                        'fatol': 1e-7,
                        'adaptive': True,
                    },
                    bounds=[(-5, 5)] * dim,
                )
            except Exception:
                pass

        # Phase 1: Structured grid search for early progress
        # Evaluate a coarse grid to quickly find promising regions
        grid_points = []
        # Create a 3x3x3x3x3 grid (243 points) - too many, use 2x2x2x2x2 = 32
        # Actually, let's use a stratified approach: evaluate center and corners
        # First, evaluate the center and a few key points
        center = np.zeros(dim)
        grid_points.append(center)

        # Add points at +/-2 in each dimension from center
        for i in range(dim):
            for sign in [-1, 1]:
                p = center.copy()
                p[i] = sign * 2.0
                grid_points.append(p)

        # Add a few more structured points
        for i in range(dim):
            for j in range(i + 1, dim):
                p = center.copy()
                p[i] = 2.0
                p[j] = 2.0
                grid_points.append(p)
                p2 = center.copy()
                p2[i] = -2.0
                p2[j] = -2.0
                grid_points.append(p2)

        # Evaluate grid points
        grid_results = []
        for p in grid_points:
            if calls >= budget or time.time() > deadline:
                break
            p_clipped = np.clip(p, -5, 5)
            f_val = eval_and_track(p_clipped)
            grid_results.append((f_val, p_clipped.copy()))

        # Phase 2: Random sampling to explore the space
        n_random = max(20, budget // 5)
        random_results = []
        for i in range(n_random):
            if calls >= budget or time.time() > deadline:
                break
            x = rng.uniform(-5, 5, size=dim)
            f_val = eval_and_track(x)
            random_results.append((f_val, x.copy()))

        # Combine all candidates
        all_candidates = grid_results + random_results
        all_candidates.sort(key=lambda t: t[0])

        # Phase 3: Local refinement using Nelder-Mead on top candidates
        top_k = min(6, len(all_candidates))
        for i in range(top_k):
            if calls >= budget or time.time() > deadline:
                break
            x0 = all_candidates[i][1].copy()
            remaining = budget - calls
            if remaining < 15:
                break
            # Allocate budget for NM: use up to 40% of remaining or 80 evals
            nm_budget = min(80, remaining // 2)
            nelder_mead(x0, nm_budget)

        # Phase 4: Coordinate descent refinement on the best point
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            remaining = budget - calls
            if remaining > 20:
                # Coarse coordinate descent
                x = coordinate_descent(x, n_sweeps=3, search_radius=1.5, max_iter_per_dim=12)
                f_val = eval_and_track(x)

                # Fine coordinate descent
                if calls < budget and time.time() < deadline:
                    x = coordinate_descent(x, n_sweeps=2, search_radius=0.5, max_iter_per_dim=10)
                    f_val = eval_and_track(x)

        # Phase 5: If budget remains, do random restarts with adaptive step size
        if calls < budget and time.time() < deadline:
            step = 0.8
            while calls < budget and time.time() < deadline:
                x = best_x + rng.normal(0, step, size=dim)
                x = np.clip(x, -5, 5)
                f_val = eval_and_track(x)
                if f_val < best_f:
                    step = max(0.1, step * 0.85)
                else:
                    step = min(2.5, step * 1.15)

        return best_x, best_f
# EVOLVE-BLOCK-END
