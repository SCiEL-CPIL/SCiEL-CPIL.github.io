"""Surrogate-assisted adaptive refinement optimizer for BBOB f22 in 5D.

Strategy:
  Phase 1: Evaluate a Sobol space-filling sequence (64 points) to build a
           coarse model of the landscape. This gives strong early AOCC progress
           by quickly locating the best basin among the 21 peaks.
  Phase 2: Iterative surrogate-guided refinement. Build an IDW-based surrogate
           from all evaluations, predict the minimum location, and evaluate
           there. Repeat until budget is exhausted or deadline.
  Phase 3: L-BFGS-B polish from the best point for fast local convergence.
  A wall-clock deadline guard (320 s) returns the best result found safely early.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Surrogate-assisted adaptive refinement optimizer for the BBOB f22 task."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        deadline = time.time() + 320.0  # safety margin before 330 s kill
        n = self.dim
        budget = self.budget
        calls = {"count": 0}

        def fwrap(x):
            if time.time() >= deadline:
                raise TimeoutError
            calls["count"] += 1
            if calls["count"] > budget:
                raise ValueError("Budget exceeded")
            return func(x)

        best = {"x": None, "f": np.inf}

        def update_best(x, f):
            if f < best["f"]:
                best["f"] = f
                best["x"] = x.copy()

        # Store all evaluations for surrogate building
        eval_points = []
        eval_values = []

        def record(x, f):
            eval_points.append(x.copy())
            eval_values.append(f)
            update_best(x, f)

        # ---------------- Phase 1: Sobol space-filling initial design ----------------
        # Evaluate a Sobol sequence for space-filling exploration
        sobol_n = 64
        sampler = qmc.Sobol(d=n, scramble=True, seed=42)
        sobol_pts = sampler.random(sobol_n)  # in [0,1]^n
        sobol_pts = sobol_pts * 10.0 - 5.0   # map to [-5,5]^n

        for i in range(sobol_n):
            if time.time() >= deadline or calls["count"] >= budget:
                break
            try:
                fi = fwrap(sobol_pts[i])
                record(sobol_pts[i], fi)
            except (TimeoutError, ValueError):
                break

        if best["x"] is None:
            x0 = np.zeros(n)
            try:
                best["f"] = fwrap(x0)
                record(x0, best["f"])
            except:
                return (None, np.inf)

        # ---------------- Phase 2: Surrogate-guided adaptive refinement ----------------
        # Use IDW (inverse distance weighted) surrogate to predict minimum location
        # and iteratively refine
        refinement_budget = min(200, budget - calls["count"])
        n_refine = 0

        while n_refine < refinement_budget:
            if time.time() >= deadline or calls["count"] >= budget:
                break

            X = np.array(eval_points)
            Y = np.array(eval_values)

            # Build IDW surrogate: predict value at candidate points
            # Use a grid of candidate points around the current best
            # to find where the surrogate predicts the lowest value

            # Candidate generation: local refinement around best + global candidates
            # Local: jitter around best point
            local_sigma = max(0.1, 1.0 / (1 + n_refine * 0.05))
            n_local = 30
            n_global = 10

            candidates = []
            # Local candidates around best
            for _ in range(n_local):
                cand = best["x"] + local_sigma * np.random.default_rng(n_refine * 1000 + _).standard_normal(n)
                cand = np.clip(cand, -5.0, 5.0)
                candidates.append(cand)
            # Global candidates from Sobol
            for _ in range(n_global):
                cand = np.random.default_rng(n_refine * 10000 + _).uniform(-5.0, 5.0, size=n)
                candidates.append(cand)

            candidates = np.array(candidates)

            # Evaluate surrogate at candidates using IDW
            # IDW: f(x) = sum(w_i * y_i) / sum(w_i), w_i = 1 / (dist_i + eps)^power
            power = 2.0
            eps = 1e-6
            # Compute distances from each candidate to all evaluated points
            # Shape: (n_cand, n_eval)
            dists = np.linalg.norm(candidates[:, np.newaxis, :] - X[np.newaxis, :, :], axis=2)
            weights = 1.0 / (dists + eps) ** power
            # Avoid division by zero issues
            surrogate_vals = (weights * Y[np.newaxis, :]).sum(axis=1) / weights.sum(axis=1)

            # Select the candidate with the lowest surrogate prediction
            best_cand_idx = np.argmin(surrogate_vals)
            next_point = candidates[best_cand_idx]

            try:
                fi = fwrap(next_point)
                record(next_point, fi)
                n_refine += 1
            except (TimeoutError, ValueError):
                break

        # ---------------- Phase 3: L-BFGS-B polish ----------------
        bounds = [(-5.0, 5.0) for _ in range(n)]

        if best["x"] is not None and calls["count"] < budget:
            try:
                res = minimize(fwrap, best["x"].copy(), method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 100, 'maxfun': 100})
                if res.fun < best["f"]:
                    best["f"] = res.fun
                    best["x"] = res.x.copy()
            except (TimeoutError, ValueError):
                pass
            except Exception:
                pass

        # Try a few more diverse starting points if budget remains
        remaining_starts = min(3, max(0, budget - calls["count"]))
        for _ in range(remaining_starts):
            if time.time() >= deadline or calls["count"] >= budget:
                break

            start_x = best["x"] + np.random.default_rng(_ * 777).standard_normal(n) * 0.5
            start_x = np.clip(start_x, -5.0, 5.0)

            try:
                res = minimize(fwrap, start_x, method='L-BFGS-B',
                               bounds=bounds, options={'maxiter': 50, 'maxfun': 50})
                if res.fun < best["f"]:
                    best["f"] = res.fun
                    best["x"] = res.x.copy()
            except (TimeoutError, ValueError):
                break
            except Exception:
                continue

        # Final safety
        if best["x"] is None:
            x0 = np.zeros(n)
            best["f"] = fwrap(x0)
            best["x"] = x0

        return best["x"], best["f"]
