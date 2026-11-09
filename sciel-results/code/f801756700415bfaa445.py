"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np


class Optimizer:
    """
    Optimizer for Bent Cigar function f12.
    Strategy:
    1. Brief global exploration using Sobol points to find a good starting basin.
    2. (1+1)-ES with sigma adaptation (1.5 on success, 0.87 on failure) for
       efficient local refinement on anisotropic Bent Cigar landscape.
    3. Final high-precision coordinate-wise polish pass.
    4. Strict budget and deadline management.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.perf_counter()
        deadline = t0 + 280.0  # 20s safety margin before 300s limit
        d = self.dim
        budget = self.budget

        state = {"count": 0, "ok": True, "best_x": None, "best_f": np.inf}

        def wf(x):
            if not state["ok"]:
                return 1e300
            if time.perf_counter() > deadline:
                state["ok"] = False
                return 1e300

            x = np.asarray(x, dtype=float)
            if not np.all(np.isfinite(x)):
                return 1e300

            try:
                v = float(func(x))
            except Exception:
                return 1e300

            state["count"] += 1
            if not np.isfinite(v):
                return 1e300

            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x.copy()

            return v

        # --- Phase 1: Global Exploration (Sobol) ---
        n_sobol = min(20, budget // 10)

        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=d, scramble=True, seed=42)
            sobol_points = sampler.random(n_sobol)
            sobol_points = -5.0 + 10.0 * sobol_points

            best_idx = 0
            best_val = np.inf

            for i in range(n_sobol):
                if not state["ok"]:
                    break
                x = sobol_points[i]
                v = wf(x)
                if v < best_val:
                    best_val = v
                    best_idx = i

            x0 = sobol_points[best_idx].copy()
        except Exception:
            rng = np.random.default_rng(0)
            x0 = rng.uniform(-5, 5, d)
            v = wf(x0)
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x0.copy()

        # --- Phase 2: (1+1)-ES with sigma adaptation ---
        remaining_budget = budget - state["count"]

        if remaining_budget > 0 and state["ok"]:
            try:
                rng = np.random.default_rng(12345)
                x_curr = state["best_x"].copy()
                f_curr = state["best_f"]
                sigma = 0.5  # initial step size
                success_streak = 0
                failure_streak = 0

                while state["ok"] and state["count"] < budget:
                    # Generate candidate
                    z = rng.standard_normal(d)
                    x_cand = x_curr + sigma * z
                    x_cand = np.clip(x_cand, -5.0, 5.0)
                    f_cand = wf(x_cand)

                    if not np.isfinite(f_cand):
                        f_cand = 1e300

                    if f_cand < f_curr - 1e-15:
                        # Success: accept and increase sigma
                        x_curr = x_cand
                        f_curr = f_cand
                        success_streak += 1
                        failure_streak = 0
                        sigma = min(sigma * 1.5, 10.0)

                        if f_curr < state["best_f"]:
                            state["best_f"] = f_curr
                            state["best_x"] = x_curr.copy()
                    else:
                        # Failure: decrease sigma
                        failure_streak += 1
                        success_streak = 0
                        sigma = max(sigma * 0.87, 1e-10)

                    # Early stopping if sigma is too small
                    if sigma < 1e-8 and failure_streak > 20:
                        break

                    # Check time
                    if time.perf_counter() > deadline:
                        state["ok"] = False
                        break

            except Exception:
                pass

        # --- Phase 3: Final high-precision coordinate-wise polish ---
        remaining_budget = budget - state["count"]
        if remaining_budget > 2 and state["ok"] and state["best_x"] is not None:
            try:
                x_best = state["best_x"].copy()
                f_best = state["best_f"]

                # Try small perturbations along each coordinate with adaptive step sizes
                step_sizes = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7]
                
                for _ in range(min(remaining_budget // (d * len(step_sizes)), 5)):
                    if not state["ok"] or time.perf_counter() > deadline:
                        break
                    improved = False
                    for j in range(d):
                        if not state["ok"] or time.perf_counter() > deadline:
                            break
                        for delta in step_sizes:
                            for sign in [1, -1]:
                                if not state["ok"] or time.perf_counter() > deadline:
                                    break
                                x_try = x_best.copy()
                                x_try[j] += sign * delta
                                x_try = np.clip(x_try, -5.0, 5.0)
                                v_try = wf(x_try)
                                if v_try < f_best - 1e-15:
                                    x_best = x_try
                                    f_best = v_try
                                    improved = True
                                    break
                        if improved:
                            break
                    if not improved:
                        break

                if f_best < state["best_f"]:
                    state["best_f"] = f_best
                    state["best_x"] = x_best
            except Exception:
                pass

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
