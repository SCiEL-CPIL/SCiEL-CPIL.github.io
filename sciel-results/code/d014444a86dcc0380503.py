"""Combined optimizer for BBOB f11 (Discus): Multi-restart L-BFGS-B with diverse initialization and polish."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """Multi-restart L-BFGS-B black-box minimizer on [-5,5]^dim."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start = time.time()
        deadline = start + 280.0  # safety margin within 300s wall-clock

        # ---- call counter + wrapper ----
        state = {"n": 0}
        best_x = None
        best_f = np.inf

        def wrapped(x):
            nonlocal best_x, best_f
            state["n"] += 1
            # clip to box and ensure finite length-dim vector
            x = np.asarray(x, dtype=float).reshape(-1)
            if x.shape[0] != dim:
                if x.shape[0] > dim:
                    x = x[:dim]
                else:
                    x = np.pad(x, (0, dim - x.shape[0]))
            x = np.clip(x, -5.0, 5.0)
            
            # Check budget before calling to avoid exceeding
            if state["n"] > budget:
                # Return last known best or inf to signal stop
                return best_f if np.isfinite(best_f) else np.inf
            
            try:
                f_val = float(func(x))
            except Exception:
                return np.inf
            
            if np.isfinite(f_val) and f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            
            return f_val

        rng = np.random.default_rng(42)
        bounds = [(-5.0, 5.0)] * dim

        # ---- Generate diverse starting points ----
        # 1. Origin (often good for BBOB)
        start_points = [np.zeros(dim)]
        
        # 2. Axis-aligned points (covers directions)
        for i in range(dim):
            for sign in [-1, 1]:
                p = np.zeros(dim)
                p[i] = sign * 3.0
                start_points.append(p)
        
        # 3. Random points to ensure global coverage
        # We want about 10-15 distinct starting points
        n_random = 10
        for _ in range(n_random):
            start_points.append(rng.uniform(-4.0, 4.0, dim))

        # Deduplicate (approximate)
        unique_starts = []
        for p in start_points:
            is_dup = False
            for u in unique_starts:
                if np.linalg.norm(p - u) < 1e-5:
                    is_dup = True
                    break
            if not is_dup:
                unique_starts.append(p)
        
        start_points = unique_starts

        # ---- Multi-restart L-BFGS-B ----
        # We will iterate through start points.
        # To maximize progress, we give each restart a portion of the budget.
        # However, since we don't know which is best, we interleave or just go sequentially?
        # Sequentially is simpler. If the first one finds a good minimum, it might take most budget.
        # To prevent this, we can limit maxiter per restart or use a dynamic budget.
        
        # Strategy: Run L-BFGS-B from each start point.
        # We'll use a fixed maxiter that is reasonable for 5D, e.g., 50-100.
        # Total budget is 500. If we have ~15 starts, 500/15 ~ 33 calls each.
        # L-BFGS-B in 5D is very fast. 33 calls is usually enough to converge locally.
        
        maxiter_per_restart = 30
        maxls_per_restart = 20
        
        for x0 in start_points:
            if state["n"] >= budget - 10: # Leave some budget for polish
                break
            if time.time() > deadline:
                break
            
            # Define a local function that respects the global budget
            def local_func(x):
                return wrapped(x)
            
            try:
                res = minimize(
                    local_func,
                    x0,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": maxiter_per_restart,
                        "maxls": maxls_per_restart,
                        "ftol": 1e-12,
                        "gtol": 1e-8,
                        "eps": 1e-8
                    }
                )
                # The wrapper already updates best_x and best_f
            except Exception:
                pass

        # ---- Phase 2: Polish the best point found ----
        # If we have budget left, do a high-precision polish on the current best.
        if best_x is not None and state["n"] < budget and time.time() < deadline:
            try:
                res = minimize(
                    wrapped,
                    best_x,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={
                        "maxiter": 100,
                        "maxls": 50,
                        "ftol": 1e-15,
                        "gtol": 1e-12,
                        "eps": 1e-10
                    }
                )
                # Wrapper updates best
            except Exception:
                pass
        
        # Final check: if we have very little budget left, just do a few random steps?
        # No, L-BFGS-B is better.
        
        return best_x, best_f
# EVOLVE-BLOCK-END
