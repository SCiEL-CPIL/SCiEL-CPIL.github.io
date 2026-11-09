"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Optimizer for Bent Cigar function f12.
    Strategy:
    1. Global exploration using Sobol points to find a good starting basin.
    2. Local refinement using Powell's method (derivative-free, adaptive step sizes per dimension)
       which is effective for anisotropic functions like Bent Cigar.
    3. Strict budget and deadline management.
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
        # Use a small number of deterministic Sobol points to cover the space.
        # This helps if the initial random start is far from the optimum.
        # Bent Cigar is unimodal but shifted/rotated, so finding the basin is key.
        # 20 points is cheap and provides a better start than a single point.
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
            # Fallback to random if Sobol fails
            rng = np.random.default_rng(0)
            x0 = rng.uniform(-5, 5, d)
            v = wf(x0)
            if v < state["best_f"]:
                state["best_f"] = v
                state["best_x"] = x0.copy()

        # --- Phase 2: Local Refinement (Powell) ---
        # Powell is robust for anisotropic functions because it adapts step sizes 
        # along the principal axes of the function's contours.
        remaining_budget = budget - state["count"]
        
        if remaining_budget > 0 and state["ok"]:
            try:
                res = minimize(
                    wf,
                    x0,
                    method="Powell",
                    options={
                        "maxiter": 1000,
                        "maxfev": remaining_budget,
                        "xrtol": 1e-10,
                        "ftol": 1e-10,
                    },
                )
                # Update best if Powell found something better
                xf = np.asarray(res.x, dtype=float)
                ff = res.fun
                if np.all(np.isfinite(xf)) and np.isfinite(ff):
                    if ff < state["best_f"]:
                        state["best_f"] = ff
                        state["best_x"] = np.clip(xf, -5.0, 5.0)
            except Exception:
                pass

        # --- Phase 3: Fine-tuning with (1+1)-ES ---
        # If we have budget left and time, do a simple (1+1)-ES to polish the result.
        # This can help if Powell got stuck in a local minimum or converged too early.
        remaining_budget = budget - state["count"]
        if remaining_budget > 0 and state["ok"] and state["best_x"] is not None:
            rng = np.random.default_rng(12345)
            x = state["best_x"].copy()
            fx = state["best_f"]
            sigma = 0.1  # Start with a small step size for fine-tuning
            
            # Adapt sigma based on success
            # For Bent Cigar, we want to keep sigma large enough to move along the flat axis
            # but small enough to converge on the steep axis.
            
            for _ in range(remaining_budget):
                if time.perf_counter() > deadline:
                    break
                
                # Generate candidate
                y = x + sigma * rng.standard_normal(d)
                y = np.clip(y, -5, 5)
                
                fy = wf(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    # Increase sigma to explore further
                    sigma *= 1.2
                else:
                    # Decrease sigma to refine
                    sigma *= 0.85

                if fx < state["best_f"]:
                    state["best_f"] = fx
                    state["best_x"] = x.copy()

        return state["best_x"], state["best_f"]
# EVOLVE-BLOCK-END
