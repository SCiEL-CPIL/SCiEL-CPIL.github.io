"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-start optimizer combining L-BFGS-B, Nelder-Mead, and (1+1)-ES.
    Strategy:
    1. Generate diverse starting points (center, grid, random).
    2. For each start, run L-BFGS-B to find a local minimum.
    3. Polish the global best with Nelder-Mead.
    4. Use remaining budget for (1+1)-ES refinement with adaptive step size.
    5. Final polish with tight tolerances if budget allows.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # 20s safety margin
        dim = self.dim
        budget = self.budget
        calls = [0]

        def wrapped(x):
            calls[0] += 1
            if calls[0] > budget:
                raise RuntimeError("budget exceeded")
            return float(func(x))

        rng = np.random.default_rng(42)

        # Generate a set of diverse starting points
        starts = []
        # Center
        starts.append(np.zeros(dim))
        # Random points
        for _ in range(15):
            starts.append(rng.uniform(-5, 5, dim))
        # Some structured points (axis-aligned)
        for i in range(dim):
            for val in [-4, -2, 0, 2, 4]:
                pt = np.zeros(dim)
                pt[i] = val
                starts.append(pt)

        # Deduplicate approximately
        unique_starts = []
        for s in starts:
            if not any(np.allclose(s, u, atol=0.1) for u in unique_starts):
                unique_starts.append(s)
        starts = unique_starts

        best_x = None
        best_f = np.inf

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_x = x.copy()
                best_f = f

        # Sort starts: center first, then random, then structured
        # We'll just shuffle to ensure diversity in order but keep center at front
        rng.shuffle(starts)
        # Move a zero-like point to the front if present
        for i, s in enumerate(starts):
            if np.allclose(s, 0, atol=0.1):
                starts[0], starts[i] = starts[i], starts[0]
                break

        # Phase 1: Multi-start L-BFGS-B
        # We will run L-BFGS-B on each start, but limit the number of evaluations per run
        # to ensure we visit many starts.
        # Budget is 300. If we have ~30 starts, we can afford ~10 evals each.
        # L-BFGS-B usually converges quickly for smooth functions.
        
        for s in starts:
            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

            s = np.clip(s, -5, 5)

            # Try L-BFGS-B
            try:
                # Limit maxiter and maxls to prevent runaway
                # maxfun is not directly exposed in minimize options for L-BFGS-B in older scipy,
                # but maxiter limits iterations. Each iteration may call func multiple times (line search).
                # To be safe, we rely on the global budget check in wrapped().
                res = minimize(
                    wrapped,
                    s,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={'maxiter': 50, 'maxls': 20, 'gtol': 1e-4, 'ftol': 1e-10}
                )
                update_best(res.x, res.fun)
            except Exception:
                pass

            if calls[0] >= budget:
                break
            if time.time() > deadline:
                break

        # Phase 2: Polish the best point with Nelder-Mead
        if best_x is not None and calls[0] < budget:
            try:
                # Use a portion of remaining budget for NM
                # NM uses dim+1 evals for simplex setup, then 1 per iter.
                remaining = budget - calls[0]
                nm_maxiter = min(100, max(10, remaining // 2))
                
                res_nm = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': nm_maxiter, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
                )
                update_best(res_nm.x, res_nm.fun)
            except Exception:
                pass

        # Phase 3: (1+1)-ES refinement from the best point
        if best_x is not None and calls[0] < budget:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start with small step since we are polishing
            no_improve = 0
            max_no_improve = 50
            
            while calls[0] < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                try:
                    fy = wrapped(y)
                except Exception:
                    break
                
                if fy < fx:
                    x, fx = y, fy
                    update_best(x, fx)
                    sigma *= 1.1
                    no_improve = 0
                else:
                    sigma *= 0.9
                    no_improve += 1
                
                if no_improve > max_no_improve:
                    break
                if sigma < 1e-6:
                    break

        # Phase 4: Final tight polish if budget allows
        if best_x is not None and calls[0] < budget - 10:
            try:
                # Very tight tolerances
                res_final = minimize(
                    wrapped,
                    best_x,
                    method='Nelder-Mead',
                    options={'maxiter': 50, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
                )
                update_best(res_final.x, res_final.fun)
            except Exception:
                pass

        if best_x is None:
            # Fallback
            best_x = rng.uniform(-5, 5, dim)
            best_f = wrapped(best_x)

        return best_x, best_f
# EVOLVE-BLOCK-END
