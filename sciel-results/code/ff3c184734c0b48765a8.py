"""Multi-phase optimizer: LHS Global Search + L-BFGS-B Refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Optimizer for BBOB f9 (Rosenbrock rotated).
    Strategy:
    1. Latin Hypercube Sampling (LHS) for diverse global coverage.
    2. Multi-start L-BFGS-B on top candidates.
    3. Final high-precision L-BFGS-B polish.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 285  # Safety margin before 300s limit

        best_x = None
        best_f = np.inf
        call_count = 0

        def safe_func(x):
            nonlocal call_count
            if call_count >= budget:
                return 1e10
            x_clipped = np.clip(x, -5, 5)
            val = func(x_clipped)
            call_count += 1
            return val

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        # --- Phase 1: Latin Hypercube Sampling (LHS) ---
        # LHS provides better coverage than random sampling in low dimensions (5D)
        # and handles the unknown rotation better than axis-aligned heuristics.
        # We use a fixed number of points for the global phase.
        n_lhs = 80  # 80 calls for global exploration
        if budget < n_lhs + 50:
            n_lhs = max(10, budget // 2)
        
        rng = np.random.default_rng(42)
        
        # Generate LHS points
        # For each dimension, divide [-5, 5] into n_lhs equal intervals.
        # Pick one random point in each interval, then shuffle.
        lhs_points = np.zeros((n_lhs, dim))
        for i in range(dim):
            # Intervals: [-5, -5+10/n], [-5+10/n, ...]
            edges = np.linspace(-5.0, 5.0, n_lhs + 1)
            # Random point in each interval
            random_offsets = rng.random(n_lhs)
            points_in_dim = edges[:-1] + random_offsets * (10.0 / n_lhs)
            # Shuffle to break correlation
            lhs_points[:, i] = rng.permutation(points_in_dim)
        
        # Evaluate all LHS points
        candidates = []
        for pt in lhs_points:
            if time.time() > deadline or call_count >= budget:
                break
            f_val = safe_func(pt)
            if not np.isnan(f_val):
                candidates.append((f_val, pt.copy()))
                update_best(pt, f_val)
        
        # --- Phase 2: Multi-start L-BFGS-B on Top Candidates ---
        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Select top K candidates for refinement
        k_top = 10
        top_candidates = [pt for _, pt in candidates[:k_top]]
        
        # Ensure the global best is in the list
        if best_x is not None:
             if not any(np.allclose(pt, best_x) for pt in top_candidates):
                 top_candidates.append(best_x.copy())
        
        # Allocate budget for refinement
        # Reserve some budget for the final polish
        # Rough estimate: L-BFGS-B on Rosenbrock needs ~10-20 calls to converge locally
        # We have ~420 calls left. 10 candidates * 30 calls = 300 calls.
        # Remaining ~120 calls for final polish.
        
        # Dynamic budget allocation per candidate
        # If we have many candidates, give each less budget.
        # If we have few, give each more.
        calls_used_phase1 = call_count
        remaining_budget = budget - calls_used_phase1
        # Reserve 20% for final polish
        polish_budget = max(20, int(remaining_budget * 0.2))
        refinement_budget = remaining_budget - polish_budget
        
        if refinement_budget > 0 and top_candidates:
            maxfev_per_candidate = max(10, int(refinement_budget / len(top_candidates)))
            
            for x0 in top_candidates:
                if time.time() > deadline or call_count >= budget:
                    break
                
                # Skip if we've already found a very good solution and this start is far
                # This heuristic helps avoid wasting time on bad local minima
                if best_f < 1.0 and best_x is not None:
                    dist = np.linalg.norm(x0 - best_x)
                    if dist > 4.0:
                        continue

                try:
                    res = minimize(
                        safe_func, 
                        x0, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 50, 'maxfev': maxfev_per_candidate}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass

        # --- Phase 3: Final High-Precision Polish ---
        if best_x is not None and time.time() < deadline and call_count < budget:
            remaining = budget - call_count
            if remaining > 5:
                try:
                    res = minimize(
                        safe_func, 
                        best_x, 
                        method='L-BFGS-B', 
                        bounds=[(-5, 5)] * dim,
                        options={'maxiter': 200, 'maxfev': remaining}
                    )
                    if res.fun < best_f:
                        update_best(res.x, res.fun)
                except Exception:
                    pass
        
        # Fallback
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = func(best_x)
            call_count += 1

        return best_x, best_f
# EVOLVE-BLOCK-END
