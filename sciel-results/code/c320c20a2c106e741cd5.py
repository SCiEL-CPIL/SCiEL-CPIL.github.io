"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _halton_point(dim, i):
    """
    Generate a deterministic point in [0, 1)^dim using the Halton sequence.
    Uses prime bases for each dimension.
    """
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    point = np.zeros(dim)
    for d in range(dim):
        base = primes[d % len(primes)]
        val = 0.0
        f = 1.0 / base
        n_val = i + 1  # 1-based index for sequence
        while n_val > 0:
            val += f * (n_val % base)
            n_val //= base
            f /= base
        point[d] = val
    return point

class Optimizer:
    """
    Global-to-Local Cascade Optimizer.
    Phase 1: Broad coverage using Halton sequence points to find good basins early (AOCC boost).
    Phase 2: Refine the best few candidates using L-BFGS-B for fast convergence on smooth/ridge functions.
    Phase 3: High-precision polish with tight tolerances if budget remains.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # Phase 1: Global Exploration with Halton Sequence
        # Use ~30-40 points for broad coverage. The first few points give immediate AOCC value.
        n_explore = 40
        n_explore = min(n_explore, budget - 1)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        candidates = [] # List of (f, x) for top candidates

        for i in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Generate Halton point
            p_unit = _halton_point(dim, i)
            p = p_unit * 10.0 - 5.0
            p = np.clip(p, -5.0, 5.0)
            
            try:
                f = func(p)
            except:
                f = np.inf
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = p.copy()
            
            candidates.append((f, p))
        
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf
            calls_made += 1
            candidates.append((best_f, best_x))

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement with L-BFGS-B
        # L-BFGS-B is robust for box-constrained problems and efficient in low dimensions.
        # We will try to refine the top K candidates.
        # L-BFGS-B typically converges in ~20-50 function evaluations for dim=5.
        
        max_starts = 5
        # Filter out candidates that are very close to already tried ones to avoid redundant work
        # Simple deduplication: keep top 5 distinct points
        unique_candidates = []
        for f, x in candidates:
            if len(unique_candidates) >= max_starts:
                break
            # Check if x is close to any already selected
            is_unique = True
            for _, ux in unique_candidates:
                if np.linalg.norm(x - ux) < 0.5:
                    is_unique = False
                    break
            if is_unique:
                unique_candidates.append((f, x))
        
        for f_init, x_init in unique_candidates:
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Custom callback to check time and budget
            def callback(xk):
                nonlocal calls_made
                calls_made += 1
                if calls_made >= budget or time.time() > deadline:
                    raise TimeoutError("Budget or time exceeded")
                return False

            try:
                res = minimize(
                    func, 
                    x_init, 
                    method='L-BFGS-B', 
                    bounds=[(-5.0, 5.0)] * dim,
                    options={'maxiter': 100, 'maxls': 50, 'disp': False},
                    callback=callback
                )
                
                # Update best if improved
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except TimeoutError:
                pass
            except Exception:
                pass

        # Phase 3: High-precision polish
        # If we have budget left, do a final tight polish on the best point found.
        if best_x is not None and calls_made < budget and time.time() < deadline:
            remaining_budget = budget - calls_made
            # Only if we have enough budget for a small polish
            if remaining_budget > 20:
                try:
                    res_polish = minimize(
                        func,
                        best_x,
                        method='L-BFGS-B',
                        bounds=[(-5.0, 5.0)] * dim,
                        options={'maxiter': 50, 'gtol': 1e-8, 'disp': False},
                        callback=callback
                    )
                    if res_polish.fun < best_f:
                        best_f = res_polish.fun
                        best_x = res_polish.x.copy()
                except:
                    pass

        return best_x, best_f
# EVOLVE-BLOCK-END
