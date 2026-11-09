"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def _halton_point(dim, n, i):
    """
    Generate a deterministic point in [0, 1)^dim using the Halton sequence.
    This ensures diverse, broad coverage without needing SciPy's QMC.
    Uses prime bases for each dimension.
    """
    # First 10 primes for up to 10 dimensions
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
    Phase 2: Refine the best few candidates using BFGS (quasi-Newton) for fast convergence on smooth/ridge functions.
    """

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin before 300s limit
        dim = self.dim
        budget = self.budget
        
        # We will use a portion of the budget for global exploration (Phase 1)
        # and the rest for local exploitation (Phase 2).
        # For dim=5, BFGS is very fast. Let's use ~50 points for exploration.
        n_explore = 50
        # Ensure we don't exceed budget
        n_explore = min(n_explore, budget - 1)
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        candidates = [] # List of (f, x) for top candidates

        # Phase 1: Global Exploration with Halton Sequence
        # Map [0, 1)^dim to [-5, 5]^dim
        for i in range(n_explore):
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Generate Halton point
            p_unit = _halton_point(dim, n_explore, i)
            p = p_unit * 10.0 - 5.0
            
            # Clip to box (though Halton is in [0,1), so it's safe, but just in case of floating point)
            p = np.clip(p, -5.0, 5.0)
            
            try:
                f = func(p)
            except:
                f = np.inf
            calls_made += 1
            
            if f < best_f:
                best_f = f
                best_x = p.copy()
            
            # Keep track of top 5 candidates for multi-start refinement
            candidates.append((f, p))
        
        if best_x is None:
            # Fallback if we ran out of time immediately (unlikely)
            best_x = np.zeros(dim)
            try:
                best_f = func(best_x)
            except:
                best_f = np.inf
            calls_made += 1
            candidates.append((best_f, best_x))

        # Sort candidates by fitness
        candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Refinement with BFGS
        # Select top K candidates to refine.
        # BFGS uses gradient approximations if analytic gradient is not provided, 
        # which costs more function calls but is robust.
        # For dim=5, BFGS is efficient.
        
        # We have remaining budget = budget - calls_made
        # Let's try to refine the top 3 candidates.
        # BFGS in scipy typically uses ~10-20 * dim calls for convergence.
        # With dim=5, that's ~50-100 calls per start.
        # We have ~450 calls left. So 4-5 starts is feasible.
        
        max_starts = 5
        starts_to_try = candidates[:max_starts]
        
        # To avoid redundant work, we can skip starts that are very close to each other.
        # But for simplicity and robustness, we just try the top ones.
        
        for f_init, x_init in starts_to_try:
            if time.time() > deadline or calls_made >= budget:
                break
            
            # Define a wrapper to count calls and enforce deadline
            # Note: scipy minimize might make many calls. We need to be careful.
            # Since we can't easily interrupt scipy mid-optimization without a custom callback,
            # we rely on the deadline check in the loop and the fact that BFGS is fast.
            
            # Custom callback to check time and budget
            def callback(xk):
                nonlocal calls_made
                calls_made += 1
                # If we exceed budget or time, raise an exception to stop scipy
                if calls_made >= budget or time.time() > deadline:
                    raise TimeoutError("Budget or time exceeded")
                return False

            try:
                res = minimize(
                    func, 
                    x_init, 
                    method='BFGS', 
                    options={'maxiter': 100, 'disp': False},
                    callback=callback
                )
                
                if res.success and res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except TimeoutError:
                # If we hit the limit, we just stop. The last valid best_x is kept.
                # We might have a partial result from the last iteration, but scipy
                # doesn't expose that easily without more complex hooks.
                # However, the 'best_x' from Phase 1 is still valid if BFGS didn't improve.
                pass
            except Exception:
                # Catch any other errors from BFGS
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
