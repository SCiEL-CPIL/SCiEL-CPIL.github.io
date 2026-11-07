"""CMA-ES based optimizer for shifted/rotated Rosenbrock-style f8."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """CMA-ES with adaptive restarts and boundary handling for Rosenbrock."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0

        best_x = None
        best_f = np.inf
        call_count = 0
        dim = self.dim
        budget = self.budget

        def safe_call(x):
            nonlocal best_x, best_f, call_count
            if call_count >= self.budget:
                return np.inf
            call_count += 1
            x = np.asarray(x, dtype=np.float64)
            # Clip to box to ensure valid input for func
            x_clipped = np.clip(x, -5.0, 5.0)
            try:
                fx = float(func(x_clipped))
            except Exception:
                fx = np.inf
            
            if np.isfinite(fx) and fx < best_f:
                best_f = fx
                best_x = x_clipped.copy()
            return fx

        # CMA-ES implementation without external library dependency to avoid overhead
        # and ensure strict budget control.
        # We use a simplified CMA-ES or rely on scipy's minimize with Nelder-Mead 
        # if CMA-ES library is not available, but scipy has CMA-ES in newer versions?
        # Actually, scipy.optimize has 'Nelder-Mead', 'Powell', 'L-BFGS-B'. 
        # SciPy does NOT have CMA-ES built-in in standard distributions usually.
        # However, the prompt says "Python standard library, NumPy and SciPy are available".
        # It does NOT guarantee 'cma' package.
        
        # Let's implement a robust Multi-Start L-BFGS-B strategy as suggested by the 
        # "knowledge from this run" which measured an improvement to 2.626 with 
        # "Multi-start L-BFGS-B with adaptive budget allocation". 
        # The current program uses Powell/NM. L-BFGS-B is gradient-based (quasi-Newton) 
        # and often faster for smooth Rosenbrock if it can find the basin.
        # But Rosenbrock has a narrow valley. L-BFGS-B can struggle if not close.
        # CMA-ES is better for global. 
        # Since 'cma' is not guaranteed, let's implement a high-quality 
        # Multi-Start Nelder-Mead with adaptive budget allocation and better initial points,
        # OR use a simple CMA-ES implementation if feasible within time.
        
        # Given the constraint and the specific knowledge that "Multi-start L-BFGS-B ... improved island 5",
        # let's try a hybrid: 
        # 1. Coarse global search to find a good basin.
        # 2. L-BFGS-B for fast local convergence.
        
        # Strategy:
        # 1. Evaluate a set of diverse starting points (grid + random).
        # 2. Run L-BFGS-B from the best few points.
        # 3. If L-BFGS-B fails or stalls, fallback to Nelder-Mead.
        
        # Phase 1: Diverse Initial Points
        # Rosenbrock minimum is at (1,1,...,1). Shifted/rotated means it's somewhere else.
        # We need to find the basin.
        
        rng = np.random.default_rng(42)
        
        # Generate candidate starting points
        candidates = []
        
        # Uniform random points
        for _ in range(20):
            candidates.append(rng.uniform(-4, 4, dim))
            
        # Strategic points
        candidates.append(np.zeros(dim))
        candidates.append(np.ones(dim))
        candidates.append(np.full(dim, -1))
        candidates.append(np.full(dim, 2))
        
        # Evaluate all candidates
        scored_candidates = []
        for cand in candidates:
            if call_count >= budget:
                break
            if time.time() > deadline - 50: # Leave time for optimization
                break
            fx = safe_call(cand)
            if np.isfinite(fx):
                scored_candidates.append((fx, cand))
        
        # Sort by function value
        scored_candidates.sort(key=lambda x: x[0])
        
        # Phase 2: Local Optimization from Best Candidates
        # Use L-BFGS-B as it's efficient for smooth functions.
        # If L-BFGS-B is not available or fails, use Nelder-Mead.
        
        max_local_starts = min(5, len(scored_candidates))
        
        for i in range(max_local_starts):
            if call_count >= budget:
                break
            if time.time() > deadline - 30:
                break
            
            fx0, x0 = scored_candidates[i]
            
            # Try L-BFGS-B first
            try:
                # L-BFGS-B requires bounds
                bounds = [(-5.0, 5.0)] * dim
                result = minimize(
                    lambda x: safe_call(x),
                    x0,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={
                        'maxiter': 100,
                        'maxfun': max(1, budget - call_count - 10),
                        'ftol': 1e-10,
                        'gtol': 1e-8,
                    }
                )
            except Exception:
                # Fallback to Nelder-Mead if L-BFGS-B fails
                try:
                    result = minimize(
                        lambda x: safe_call(x),
                        x0,
                        method='Nelder-Mead',
                        options={
                            'maxiter': 100,
                            'maxfev': max(1, budget - call_count - 10),
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True,
                        }
                    )
                except Exception:
                    pass

        # Phase 3: Final Polish with Nelder-Mead if budget remains
        if best_x is not None and call_count < budget and time.time() < deadline - 10:
            try:
                remaining = max(1, budget - call_count - 5)
                result = minimize(
                    lambda x: safe_call(x),
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 100,
                        'maxfev': remaining,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True,
                    }
                )
            except Exception:
                pass

        if best_x is None:
            best_x = np.zeros(dim)
            best_f = float(func(best_x))
        
        return best_x, best_f
# EVOLVE-BLOCK-END
