"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Hybrid Optimizer for Katsuura (f23).
    Strategy:
    1. Diverse initialization (vertices + random) to cover the box.
    2. Nelder-Mead local optimization on the best initial points.
    3. Fine-grained (1+1)-ES polishing for precision.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        rng = np.random.default_rng(42)

        call_count = 0

        def counted_func(x):
            nonlocal call_count
            if call_count >= budget:
                return np.inf
            call_count += 1
            # Clip to box [-5, 5] as per task description
            x_clipped = np.clip(x, -5.0, 5.0)
            return func(x_clipped)

        def remaining_time():
            return deadline - time.time()

        # Phase 1: Diverse Initialization
        # The Katsuura function is multimodal. We need to sample the space well.
        starts = []
        
        # 1. Center
        starts.append(np.zeros(dim))
        
        # 2. Vertices of the box mapped to [-4, 4] to stay inside safe zone
        # This ensures we check corners which might contain shifted minima
        for i in range(2 ** dim):
            v = np.array([(i >> j) & 1 for j in range(dim)], dtype=float)
            v = v * 2.0 - 1.0 # Map 0,1 to -1,1
            v = v * 4.0       # Scale to [-4, 4]
            starts.append(v)
            
        # 3. Random points in the interior
        for _ in range(10):
            starts.append(rng.uniform(-4.5, 4.5, dim))

        best_x = None
        best_f = np.inf
        evaluated_starts = []

        # Evaluate initial points
        for x0 in starts:
            if call_count >= budget or remaining_time() < 5.0:
                break
            f0 = counted_func(x0)
            evaluated_starts.append((x0.copy(), f0))
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()

        if best_x is None:
            # Fallback if no calls were made
            best_x = np.zeros(dim)
            best_f = counted_func(best_x)
            evaluated_starts.append((best_x.copy(), best_f))

        # Sort starts by fitness
        evaluated_starts.sort(key=lambda t: t[1])

        # Phase 2: Local Optimization using Nelder-Mead
        # Nelder-Mead is a population-based direct search method.
        # It is effective for small dimensions (dim=5) and handles non-smooth
        # or rugged landscapes better than gradient-based methods.
        
        # We try the top 3-5 distinct starting points.
        # Distinctness check to avoid redundant work on nearly identical points.
        tried_keys = set()
        
        for x0, f0 in evaluated_starts:
            if call_count >= budget or remaining_time() < 10.0:
                break
            
            # Skip if too close to a previously tried start
            key = tuple(np.round(x0, 2))
            if key in tried_keys:
                continue
            tried_keys.add(key)
            
            # Run Nelder-Mead
            # maxiter is a safety net, but we rely on maxfev and time.
            # We allow up to ~50% of remaining budget for this run, or a fixed cap.
            # For dim=5, NM usually converges in < 100 evals.
            max_fev_allowed = min(150, budget - call_count - 20) 
            if max_fev_allowed < 10:
                break
                
            try:
                result = minimize(
                    counted_func,
                    x0,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200,
                        'maxfev': max_fev_allowed,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True  # Adaptive simplex resizing helps with scaling
                    }
                )
                
                if result.fun < best_f:
                    best_f = result.fun
                    best_x = result.x.copy()
                    
            except Exception:
                pass

        # Phase 3: Polishing with (1+1)-ES
        # After NM finds a local minimum, the landscape is likely smoother.
        # (1+1)-ES is very efficient for final convergence.
        if call_count < budget and remaining_time() > 5.0 and best_x is not None:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1  # Start with a small step since we are near a minimum
            min_sigma = 1e-10
            max_sigma = 1.0
            
            # Run ES until budget or time runs out
            while call_count < budget and remaining_time() > 2.0:
                y = x + sigma * rng.standard_normal(dim)
                y = np.clip(y, -5.0, 5.0)
                fy = counted_func(y)
                
                if fy < fx:
                    x, fx = y, fy
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
                    # Increase step size on success
                    sigma = min(sigma * 1.1, max_sigma)
                else:
                    # Decrease step size on failure
                    sigma = max(sigma * 0.9, min_sigma)
                    if sigma <= min_sigma * 1.01:
                        break

        return best_x, best_f
# EVOLVE-BLOCK-END
