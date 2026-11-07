"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """Multi-start Powell with adaptive restarts for fast AOCC."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        start_time = time.time()
        deadline = start_time + 285  # Safety margin for 300s limit
        
        call_count = 0
        max_calls = self.budget
        
        best_x = None
        best_f = float('inf')
        
        def counted_func(x):
            nonlocal call_count
            if call_count >= max_calls:
                raise ValueError("Budget exceeded")
            call_count += 1
            x_clipped = np.clip(x, -5, 5)
            return func(x_clipped)

        def check_time():
            return time.time() < deadline and call_count < max_calls

        # Phase 1: Fast coarse exploration with short Powell runs
        # Generate diverse starting points: grid + random
        starts = []
        
        # Grid points at multiple scales
        grid_coords = [-4.0, -2.0, 0.0, 2.0, 4.0]
        for c1 in grid_coords:
            for c2 in grid_coords:
                if self.dim >= 2:
                    pt = np.zeros(self.dim)
                    pt[0] = c1
                    pt[1] = c2
                    starts.append(pt)
        
        # Add pure random points
        for _ in range(20):
            starts.append(rng.uniform(-5, 5, self.dim))
        
        # Shuffle to mix grid and random
        rng.shuffle(starts)
        
        # Phase 1: Short Powell runs (maxiter=20) for fast basin finding
        phase1_maxiter = 20
        phase1_limit = min(150, max_calls)  # Use at most 150 calls for phase 1
        
        for x0 in starts:
            if not check_time():
                break
            if call_count >= phase1_limit:
                break
                
            try:
                res = minimize(counted_func, x0, method='Powell',
                               options={'maxiter': phase1_maxiter, 'xtol': 1e-4, 'ftol': 1e-4})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 2: Longer refinement from best points found
        if best_x is not None and check_time():
            # Refine the best point with tighter tolerances
            try:
                res = minimize(counted_func, best_x, method='Powell',
                               options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Phase 3: If budget remains, try more random restarts with medium Powell
        if check_time() and call_count < max_calls:
            remaining = max_calls - call_count
            num_extra = min(remaining // 30, 10)  # ~30 calls per medium run
            
            for _ in range(num_extra):
                if not check_time():
                    break
                
                x0 = rng.uniform(-5, 5, self.dim)
                try:
                    res = minimize(counted_func, x0, method='Powell',
                                   options={'maxiter': 30, 'xtol': 1e-6, 'ftol': 1e-6})
                    if res.fun < best_f:
                        best_f = res.fun
                        best_x = res.x.copy()
                except Exception:
                    pass
        
        # Final polish if we have time and budget
        if best_x is not None and check_time() and call_count < max_calls - 10:
            try:
                res = minimize(counted_func, best_x, method='Powell',
                               options={'maxiter': 20, 'xtol': 1e-10, 'ftol': 1e-10})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
            except Exception:
                pass
        
        # Fallback if no successful optimization
        if best_x is None:
            best_x = np.zeros(self.dim)
            if call_count < max_calls:
                best_f = func(best_x)
                call_count += 1
            else:
                best_f = float('inf')
        
        return best_x, best_f
# EVOLVE-BLOCK-END
