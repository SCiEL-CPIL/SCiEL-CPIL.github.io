import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

class Optimizer:
    """Active Set / Anisotropic 1D Search Optimizer for BBOB f12 (Bent Cigar)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        self.calls = 0
        self.start_time = time.time()
        # 300s limit, use 290s to be safe
        self.time_limit = 290.0
        
        d = self.dim
        N = self.budget
        
        # Global best tracking
        best_x = np.zeros(d)
        best_fx = float('inf')
        
        def eval_func(x):
            nonlocal best_x, best_fx
            if self.calls >= N:
                raise StopIteration("Budget exceeded")
            if time.time() - self.start_time > self.time_limit:
                raise StopIteration("Time limit exceeded")
            
            self.calls += 1
            fx = func(x)
            
            if np.isfinite(fx) and fx < best_fx:
                best_fx = fx
                best_x = x.copy()
                
            return fx

        # Helper for 1D minimization along a direction
        def line_search_1d(center, direction, max_evals=50, bracket_range=5.0):
            """
            Performs a 1D minimization along 'direction' starting from 'center'.
            Uses scipy's minimize_scalar with bounded method.
            """
            nonlocal best_x, best_fx
            
            def f1d(t):
                x_new = center + t * direction
                return eval_func(x_new)
            
            # We need to bracket the minimum.
            # We try a few different bracket ranges.
            # Note: f1d(0) is not called here to avoid wasting a call if center is known,
            # but minimize_scalar will call it.
            # To be safe and efficient, we rely on the bounded search.
            
            best_t = 0.0
            best_f = float('inf')
            
            # Try a small range first, then larger if needed? 
            # For Bent Cigar, the valley is narrow.
            # Let's try a sequence of expanding brackets or just a fixed reasonable one.
            # Since we are doing coordinate descent style, a local search is good.
            
            try:
                # Use a moderate bracket. If the step is too big, we might miss the valley.
                # If too small, we might be stuck.
                # Let's try 0.5, 1.0, 2.0
                # Actually, adaptive Powell handles this better, but we are doing manual 1D.
                # Let's just use a single bounded search with a reasonably large bound.
                # If the function is convex along the line, bounded search works well.
                
                res = minimize_scalar(f1d, bounds=(-bracket_range, bracket_range), method='bounded', options={'maxiter': 20, 'xatol': 1e-8})
                if res.fun < best_f:
                    best_f = res.fun
                    best_t = res.x
                    x_candidate = center + best_t * direction
                    if best_f < best_fx:
                        best_fx = best_f
                        best_x = x_candidate.copy()
            except Exception:
                pass

            return best_t, best_f

        # 1. Coarse Global Sampling to identify the best basin
        # We evaluate a small set of diverse points to find the most promising start
        rng = np.random.default_rng(42)
        
        # Define candidate points
        candidates = [np.zeros(d)]
        # Add random points
        for _ in range(5):
            candidates.append(rng.uniform(-4.5, 4.5, size=d))
        # Add points near boundaries
        candidates.append(np.array([4.0] * d))
        candidates.append(np.array([-4.0] * d))
        
        # Evaluate all candidates
        for c in candidates:
            if self.calls >= N or time.time() - self.start_time > self.time_limit:
                break
            try:
                eval_func(c)
            except StopIteration:
                break
            except Exception:
                pass

        # 2. Anisotropic Refinement Phase
        # Strategy: Use Powell's method which is good for narrow valleys,
        # but enhance it with explicit 1D searches if Powell stalls or to polish.
        # The inspiration suggests using Powell to align directions, then 1D searches.
        
        # First, run a robust Powell optimization.
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            try:
                # Powell is effective for Bent Cigar.
                # We give it a generous budget but monitor time.
                minimize(
                    eval_func,
                    best_x,
                    method='Powell',
                    options={
                        'maxiter': 100, # Limit iterations to allow time for other phases
                        'ftol': 1e-12,
                        'xatol': 1e-12,
                        'disp': False
                    }
                )
            except StopIteration:
                pass
            except Exception:
                pass

        # 3. Additional 1D Coordinate Descent Polish
        # Powell might leave some residual error in the narrow direction.
        # We perform a few rounds of 1D searches along the coordinate axes
        # and also along the directions that Powell might have implicitly found?
        # Since we don't have access to Powell's internal directions, we stick to coordinates
        # but with a tighter tolerance.
        
        if self.calls < N and time.time() - self.start_time < self.time_limit:
            # Run a few rounds of coordinate descent
            for _ in range(5):
                if self.calls >= N or time.time() - self.start_time > self.time_limit:
                    break
                
                for i in range(d):
                    if self.calls >= N or time.time() - self.start_time > self.time_limit:
                        break
                    
                    direction = np.zeros(d)
                    direction[i] = 1.0
                    
                    # Use a small bracket for polishing
                    _, _ = line_search_1d(best_x, direction, max_evals=20, bracket_range=0.5)

        return best_x, best_fx
