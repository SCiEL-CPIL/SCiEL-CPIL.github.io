import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    """
    Derivative-free Nelder-Mead optimizer for BBOB-style Rosenbrock (rotated).
    
    Strategy: 
    1. Avoids the 10x overhead of finite-difference gradients used by L-BFGS-B.
    2. Nelder-Mead adapts the simplex shape to navigate the narrow Rosenbrock valley.
    3. Multi-start with structured global sampling (center, boundaries, random) 
       to identify the basin of attraction early, which AOCC rewards heavily.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 325  # Safety margin before 330s kill
        
        dim = self.dim
        budget = self.budget
        
        # State for best solution
        best_x = None
        best_f = np.inf
        calls_made = 0
        
        # Helper to call func and track budget
        def tracked_func(x):
            nonlocal calls_made, best_f, best_x
            if calls_made >= budget:
                # Soft stop: return a value that signals "no improvement" or "converged"
                # to stop the optimizer without making an illegal 501st call.
                # Returning best_f is safe; it won't be lower than best_f.
                return best_f if best_f != np.inf else 1e100
            
            x_clipped = np.clip(x, -5, 5)
            f_val = func(x_clipped)
            calls_made += 1
            
            if f_val < best_f:
                best_f = f_val
                best_x = x_clipped.copy()
                
            return f_val

        # Strategy:
        # Phase 1: Coarse Global Sampling (Structured)
        # Use a small fraction of the budget to sample key points: center, corners, mid-edges.
        # This helps find the "basin" quickly.
        # For dim=5, a full grid is too expensive. We'll use a set of heuristic points.
        
        rng = np.random.default_rng(42)
        
        # Generate a set of initial points for multi-start
        # 1. Center
        # 2. +/- 1 along each axis (local neighborhood)
        # 3. +/- 4 along each axis (boundary exploration)
        # 4. A few random points
        
        initial_points = []
        # Center
        initial_points.append(np.zeros(dim))
        
        # Small perturbations from center
        for i in range(dim):
            for delta in [1.0, -1.0, 2.0, -2.0]:
                p = np.zeros(dim)
                p[i] = delta
                initial_points.append(p)
        
        # Boundary-ish points
        for i in range(dim):
            for delta in [4.0, -4.0]:
                p = np.zeros(dim)
                p[i] = delta
                initial_points.append(p)
        
        # Random points
        for _ in range(10):
            initial_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate all initial points first to find the best starting point for NM
        # We have 500 budget. 
        # Number of initial points: 1 + 10 + 10 + 10 = 31 points.
        # This is ~6% of budget. Acceptable for early progress.
        
        for p in initial_points:
            if time.time() > deadline - 5:
                break
            if calls_made >= budget:
                break
            tracked_func(p)
            
        # Phase 2: Nelder-Mead Local Refinement
        # Start from the best point found in Phase 1.
        # If no points were evaluated (shouldn't happen), fallback to center.
        if best_x is None:
            best_x = np.zeros(dim)
            best_f = tracked_func(best_x)
            
        # Run Nelder-Mead from the best point
        # We need to ensure we don't exceed the budget.
        # Nelder-Mead in scipy uses 'maxiter' and 'xatol', 'fatol'.
        # We can't directly limit function evaluations in scipy's minimize for Nelder-Mead,
        # but we can rely on the 'tracked_func' soft stop.
        # However, if the soft stop returns best_f, NM might think it converged or failed.
        # To be safer, we can run NM in a loop with decreasing tolerances or just let it run
        # and rely on the time/budget guard.
        
        # Since NM is derivative-free, it uses 1 call per vertex update.
        # It's much cheaper per iteration than L-BFGS-B with finite diffs.
        
        try:
            # Use the best point as x0
            res = minimize(
                tracked_func, 
                best_x, 
                method='Nelder-Mead',
                bounds=[(-5, 5)] * dim,
                options={
                    'maxiter': 1000, # High limit, budget will stop it
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True # Helps with ill-conditioned problems like Rosenbrock
                }
            )
        except Exception:
            pass

        # Phase 3: If budget remains, try a few more restarts from perturbed best_x
        # This helps escape local minima or refine if NM got stuck.
        
        if time.time() < deadline - 10 and calls_made < budget:
            # Try 3 more restarts from best_x with small perturbations
            for _ in range(3):
                if time.time() > deadline - 5:
                    break
                if calls_made >= budget:
                    break
                
                # Perturb best_x
                perturbation = rng.normal(0, 0.5, dim)
                x0 = np.clip(best_x + perturbation, -5, 5)
                
                try:
                    res = minimize(
                        tracked_func, 
                        x0, 
                        method='Nelder-Mead',
                        bounds=[(-5, 5)] * dim,
                        options={
                            'maxiter': 500,
                            'xatol': 1e-8,
                            'fatol': 1e-10,
                            'adaptive': True
                        }
                    )
                except Exception:
                    pass

        # Final check: if we have some budget left, do a few random restarts?
        # Probably not needed if NM converged.
        
        # Return best found
        if best_x is None:
            # Fallback
            best_x = np.zeros(dim)
            if calls_made < budget:
                best_f = tracked_func(best_x)
            else:
                best_f = 0.0 # Dummy, shouldn't happen

        return best_x, best_f
