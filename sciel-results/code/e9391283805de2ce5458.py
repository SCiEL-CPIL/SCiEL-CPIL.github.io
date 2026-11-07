import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, keep 45s buffer
        deadline = start_time + 285
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = np.zeros(dim)
        best_f = float('inf')
        
        def can_call():
            nonlocal calls_made
            if calls_made >= budget:
                return False
            if time.time() > deadline:
                return False
            return True

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if not can_call():
                return float('inf')
            x = np.asarray(x, dtype=float)
            x = np.clip(x, -5, 5)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # 1. Radial Sweep Initialization
        # The Discus function f11 has an anisotropic scaling: 
        # f(x) = sum(10^i * (x_{i+1} - x_i)^2) + (x_1 + 10)^2
        # This creates a narrow valley along the direction (1, 1, ..., 1) roughly.
        # By sweeping radially from the origin, we can detect the scale of the valley.
        
        # We allocate ~20% of the budget to initialization.
        init_budget = int(budget * 0.2)
        
        # Start at 0,0,0,0,0
        x_center = np.zeros(dim)
        f_center = evaluate(x_center)
        
        # Radial sweep: Check points at increasing distances along the diagonal (1,1,1,1,1)/sqrt(5)
        # and along axes to catch if the shift is off-diagonal.
        # Since the shift is hidden but in [-4,4], and rotation is hidden, 
        # a pure radial sweep from 0 might miss the center if shifted far.
        # However, f11's structure makes the "valley" direction important.
        
        # Strategy: 
        # 1. Evaluate a few points to estimate the "scale" of the function's features.
        # 2. Use Nelder-Mead with adaptive simplex size based on this scale.
        
        # Let's do a coarse radial search to find a better starting point and scale.
        # Directions: Main diagonal and a few random directions.
        directions = [
            np.ones(dim) / np.sqrt(dim),
            np.zeros(dim),
        ]
        # Add some random directions for robustness against rotation
        rng = np.random.default_rng(42)
        for _ in range(3):
            d = rng.normal(size=dim)
            d /= np.linalg.norm(d)
            directions.append(d)
            
        # Distances: 1, 2, 3, 4
        distances = [1.0, 2.0, 3.0, 4.0]
        
        candidate_starts = []
        
        # Always include the center
        candidate_starts.append((x_center, f_center))
        
        for d in directions:
            for dist in distances:
                if not can_call():
                    break
                x_try = x_center + dist * d
                f_try = evaluate(x_try)
                candidate_starts.append((x_try, f_try))
                
                # Also check opposite direction? Maybe not necessary for radial scale estimation,
                # but good for finding the center if shifted.
                x_try_neg = x_center - dist * d
                f_try_neg = evaluate(x_try_neg)
                candidate_starts.append((x_try_neg, f_try_neg))
                
                if not can_call():
                    break

        # Sort candidates by function value
        candidate_starts.sort(key=lambda x: x[1])
        
        # The best candidate is our starting point for local optimization.
        # The "scale" can be estimated by the distance of the best candidate from 0,
        # or by the differences in function values.
        # For Nelder-Mead, the initial simplex size is crucial.
        # If the best point is far from 0, the scale is large. If close, small.
        
        best_init_x, best_init_f = candidate_starts[0]
        
        # Estimate initial step size (sigma) for Nelder-Mead
        # If the best point is at distance D from 0, the features might be on scale D.
        # Otherwise, default to 1.0.
        dist_best = np.linalg.norm(best_init_x)
        if dist_best < 0.5:
            sigma = 0.5
        else:
            sigma = max(0.5, dist_best * 0.5)
            
        # 2. Local Optimization with Nelder-Mead
        # We use scipy.optimize.minimize with method='Nelder-Mead'.
        # We must wrap func to count calls and respect the deadline.
        
        # We will run multiple Nelder-Mead instances if budget remains.
        # Since Nelder-Mead is deterministic and fast, we can run it from the top 2-3 candidates.
        
        top_candidates = candidate_starts[:3]
        
        for x_start, f_start in top_candidates:
            if not can_call():
                break
            if time.time() > deadline - 5: # Leave 5s for cleanup
                break
                
            # Define the objective function for scipy
            def objective(x):
                if not can_call():
                    return float('inf')
                return evaluate(x)
            
            # Initial simplex
            # Nelder-Mead in scipy creates the simplex based on x0 and initial_simplex or step size.
            # We provide initial_simplex for control.
            
            # Create simplex
            simplex = [x_start.copy()]
            for i in range(dim):
                x_i = x_start.copy()
                # Step size in direction i
                if abs(x_start[i]) > sigma:
                    step = 0.05 * x_start[i]
                else:
                    step = sigma
                x_i[i] += step
                simplex.append(x_i)
            
            try:
                res = minimize(
                    objective, 
                    x_start, 
                    method='Nelder-Mead', 
                    options={
                        'initial_simplex': simplex,
                        'maxiter': 200, # Limit iterations to ensure we don't waste time
                        'xatol': 1e-8,
                        'fatol': 1e-8,
                        'adaptive': True
                    }
                )
                
                # Update global best
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = res.x.copy()
                    
            except Exception:
                pass

        # 3. Final Polish (Optional)
        # If we have budget left, we can do a final L-BFGS-B run from the best point found.
        # L-BFGS-B uses gradient approximations and is very fast for smooth functions.
        if can_call() and time.time() < deadline - 10:
            def objective_lbfgsb(x):
                if not can_call():
                    return float('inf')
                return evaluate(x)
            
            try:
                res_lbfgsb = minimize(
                    objective_lbfgsb,
                    best_x,
                    method='L-BFGS-B',
                    bounds=[(-5, 5)] * dim,
                    options={
                        'maxiter': 50,
                        'gtol': 1e-10,
                        'ftol': 1e-10
                    }
                )
                if res_lbfgsb.fun < best_f:
                    best_f = res_lbfgsb.fun
                    best_x = res_lbfgsb.x.copy()
            except Exception:
                pass

        return best_x
