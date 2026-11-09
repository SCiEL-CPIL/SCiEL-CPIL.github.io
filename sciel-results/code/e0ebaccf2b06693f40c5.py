"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Deterministic Nelder-Mead with multi-start."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = np.inf
        
        # Fixed seed for reproducibility within the instance
        rng = np.random.RandomState(42)

        def safe_call(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5, 5)
            calls_made += 1
            try:
                f_val = float(func(x))
            except Exception:
                f_val = np.inf
            
            if f_val < best_f:
                best_f = f_val
                best_x = x.copy()
            return f_val

        def nelder_mead(x0, f0):
            nonlocal calls_made, best_x, best_f
            if time.time() > deadline:
                return x0, f0
            
            # Initialize simplex
            alpha = 0.05
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            fvals = np.zeros(dim + 1)
            fvals[0] = f0
            
            for i in range(1, dim + 1):
                simplex[i] = x0.copy()
                simplex[i][i-1] += alpha
                fvals[i] = safe_call(simplex[i])
            
            # Nelder-Mead parameters
            alpha_n = 1.0   # Reflection
            gamma = 2.0     # Expansion
            rho = 0.5       # Contraction
            sigma = 0.5     # Shrink
            
            max_iter = 50
            for _ in range(max_iter):
                if calls_made >= budget - 10 or time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = simplex[order]
                fvals = fvals[order]
                
                # Check convergence
                if np.std(fvals) < 1e-10 * (1 + abs(fvals[0])):
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha_n * (centroid - simplex[-1])
                fr = safe_call(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (simplex[-1] - centroid)
                    
                    fc = safe_call(xc)
                    
                    if fc < min(fr, fvals[-1]):
                        simplex[-1] = xc
                        fvals[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fvals[i] = safe_call(simplex[i])
            
            return simplex[0], fvals[0]

        # Multi-start strategy
        # Start with a few diverse points to cover the search space
        # Then refine the best one
        
        # Initial diverse starts
        num_starts = 5
        starts = []
        
        # Center
        starts.append(np.zeros(dim))
        
        # Corners and other diverse points
        for i in range(num_starts - 1):
            x = rng.uniform(-4, 4, dim)
            starts.append(x)
        
        # Evaluate all starts
        start_fvals = []
        for x in starts:
            f = safe_call(x)
            start_fvals.append(f)
        
        # Find the best start
        best_start_idx = np.argmin(start_fvals)
        best_x = starts[best_start_idx].copy()
        best_f = start_fvals[best_start_idx]
        
        # Run Nelder-Mead from the best start
        best_x, best_f = nelder_mead(best_x, best_f)
        
        # If we still have budget, try a few more starts if the first one didn't converge well
        # Check if we're stuck (high function value)
        if best_f > 10 and calls_made < budget - 50:
            # Try 2 more diverse starts
            for _ in range(2):
                x = rng.uniform(-4, 4, dim)
                f = safe_call(x)
                
                if f < best_f:
                    best_x = x.copy()
                    best_f = f
                    # Refine this new best
                    best_x, best_f = nelder_mead(best_x, best_f)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
