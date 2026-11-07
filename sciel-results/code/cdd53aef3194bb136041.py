import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for 5D BBOB f6."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: evaluator kills at 300s. We stop at 290s.
        deadline = start_time + 290.0
        
        # Strategy: Multi-start Nelder-Mead
        # Budget: 500
        # Dim: 5
        # Initial simplex size: 6
        # We divide budget into 5 restarts of 100 evals each.
        n_restarts = 5
        evals_per_restart = self.budget // n_restarts
        
        best_x = None
        best_f = float('inf')
        
        # Use a fixed seed for deterministic behavior, 
        # but different seeds for each restart to diversify.
        base_rng = np.random.default_rng(12345)
        
        for r in range(n_restarts):
            if time.time() > deadline:
                break
                
            # Create a local RNG for this restart
            rng = np.random.default_rng(base_rng.integers(0, 2**32))
            
            # Initialize center
            # Minimizer is in [-4, 4]^5. 
            # We sample from [-4, 4] to be closer to the optimum.
            center = rng.uniform(-4, 4, self.dim)
            
            # Simplex initialization
            # Point 0: center
            # Points 1..dim: center + delta * e_i
            delta = 1.0
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = center
            for i in range(self.dim):
                simplex[i+1] = center
                simplex[i+1][i] += delta
            
            # Clip to box [-5, 5]
            simplex = np.clip(simplex, -5, 5)
            
            # Evaluate initial simplex
            simplex_f = np.zeros(self.dim + 1)
            for i in range(self.dim + 1):
                simplex_f[i] = func(simplex[i])
                
            # Update global best
            min_idx = np.argmin(simplex_f)
            if simplex_f[min_idx] < best_f:
                best_f = simplex_f[min_idx]
                best_x = simplex[min_idx].copy()
                
            # Nelder-Mead parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            # Remaining evaluations for this restart
            remaining = evals_per_restart - (self.dim + 1)
            
            while remaining > 0:
                if time.time() > deadline:
                    break
                    
                # Sort simplex by function value
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Update global best
                if simplex_f[0] < best_f:
                    best_f = simplex_f[0]
                    best_x = simplex[0].copy()
                    
                # Check for convergence (optional, but good for stopping early if stuck)
                # If spread is very small, we might be stuck.
                # But we have restarts, so we can just let it run or break.
                # For now, just run until budget or time.
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # 1. Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = func(xr)
                remaining -= 1
                
                if fr < simplex_f[0]:
                    # Try Expansion
                    if remaining <= 0:
                        break
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = func(xe)
                    remaining -= 1
                    
                    if fe < fr:
                        simplex[-1] = xe
                        simplex_f[-1] = fe
                    else:
                        simplex[-1] = xr
                        simplex_f[-1] = fr
                elif fr < simplex_f[-2]:
                    # Accept Reflection
                    simplex[-1] = xr
                    simplex_f[-1] = fr
                else:
                    # Try Contraction
                    if remaining <= 0:
                        break
                    # Determine contraction type
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - xr)
                    xc = np.clip(xc, -5, 5)
                    fc = func(xc)
                    remaining -= 1
                    
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        # Move all points except best towards best
                        # This consumes dim evaluations
                        for i in range(1, self.dim + 1):
                            if remaining <= 0:
                                break
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            simplex_f[i] = func(simplex[i])
                            remaining -= 1
                            
                        if remaining <= 0:
                            break

        return best_x, best_f
