"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import math


class Optimizer:
    """Multi-restart Nelder-Mead local search."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        dim = self.dim
        budget = self.budget
        
        calls_made = 0
        best_x = None
        best_f = float('inf')
        
        # Parameters for Nelder-Mead
        step = 0.5
        alpha = 1.0  # reflection
        gamma = 2.0  # expansion
        rho = 0.5    # contraction
        sigma = 0.5  # shrinkage
        
        num_restarts = 10
        budget_per_restart = max(1, budget // num_restarts)
        
        for _ in range(num_restarts):
            if calls_made >= budget:
                break
                
            # Random start
            x0 = rng.uniform(-5, 5, dim)
            f0 = func(x0)
            calls_made += 1
            if f0 < best_f:
                best_f = f0
                best_x = x0.copy()
                
            if calls_made >= budget:
                break
                
            # Initialize Simplex
            # Simplex has dim+1 points
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += step
                simplex.append(xi)
            
            # Evaluate initial simplex
            simplex_f = [f0]
            for i in range(1, dim + 1):
                fi = func(simplex[i])
                calls_made += 1
                simplex_f.append(fi)
                if fi < best_f:
                    best_f = fi
                    best_x = simplex[i].copy()
                    
            # Nelder-Mead Loop
            while calls_made < budget:
                # Sort simplex by fitness
                indices = np.argsort(simplex_f)
                simplex = [simplex[i] for i in indices]
                simplex_f = [simplex_f[i] for i in indices]
                
                # Check convergence (optional, but good for stopping early if stuck)
                # If range is very small, break
                if simplex_f[-1] - simplex_f[0] < 1e-10:
                    break
                    
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                fr = func(xr)
                calls_made += 1
                if fr < best_f:
                    best_f = fr
                    best_x = xr.copy()
                    
                if calls_made >= budget:
                    break
                    
                if fr < simplex_f[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    fe = func(xe)
                    calls_made += 1
                    if fe < best_f:
                        best_f = fe
                        best_x = xe.copy()
                    if calls_made >= budget:
                        break
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
                    # Contraction
                    if fr < simplex_f[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                    else:
                        # Inside contraction
                        xc = centroid + rho * (simplex[-1] - centroid)
                    
                    fc = func(xc)
                    calls_made += 1
                    if fc < best_f:
                        best_f = fc
                        best_x = xc.copy()
                        
                    if calls_made >= budget:
                        break
                        
                    if fc < min(fr, simplex_f[-1]):
                        simplex[-1] = xc
                        simplex_f[-1] = fc
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            fi = func(simplex[i])
                            calls_made += 1
                            if fi < best_f:
                                best_f = fi
                                best_x = simplex[i].copy()
                            simplex_f[i] = fi
                            if calls_made >= budget:
                                break
                        
                        if calls_made >= budget:
                            break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
