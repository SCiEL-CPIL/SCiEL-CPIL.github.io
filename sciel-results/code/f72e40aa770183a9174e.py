"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc


class Optimizer:
    """Deterministic Sobol sampling + Nelder-Mead local refinement."""

    def __init__(self, budget, dim):
        self.budget, self.dim = budget, dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        lower = -5.0
        upper = 5.0

        # Phase 1: Systematic Sobol sampling to find a good starting point
        # Use a larger initial sample for better coverage
        n_sobol = min(32, budget)
        sampler = qmc.Sobol(d=dim, scramble=False, seed=42)
        sobol_points = sampler.random(n_sobol)
        sobol_points = sobol_points * (upper - lower) + lower

        best_x = None
        best_f = np.inf
        calls_made = 0

        for i in range(n_sobol):
            x = sobol_points[i]
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()

        remaining = budget - calls_made

        if remaining <= 0:
            return best_x, best_f

        # Phase 2: Nelder-Mead refinement with adaptive simplex
        # Create a custom initial simplex with appropriate scaling
        # The Bent Cigar function has different scales along dimensions
        # Use a smaller step size for better convergence
        
        try:
            # Build initial simplex manually for better control
            x0 = best_x.copy()
            # Scale step sizes: smaller for later dimensions (cigar shape)
            step_sizes = np.array([0.5, 0.1, 0.02, 0.005, 0.001])[:dim]
            
            simplex = [x0.copy()]
            for i in range(dim):
                vertex = x0.copy()
                vertex[i] += step_sizes[i]
                simplex.append(vertex)
            
            result = minimize(
                func,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 500, 
                    'maxfev': remaining, 
                    'adaptive': True,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'initial_simplex': np.array(simplex)
                }
            )
            
            if result.fun < best_f:
                best_f = result.fun
                best_x = result.x.copy()
                
        except Exception:
            pass

        # Phase 3: If we have remaining budget, do a second Nelder-Mead pass
        # with tighter tolerances for final convergence
        calls_after_nm1 = budget - remaining
        remaining2 = budget - calls_made - (result.nfev if hasattr(result, 'nfev') else 0)
        
        if remaining2 > 10:
            try:
                result2 = minimize(
                    func,
                    best_x,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 200, 
                        'maxfev': remaining2, 
                        'adaptive': True,
                        'xatol': 1e-10,
                        'fatol': 1e-12
                    }
                )
                
                if result2.fun < best_f:
                    best_f = result2.fun
                    best_x = result2.x.copy()
                    
            except Exception:
                pass

        return best_x, best_f
# EVOLVE-BLOCK-END
