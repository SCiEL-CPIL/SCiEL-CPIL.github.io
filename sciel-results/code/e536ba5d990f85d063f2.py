"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Nelder-Mead Simplex Optimizer with adaptive restarts and deadline guard."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280.0  # Safety margin before 300s limit
        
        def safe_call(x):
            try:
                return float(func(x))
            except:
                return float('inf')
        
        rng = np.random.default_rng(42)
        best_x = None
        best_f = float('inf')
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # Initial simplex size
        init_step = 1.0
        
        # Run multiple restarts
        for restart in range(3):
            if time.time() > deadline:
                break
            
            # Initialize simplex
            if restart == 0:
                x0 = rng.uniform(-4, 4, dim)
            else:
                # Perturb from best found so far
                if best_x is not None:
                    x0 = np.clip(best_x + rng.normal(0, 1.0, dim), -5, 5)
                else:
                    x0 = rng.uniform(-4, 4, dim)
            
            # Build simplex
            simplex = [x0.copy()]
            for i in range(dim):
                xi = x0.copy()
                xi[i] += init_step
                simplex.append(xi)
            
            # Evaluate simplex
            fvals = [safe_call(x) for x in simplex]
            
            # Track best
            for i, (x, f) in enumerate(zip(simplex, fvals)):
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
            
            # Nelder-Mead iterations
            no_improve = 0
            for it in range(budget // 2):
                if time.time() > deadline:
                    break
                
                # Sort simplex by function value
                order = np.argsort(fvals)
                simplex = [simplex[i] for i in order]
                fvals = [fvals[i] for i in order]
                
                # Check convergence
                if np.std(fvals) < 1e-8:
                    break
                
                # Centroid of all but worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                xr = centroid + alpha * (centroid - simplex[-1])
                xr = np.clip(xr, -5, 5)
                fr = safe_call(xr)
                
                if fr < fvals[0]:
                    # Expansion
                    xe = centroid + gamma * (centroid - simplex[-1])
                    xe = np.clip(xe, -5, 5)
                    fe = safe_call(xe)
                    if fe < fr:
                        simplex[-1] = xe
                        fvals[-1] = fe
                    else:
                        simplex[-1] = xr
                        fvals[-1] = fr
                    no_improve = 0
                elif fr < fvals[-2]:
                    # Accept reflection
                    simplex[-1] = xr
                    fvals[-1] = fr
                    no_improve = 0
                else:
                    # Contraction
                    if fr < fvals[-1]:
                        # Outside contraction
                        xc = centroid + rho * (xr - centroid)
                        fc = safe_call(np.clip(xc, -5, 5))
                        if fc <= fr:
                            simplex[-1] = np.clip(xc, -5, 5)
                            fvals[-1] = fc
                            no_improve = 0
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = safe_call(simplex[i])
                            no_improve += 1
                    else:
                        # Inside contraction
                        xc = centroid - rho * (centroid - simplex[-1])
                        fc = safe_call(np.clip(xc, -5, 5))
                        if fc < fvals[-1]:
                            simplex[-1] = np.clip(xc, -5, 5)
                            fvals[-1] = fc
                            no_improve = 0
                        else:
                            # Shrink
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                fvals[i] = safe_call(simplex[i])
                            no_improve += 1
                
                # Update best
                for i, (x, f) in enumerate(zip(simplex, fvals)):
                    if f < best_f:
                        best_f = f
                        best_x = x.copy()
                
                if no_improve > 10:
                    break
        
        return best_x, best_f
# EVOLVE-BLOCK-END
