import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280
        
        calls = 0
        def f(x):
            nonlocal calls
            if time.time() > deadline:
                return 1e100
            calls += 1
            return func(x)
            
        # 1. Deterministic Grid Initialization
        # Use a grid of {-2, 0, 2}^dim to cover the likely basin
        coords = [-2.0, 0.0, 2.0]
        # Generate grid points
        # For dim=5, 3^5 = 243 points
        candidates = np.array(np.meshgrid(*[coords]*self.dim)).T.reshape(-1, self.dim)
        
        best_x = None
        best_f = np.inf
        
        # Evaluate grid points
        # We can batch if func supports it, but it takes a single vector.
        for pt in candidates:
            if calls >= self.budget:
                break
            val = f(pt)
            if val < best_f:
                best_f = val
                best_x = pt
        
        if calls >= self.budget or best_x is None:
            return best_x, best_f

        # 2. Nelder-Mead Optimization
        # Initialize simplex
        # One vertex is best_x
        # Others are best_x + h * e_i
        h = 0.5
        simplex = [best_x.copy()]
        simplex_f = [best_f]
        
        for i in range(self.dim):
            pt = best_x.copy()
            pt[i] += h
            simplex.append(pt)
            simplex_f.append(f(pt))
            
        if calls >= self.budget:
            return best_x, best_f

        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        chi = 2.0
        
        # Main loop
        while calls < self.budget:
            if time.time() > deadline:
                break
            
            # Sort simplex
            indices = np.argsort(simplex_f)
            simplex = [simplex[i] for i in indices]
            simplex_f = [simplex_f[i] for i in indices]
            
            x0 = simplex[0]
            f0 = simplex_f[0]
            
            # Centroid
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            xr = centroid + alpha * (centroid - simplex[-1])
            xr = np.clip(xr, -5, 5)
            fr = f(xr)
            
            if fr < f0:
                # Expansion
                xe = centroid + gamma * (centroid - simplex[-1])
                xe = np.clip(xe, -5, 5)
                fe = f(xe)
                if fe < fr:
                    simplex[-1] = xe
                    simplex_f[-1] = fe
                else:
                    simplex[-1] = xr
                    simplex_f[-1] = fr
            elif fr < simplex_f[-2]:
                # Accept reflection
                simplex[-1] = xr
                simplex_f[-1] = fr
            else:
                # Contraction
                if fr < simplex_f[-1]:
                    xc = centroid + rho * (simplex[-1] - centroid)
                else:
                    xc = centroid - rho * (centroid - simplex[-1])
                xc = np.clip(xc, -5, 5)
                fc = f(xc)
                if fc < simplex_f[-1]:
                    simplex[-1] = xc
                    simplex_f[-1] = fc
                else:
                    # Shrink
                    for i in range(1, len(simplex)):
                        simplex[i] = x0 + 0.5 * (simplex[i] - x0)
                        simplex_f[i] = f(simplex[i])
                        
            # Convergence check
            if max(simplex_f) - min(simplex_f) < 1e-10:
                break
                
        best_idx = np.argmin(simplex_f)
        return simplex[best_idx], simplex_f[best_idx]
