import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        start_time = time.time()
        deadline = start_time + 290.0  # 10s safety margin

        # Initial Simplex with larger spread to cover the domain
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1
        
        simplex = [x0.copy()]
        fvals = [f0]
        
        # Generate other simplex points with spread of 3.0
        spread = 3.0
        for i in range(dim):
            xi = x0.copy()
            xi[i] = spread
            fi = func(xi)
            calls += 1
            simplex.append(xi)
            fvals.append(fi)
            
        # Sort by function value
        idx = np.argsort(fvals)
        simplex = [simplex[i] for i in idx]
        fvals = [fvals[i] for i in idx]
        
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        best_x = simplex[0].copy()
        best_f = fvals[0]
        
        while calls < budget:
            # Time guard
            if time.time() > deadline:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = func(x_r)
            calls += 1
            
            if f_r < fvals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    fvals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
            elif f_r < fvals[-2]:
                # Simple Reflection
                simplex[-1] = x_r
                fvals[-1] = f_r
            else:
                # Contraction
                if f_r < fvals[-1]:
                    # Outside Contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside Contraction
                    x_c = centroid - rho * (simplex[-1] - centroid)
                    
                f_c = func(x_c)
                calls += 1
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        if time.time() > deadline:
                            break
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget or time.time() > deadline:
                        break
            
            # Sort
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0].copy()
                
            # Convergence check - use relative tolerance on function values
            if len(fvals) > 1:
                f_range = fvals[-1] - fvals[0]
                if f_range < 1e-12 * max(1.0, abs(best_f)):
                    break

        # Final polish: use remaining budget for a local search around best point
        if calls < budget and time.time() < deadline:
            # Adaptive local search with decreasing step size
            step = 0.5
            for _ in range(50):
                if calls >= budget or time.time() > deadline:
                    break
                # Try small perturbations in each dimension
                improved = False
                for d in range(dim):
                    if calls >= budget or time.time() > deadline:
                        break
                    for sign in [1, -1]:
                        if calls >= budget or time.time() > deadline:
                            break
                        x_try = best_x.copy()
                        x_try[d] += sign * step
                        x_try = np.clip(x_try, -5, 5)
                        f_try = func(x_try)
                        calls += 1
                        if f_try < best_f:
                            best_x = x_try
                            best_f = f_try
                            improved = True
                            break
                if not improved:
                    step *= 0.5
                    if step < 1e-8:
                        break

        return best_x, best_f
