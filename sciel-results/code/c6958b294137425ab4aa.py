import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        
        # Initial Simplex
        # Start at center 0, spread 2.0
        x0 = np.zeros(dim)
        f0 = func(x0)
        calls = 1
        
        simplex = [x0.copy()]
        fvals = [f0]
        
        # Generate other simplex points
        for i in range(dim):
            xi = x0.copy()
            xi[i] = 2.0 # Spread
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
        
        best_x = simplex[0]
        best_f = fvals[0]
        
        while calls < budget:
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
                    x_c = centroid + 0.5 * (x_r - centroid)
                else:
                    # Inside Contraction
                    x_c = centroid - 0.5 * (simplex[-1] - centroid)
                    
                f_c = func(x_c)
                calls += 1
                
                if f_c < min(f_r, fvals[-1]):
                    simplex[-1] = x_c
                    fvals[-1] = f_c
                else:
                    # Shrinkage
                    for i in range(1, dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        fvals[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                    if calls >= budget:
                        break
            
            # Sort
            idx = np.argsort(fvals)
            simplex = [simplex[i] for i in idx]
            fvals = [fvals[i] for i in idx]
            
            # Update best
            if fvals[0] < best_f:
                best_f = fvals[0]
                best_x = simplex[0]
                
            # Convergence check (optional, budget is strict)
            if len(fvals) > 1 and abs(fvals[0] - fvals[-1]) < 1e-10:
                break

        return best_x, best_f
