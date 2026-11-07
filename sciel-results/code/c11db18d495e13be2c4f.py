import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        n = self.dim
        budget = self.budget
        
        # Nelder-Mead Parameters
        alpha = 1.0      # Reflection
        gamma = 2.0      # Expansion
        rho = 0.5        # Contraction
        sigma = 0.5      # Shrinkage
        
        # Initialization
        # Start at center
        x0 = np.zeros(n)
        f0 = func(x0)
        calls = 1
        
        # Create simplex
        # x0 is the first vertex.
        # Other vertices are offset by 'spread' in each dimension.
        # Use a larger spread to explore the box [-5, 5] more effectively
        spread = 4.0
        simplex = [x0.copy()]
        f_values = [f0]
        
        for i in range(n):
            xi = x0.copy()
            xi[i] += spread
            fi = func(xi)
            simplex.append(xi)
            f_values.append(fi)
            calls += 1
            
        # Sort simplex by function value
        # Indices: 0 is best, n is worst
        def sort_simplex():
            nonlocal simplex, f_values
            order = np.argsort(f_values)
            simplex = [simplex[i] for i in order]
            f_values = [f_values[i] for i in order]
            
        sort_simplex()
        best_x = simplex[0].copy()
        best_f = f_values[0]
        
        while calls < budget:
            if time.time() - start_time > 280:
                break
                
            # 1. Centroid of all points except the worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # 2. Reflection
            x_ref = centroid + alpha * (centroid - simplex[-1])
            x_ref = np.clip(x_ref, -5, 5)
            f_ref = func(x_ref)
            calls += 1
            
            if f_ref < f_values[0]:
                # Try Expansion
                x_exp = centroid + gamma * (x_ref - centroid)
                x_exp = np.clip(x_exp, -5, 5)
                f_exp = func(x_exp)
                calls += 1
                if f_exp < f_ref:
                    simplex[-1] = x_exp
                    f_values[-1] = f_exp
                else:
                    simplex[-1] = x_ref
                    f_values[-1] = f_ref
            elif f_ref < f_values[-2]:
                # Good point
                simplex[-1] = x_ref
                f_values[-1] = f_ref
            else:
                # Contraction
                if f_ref < f_values[-1]:
                    # Outside contraction
                    x_con = centroid + rho * (x_ref - centroid)
                else:
                    # Inside contraction
                    x_con = centroid - rho * (x_ref - centroid)
                    
                x_con = np.clip(x_con, -5, 5)
                f_con = func(x_con)
                calls += 1
                
                if f_con < f_ref:
                    simplex[-1] = x_con
                    f_values[-1] = f_con
                else:
                    # Shrinkage
                    for i in range(1, n + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = func(simplex[i])
                        calls += 1
                        if calls >= budget:
                            break
                
            sort_simplex()
            if simplex[0] is not best_x or f_values[0] < best_f:
                best_x = simplex[0].copy()
                best_f = f_values[0]
                
            # Convergence check
            # If spread is too small, break?
            # For Sphere, we want to keep going to refine.
            # Check if step size is negligible
            spread_current = np.max(np.abs(simplex[0] - simplex[1]))
            if spread_current < 1e-8:
                break

        return best_x, best_f
