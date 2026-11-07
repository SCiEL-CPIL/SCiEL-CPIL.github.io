import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        best_x = None
        best_f = np.inf
        
        # Number of restarts
        n_restarts = 8
        budget_per_restart = self.budget // n_restarts
        
        for r in range(n_restarts):
            # Initialize center with diversity
            x0 = rng.uniform(-4.5, 4.5, self.dim)
            
            # Initialize simplex with larger initial step for better exploration
            # Use a step size that covers a reasonable portion of the domain
            delta = 2.0
            
            simplex = np.zeros((self.dim + 1, self.dim))
            simplex[0] = x0
            for i in range(self.dim):
                simplex[i+1] = x0.copy()
                simplex[i+1][i] += delta
            
            # Clip initial simplex to domain
            simplex = np.clip(simplex, -5, 5)
            
            # Evaluate initial simplex
            fvals = np.array([func(v) for v in simplex])
            
            # Track best so far
            idx_min = np.argmin(fvals)
            if fvals[idx_min] < best_f:
                best_f = fvals[idx_min]
                best_x = simplex[idx_min].copy()
            
            calls_used = self.dim + 1
            budget_left = budget_per_restart - calls_used
            
            # Nelder-Mead parameters tuned for rugged landscapes
            alpha = 1.0   # Reflection
            gamma = 2.0   # Expansion
            rho = 0.5     # Contraction
            sigma = 0.5   # Shrink
            
            # Sort simplex by function value
            def sort_simplex():
                nonlocal simplex, fvals
                idx = np.argsort(fvals)
                simplex = simplex[idx]
                fvals = fvals[idx]
            
            sort_simplex()
            
            # Track consecutive iterations without improvement
            no_improve_count = 0
            prev_best_in_local = best_f
            
            while budget_left > 0:
                # Current worst point is simplex[-1]
                # Current best is simplex[0]
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                budget_left -= 1
                
                if f_r < fvals[0]:
                    # Try expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if budget_left <= 0:
                        break
                    f_e = func(x_e)
                    budget_left -= 1
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        fvals[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        fvals[-1] = f_r
                elif f_r < fvals[-2]:
                    simplex[-1] = x_r
                    fvals[-1] = f_r
                else:
                    # Contraction
                    if f_r < fvals[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    if budget_left <= 0:
                        break
                    f_c = func(x_c)
                    budget_left -= 1
                    
                    if f_c < min(f_r, fvals[-1]):
                        simplex[-1] = x_c
                        fvals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if budget_left <= 0:
                                break
                            fvals[i] = func(simplex[i])
                            budget_left -= 1
                
                # Update best
                idx_min = np.argmin(fvals)
                if fvals[idx_min] < best_f:
                    best_f = fvals[idx_min]
                    best_x = simplex[idx_min].copy()
                    no_improve_count = 0
                else:
                    no_improve_count += 1
                
                # Sort simplex again for next iteration
                sort_simplex()
                
                # If no improvement for many iterations, break early to save budget for restarts
                if no_improve_count > 15:
                    break
        
        return best_x, best_f
