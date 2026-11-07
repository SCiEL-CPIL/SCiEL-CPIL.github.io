import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        best_x = None
        best_f = float('inf')
        
        # Determine number of restarts
        # At least 1 restart. 
        # 3 restarts seems good for 500 budget.
        num_restarts = 3
        budget_per_restart = self.budget // num_restarts
        
        # If budget is very small, just 1 restart
        if budget_per_restart < 20:
            num_restarts = 1
            budget_per_restart = self.budget

        for i in range(num_restarts):
            # Random start
            rng = np.random.default_rng(123 + i)
            center = rng.uniform(-4, 4, self.dim) # Start slightly inside to avoid boundary issues
            
            # Run Nelder-Mead
            x, f = self._nelder_mead(func, center, budget_per_restart, self.dim)
            
            if f < best_f:
                best_f = f
                best_x = x

        return best_x, best_f

    def _nelder_mead(self, func, start_point, budget, dim):
        rng = np.random.default_rng(42)
        
        # Initial Simplex
        # d+1 points
        simplex = [start_point.copy()]
        for i in range(dim):
            p = start_point.copy()
            p[i] += 0.5
            simplex.append(p)
            
        # Evaluate initial simplex
        values = []
        calls = 0
        for i in range(dim + 1):
            f_val = func(simplex[i])
            values.append(f_val)
            calls += 1
            
        best_idx = np.argmin(values)
        best_x = simplex[best_idx].copy()
        best_f = values[best_idx]
        
        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrink
        
        # Loop
        while calls < budget:
            # Sort simplex
            order = np.argsort(values)
            simplex = [simplex[i] for i in order]
            values = [values[i] for i in order]
            
            # Update best
            if values[0] < best_f:
                best_f = values[0]
                best_x = simplex[0].copy()
            
            # Centroid of best d points
            centroid = np.mean(simplex[:dim], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            f_r = func(x_r)
            calls += 1
            
            if f_r < values[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                f_e = func(x_e)
                calls += 1
                if f_e < f_r:
                    simplex[-1] = x_e
                    values[-1] = f_e
                else:
                    simplex[-1] = x_r
                    values[-1] = f_r
            elif f_r < values[-2]:
                # Accept Reflection
                simplex[-1] = x_r
                values[-1] = f_r
            else:
                # Contraction
                if f_r < values[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                    f_c = func(x_c)
                    calls += 1
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            values[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                else:
                    # Inside contraction
                    x_c = centroid - rho * (centroid - simplex[-1])
                    f_c = func(x_c)
                    calls += 1
                    if f_c < values[-1]:
                        simplex[-1] = x_c
                        values[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            values[i] = func(simplex[i])
                            calls += 1
                            if calls >= budget:
                                break
                
                if calls >= budget:
                    break

        return best_x, best_f
