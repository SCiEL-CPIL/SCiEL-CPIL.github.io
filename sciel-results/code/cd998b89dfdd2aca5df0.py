import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 300.0
        dim = self.dim
        calls = 0
        best_x = None
        best_f = float('inf')
        rng = np.random.default_rng(42)

        def wrapped_func(x):
            nonlocal calls, best_x, best_f
            if calls >= self.budget:
                return best_f
            if time.time() - start_time > time_limit - 5.0:
                return best_f
            x = np.clip(x, -5, 5)
            v = func(x)
            calls += 1
            if v < best_f:
                best_f = v
                best_x = x.copy()
            return v

        # Initialize population
        # Use LHS or random in [-4, 4]
        lambda_pop = 10
        mu_sel = 5
        
        # Initial points
        # 1 point at 0, rest random
        init_points = [np.zeros(dim)]
        for _ in range(lambda_pop - 1):
            init_points.append(rng.uniform(-4, 4, dim))
        
        # Evaluate initial population
        pop_values = []
        for p in init_points:
            pop_values.append(wrapped_func(p))
        
        # Sort and select best mu
        indices = np.argsort(pop_values)
        best_indices = indices[:mu_sel]
        
        sigma = 1.0
        no_improve_count = 0
        max_no_improve = 20
        
        while calls < self.budget - 5 and time.time() - start_time < time_limit - 5.0:
            # Generate new population
            new_pop = []
            new_values = []
            
            for i in range(lambda_pop):
                # Select a parent from the best mu
                parent_idx = best_indices[i % mu_sel]
                parent = init_points[parent_idx]
                
                # Mutate
                mutation = rng.normal(0, 1, dim)
                child = parent + sigma * mutation
                child = np.clip(child, -5, 5)
                
                new_pop.append(child)
                new_values.append(wrapped_func(child))
                
                if calls >= self.budget - 5 or time.time() - start_time > time_limit - 5.0:
                    break
            
            if len(new_pop) < lambda_pop:
                break
                
            # Combine old and new populations
            all_points = init_points + new_pop
            all_values = pop_values + new_values
            
            # Sort and select best mu
            combined_indices = np.argsort(all_values)
            best_indices = combined_indices[:mu_sel]
            
            # Update population
            init_points = [all_points[i] for i in best_indices]
            pop_values = [all_values[i] for i in best_indices]
            
            # Check for improvement
            if pop_values[0] < best_f - 1e-10:
                no_improve_count = 0
            else:
                no_improve_count += 1
            
            # Adapt sigma
            if no_improve_count > max_no_improve:
                sigma *= 2.0 # Increase step size to escape local minimum
                no_improve_count = 0
            else:
                sigma *= 0.95 # Slowly decrease step size
        
        # Final polish with Nelder-Mead if budget allows
        if best_x is not None and calls < self.budget - 10 and time.time() - start_time < time_limit - 10.0:
            try:
                from scipy.optimize import minimize
                res = minimize(wrapped_func, best_x, method='Nelder-Mead', 
                               options={'maxiter': 50, 'xatol': 1e-8, 'fatol': 1e-12})
                if res.fun < best_f:
                    best_f = res.fun
                    best_x = np.clip(res.x, -5, 5)
            except Exception:
                pass

        return best_x, best_f
