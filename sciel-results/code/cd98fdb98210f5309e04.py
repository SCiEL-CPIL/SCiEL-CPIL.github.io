import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 280.0  # Safety margin for 330s limit
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            # Time check
            if time.time() - start_time > time_limit:
                return best_f
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coordinate descent with adaptive step sizes for Rosenbrock
        # Start with multiple random initializations and coordinate descent
        n_initializations = 5
        best_overall_x = None
        best_overall_f = float('inf')
        
        for init_idx in range(n_initializations):
            if calls_made >= self.budget or time.time() - start_time > time_limit:
                break
            
            # Random starting point
            x = np.random.uniform(-3, 3, self.dim)
            fx = safe_eval(x)
            
            if fx >= best_overall_f:
                continue
            
            # Adaptive coordinate descent
            # Rosenbrock has a curved valley; use larger steps initially
            step_sizes = np.ones(self.dim) * 1.0
            min_step = 1e-4
            max_step = 5.0
            stagnation_count = 0
            
            for cd_iter in range(100):  # Max iterations per initialization
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                
                improved = False
                
                for d in range(self.dim):
                    if calls_made >= self.budget or time.time() - start_time > time_limit:
                        break
                    
                    # Try positive and negative steps
                    for sign in [1.0, -1.0]:
                        x_trial = x.copy()
                        x_trial[d] += sign * step_sizes[d]
                        x_trial = np.clip(x_trial, -5, 5)
                        
                        fx_trial = safe_eval(x_trial)
                        
                        if fx_trial < fx:
                            # Accept the step
                            x = x_trial
                            fx = fx_trial
                            step_sizes[d] = min(step_sizes[d] * 1.5, max_step)
                            improved = True
                            break  # Move to next dimension
                    else:
                        # No improvement in this dimension, shrink step
                        step_sizes[d] = max(step_sizes[d] * 0.5, min_step)
                
                # Update overall best
                if fx < best_overall_f:
                    best_overall_f = fx
                    best_overall_x = x.copy()
                
                # Check for convergence
                if not improved:
                    stagnation_count += 1
                    if stagnation_count >= 3:
                        break
                else:
                    stagnation_count = 0
                
                # Check if all steps are very small
                if np.all(step_sizes < min_step * 10):
                    break
        
        # Phase 2: Nelder-Mead from the best coordinate descent result
        if best_overall_x is not None and calls_made < self.budget:
            x0 = best_overall_x.copy()
            f0 = best_overall_f
            
            # Initial simplex with smaller steps since we're already in a good region
            step_size = 0.3
            simplex = [x0.copy()]
            for i in range(self.dim):
                v = x0.copy()
                v[i] += step_size
                simplex.append(v)
            
            f_values = [f0]
            for i in range(1, self.dim + 1):
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                fx = safe_eval(simplex[i])
                f_values.append(fx)
            
            while len(f_values) < self.dim + 1:
                f_values.append(float('inf'))
            
            # Nelder-Mead Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            while calls_made < self.budget and time.time() - start_time < time_limit:
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                f_min = f_values[0]
                f_max = f_values[-1]
                if abs(f_max - f_min) < 1e-8 * max(1.0, abs(f_min)):
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)
                
                if f_r < f_values[0]:
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_eval(x_e)
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_values[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_values[-1] = f_r
                elif f_r < f_values[-2]:
                    simplex[-1] = x_r
                    f_values[-1] = f_r
                else:
                    if f_r < f_values[-1]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (x_r - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = safe_eval(x_c)
                    
                    if f_c < f_r:
                        simplex[-1] = x_c
                        f_values[-1] = f_c
                    else:
                        for i in range(1, self.dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = safe_eval(simplex[i])
            
            # Update best if Nelder-Mead found something better
            if f_values[0] < best_f:
                best_f = f_values[0]
                best_x = simplex[0].copy()
        
        return best_x, best_f
