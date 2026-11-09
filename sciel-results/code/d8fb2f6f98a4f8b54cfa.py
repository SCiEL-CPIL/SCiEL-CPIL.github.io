import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        time_limit = 290.0  # Safety margin for 300s limit (330s kill, allow 40s buffer)
        
        # State tracking
        best_x = None
        best_f = float('inf')
        calls_made = 0
        
        def safe_eval(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= self.budget:
                return best_f
            if time.time() - start_time > time_limit:
                return best_f
            
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        def adaptive_coordinate_descent(x0, max_iter=100):
            """
            Adaptive Coordinate Descent tailored for Rosenbrock.
            Uses large initial steps to quickly enter the valley, then shrinks.
            """
            x = x0.copy()
            fx = safe_eval(x)
            
            # Initial step size: large enough to traverse the domain quickly
            step_sizes = np.ones(self.dim) * 2.0
            min_step = 1e-6
            max_step = 5.0
            stagnation_count = 0
            
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                
                improved = False
                for d in range(self.dim):
                    if calls_made >= self.budget or time.time() - start_time > time_limit:
                        break
                    
                    # Try positive and negative steps
                    # Order: try larger step first? No, just try current step.
                    found_improvement = False
                    for sign in [1.0, -1.0]:
                        x_trial = x.copy()
                        x_trial[d] += sign * step_sizes[d]
                        x_trial = np.clip(x_trial, -5, 5)
                        
                        fx_trial = safe_eval(x_trial)
                        
                        if fx_trial < fx:
                            x = x_trial
                            fx = fx_trial
                            # Increase step size slightly if we made a good move,
                            # but cap it to avoid oscillation
                            step_sizes[d] = min(step_sizes[d] * 1.2, max_step)
                            improved = True
                            found_improvement = True
                            break # Move to next dimension
                            
                    if not found_improvement:
                        # Shrink step size for this dimension
                        step_sizes[d] = max(step_sizes[d] * 0.5, min_step)
                
                if not improved:
                    stagnation_count += 1
                    if stagnation_count >= 5:
                        break
                else:
                    stagnation_count = 0
                    
                if np.all(step_sizes < min_step * 10):
                    break
                
            return x, fx

        def nelder_mead(x0, max_iter=100):
            """
            Nelder-Mead optimization with adaptive parameters.
            """
            dim = len(x0)
            # Initial simplex
            step_size = 0.5
            simplex = [x0.copy()]
            for i in range(dim):
                v = x0.copy()
                v[i] += step_size
                simplex.append(v)
            
            f_values = [safe_eval(simplex[0])]
            for i in range(1, dim + 1):
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                f_values.append(safe_eval(simplex[i]))
            
            while len(f_values) < dim + 1:
                f_values.append(float('inf'))
            
            # Parameters
            alpha = 1.0
            gamma = 2.0
            rho = 0.5
            sigma = 0.5
            
            for _ in range(max_iter):
                if calls_made >= self.budget or time.time() - start_time > time_limit:
                    break
                
                indices = np.argsort(f_values)
                simplex = [simplex[i] for i in indices]
                f_values = [f_values[i] for i in indices]
                
                f_min = f_values[0]
                f_max = f_values[-1]
                if abs(f_max - f_min) < 1e-8 * max(1.0, abs(f_min)):
                    break
                
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[-1])
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_eval(x_r)
                
                if f_r < f_values[0]:
                    # Expansion
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
                    # Contraction
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
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            f_values[i] = safe_eval(simplex[i])
            
            return simplex[0].copy(), f_values[0]

        # Phase 1: Global Search using Adaptive Coordinate Descent
        # Use diverse initial points to cover the space
        # Latin Hypercube Sampling for better coverage than pure random
        n_initializations = 12
        
        # Generate LHS points
        # In 5D, 12 points is sparse, but CD will expand the search.
        # We use stratified random to ensure coverage in each dimension
        points = []
        for _ in range(n_initializations):
            # Stratified sampling: divide each dimension into bins, pick one per bin
            # Simple approach: random permutation of bins
            bins = np.linspace(-4, 4, n_initializations)
            x = np.array([
                np.random.uniform(bins[i], bins[i+1]) if i < n_initializations-1 
                else np.random.uniform(bins[-2], bins[-1]) 
                for i in range(self.dim)
            ])
            # Ensure we don't pick the same point twice
            points.append(x)
        
        # Shuffle to avoid bias in order
        np.random.shuffle(points)
        
        best_cd_x = None
        best_cd_f = float('inf')
        
        for i, x0 in enumerate(points):
            if calls_made >= self.budget or time.time() - start_time > time_limit:
                break
            
            # Allocate budget for CD based on remaining budget
            # Leave at least 100 calls for the final NM polish
            remaining_budget = self.budget - calls_made
            if remaining_budget < 50:
                break
            
            # Run CD
            x_opt, f_opt = adaptive_coordinate_descent(x0, max_iter=50)
            
            if f_opt < best_cd_f:
                best_cd_f = f_opt
                best_cd_x = x_opt.copy()
            
            # Update global best
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        # Phase 2: Local Refinement using Nelder-Mead
        # Start from the best CD result
        if best_cd_x is not None:
            # Reserve budget for NM
            # If we have > 100 calls left, use them for NM
            if self.budget - calls_made > 20:
                x_opt, f_opt = nelder_mead(best_cd_x, max_iter=200)
                if f_opt < best_f:
                    best_f = f_opt
                    best_x = x_opt.copy()
        else:
            # Fallback: if CD failed or didn't run, try NM from origin or random
            x0 = np.zeros(self.dim)
            x_opt, f_opt = nelder_mead(x0, max_iter=100)
            if f_opt < best_f:
                best_f = f_opt
                best_x = x_opt.copy()

        return best_x, best_f
