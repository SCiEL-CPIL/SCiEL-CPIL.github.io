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

        # Phase 1: Broad random sampling to find promising basins
        # Rosenbrock has a long, curved valley; we need to get close to the (1,...,1) region (shifted/rotated)
        # We sample broadly to ensure we hit the general direction of the valley.
        n_initial_samples = 20
        initial_points = np.random.uniform(-4.5, 4.5, (n_initial_samples, self.dim))
        
        best_start_x = None
        best_start_f = float('inf')
        
        for i in range(n_initial_samples):
            if calls_made >= self.budget or time.time() - start_time > time_limit:
                break
            x = initial_points[i]
            fx = safe_eval(x)
            if fx < best_start_f:
                best_start_f = fx
                best_start_x = x.copy()

        if best_start_x is None:
            # Fallback if no calls were made (shouldn't happen)
            return best_x, best_f

        # Phase 2: Nelder-Mead from the best initial sample
        # This is a global-ish search that can navigate the curved valley better than coordinate descent
        # if started reasonably close.
        x0 = best_start_x
        f0 = best_start_f
        
        # Initial simplex: x0 and x0 + step * e_i
        # Use a moderate step size. Too small gets stuck in flat areas, too large might jump out of the valley.
        step_size = 0.5
        simplex = [x0.copy()]
        for i in range(self.dim):
            v = x0.copy()
            v[i] += step_size
            simplex.append(v)
        
        # Evaluate initial simplex points (except x0 which we already have)
        f_values = [f0]
        for i in range(1, self.dim + 1):
            if calls_made >= self.budget or time.time() - start_time > time_limit:
                break
            fx = safe_eval(simplex[i])
            f_values.append(fx)
        
        # Ensure we have all values
        while len(f_values) < self.dim + 1:
            f_values.append(float('inf'))

        # Nelder-Mead Parameters
        alpha = 1.0  # Reflection
        gamma = 2.0  # Expansion
        rho = 0.5    # Contraction
        sigma = 0.5  # Shrinkage
        
        # Main Nelder-Mead Loop
        # We run this until budget or time runs out.
        while calls_made < self.budget and time.time() - start_time < time_limit:
            # Sort simplex by function value
            indices = np.argsort(f_values)
            simplex = [simplex[i] for i in indices]
            f_values = [f_values[i] for i in indices]
            
            # Check convergence
            f_min = f_values[0]
            f_max = f_values[-1]
            if abs(f_max - f_min) < 1e-6 * max(1.0, abs(f_min)):
                # If converged, try a small perturbation to escape potential local minima or flat regions
                # However, for Rosenbrock, convergence usually means we are at the minimum.
                # Let's break if we are very close to the expected minimum value (which is 0 for standard Rosenbrock,
                # but here it is offset). We don't know the offset, so we rely on the gradient being small.
                # A better check: if the simplex is very small in size.
                simplex_size = np.max(np.linalg.norm(np.array(simplex) - simplex[0], axis=1))
                if simplex_size < 1e-4:
                    break
            
            # Centroid of all points except the worst
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
                # Accept reflection
                simplex[-1] = x_r
                f_values[-1] = f_r
            else:
                # Contraction
                if f_r < f_values[-1]:
                    # Outside contraction
                    x_c = centroid + rho * (x_r - centroid)
                else:
                    # Inside contraction
                    x_c = centroid - rho * (x_r - centroid)
                    
                x_c = np.clip(x_c, -5, 5)
                f_c = safe_eval(x_c)
                
                if f_c < f_r:
                    simplex[-1] = x_c
                    f_values[-1] = f_c
                else:
                    # Shrinkage
                    for i in range(1, self.dim + 1):
                        simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                        simplex[i] = np.clip(simplex[i], -5, 5)
                        f_values[i] = safe_eval(simplex[i])

        # Phase 3: Fine-tuning with a local search (e.g., small steps Nelder-Mead or Coordinate Descent)
        # If we have budget left, we can try to polish the result.
        if calls_made < self.budget and best_x is not None:
            x0 = best_x.copy()
            f0 = best_f
            
            # Smaller step size for fine-tuning
            step_size = 0.1
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

            # Run Nelder-Mead again with tighter parameters
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

        return best_x, best_f
