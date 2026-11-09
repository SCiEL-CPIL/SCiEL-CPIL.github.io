import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.best_x = None
        self.best_f = float('inf')

    def __call__(self, func):
        self.start_time = time.time()
        deadline = self.start_time + 280  # Safety margin

        def safe_func(x):
            if self.calls >= self.budget:
                raise ValueError("Budget exceeded")
            if time.time() > deadline:
                raise TimeoutError("Time limit exceeded")
            self.calls += 1
            f_val = func(x)
            if f_val < self.best_f:
                self.best_f = f_val
                self.best_x = x.copy()
            return f_val

        rng = np.random.default_rng(42)
        
        # Parameters for Nelder-Mead
        # Standard NM parameters
        alpha = 1.0   # Reflection
        gamma = 2.0   # Expansion
        rho = 0.5     # Contraction
        sigma = 0.5   # Shrinkage
        
        # Initial simplex size
        initial_step = 0.5
        
        # Stagnation limit: If best value in simplex doesn't improve for this many iterations, restart
        stagnation_limit = 20 
        max_restarts = 15 # We will try to fit as many restarts as possible within budget
        
        for restart_i in range(max_restarts):
            if self.calls >= self.budget or time.time() > deadline:
                break
            
            # Choose a starting point
            # Mix of random and centered
            if restart_i == 0:
                start_x = np.zeros(self.dim)
            else:
                start_x = rng.uniform(-5, 5, self.dim)
                
            # Initialize simplex
            # Vertex 0 is start_x
            # Vertices 1..d are start_x + step * e_i
            simplex = np.empty((self.dim + 1, self.dim))
            simplex[0] = start_x
            for i in range(self.dim):
                simplex[i+1] = start_x
                simplex[i+1][i] += initial_step
                
            # Evaluate initial simplex
            simplex_f = np.empty(self.dim + 1)
            for i in range(self.dim + 1):
                if self.calls >= self.budget or time.time() > deadline:
                    break
                simplex_f[i] = safe_func(simplex[i])
                
            if self.calls >= self.budget or time.time() > deadline:
                break
                
            # Sort simplex by function value
            order = np.argsort(simplex_f)
            simplex = simplex[order]
            simplex_f = simplex_f[order]
            
            best_simplex_f = simplex_f[0]
            stagnant_count = 0
            
            # Nelder-Mead Loop
            while self.calls < self.budget and time.time() < deadline:
                # Check stagnation
                if simplex_f[0] < best_simplex_f - 1e-10:
                    best_simplex_f = simplex_f[0]
                    stagnant_count = 0
                else:
                    stagnant_count += 1
                    
                if stagnant_count > stagnation_limit:
                    break
                    
                # Identify vertices
                x_worst = simplex[-1]
                f_worst = simplex_f[-1]
                x_second_worst = simplex[-2]
                f_second_worst = simplex_f[-2]
                x_best = simplex[0]
                f_best = simplex_f[0]
                
                # Centroid of all points except the worst
                x_c = np.mean(simplex[:-1], axis=0)
                
                # Reflection
                x_r = x_c + alpha * (x_c - x_worst)
                x_r = np.clip(x_r, -5, 5)
                f_r = safe_func(x_r)
                
                if self.calls >= self.budget or time.time() > deadline:
                    break
                
                if f_r < f_best:
                    # Try Expansion
                    x_e = x_c + gamma * (x_r - x_c)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = safe_func(x_e)
                    
                    if self.calls >= self.budget or time.time() > deadline:
                        break
                    
                    if f_e < f_r:
                        # Accept Expansion
                        simplex[-1] = x_e
                        simplex_f[-1] = f_e
                    else:
                        # Accept Reflection
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                        
                elif f_r < f_second_worst:
                    # Accept Reflection
                    simplex[-1] = x_r
                    simplex_f[-1] = f_r
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside Contraction
                        x_c_new = x_c + 0.5 * (x_r - x_c)
                    else:
                        # Inside Contraction
                        x_c_new = x_c - 0.5 * (x_r - x_c)
                        
                    x_c_new = np.clip(x_c_new, -5, 5)
                    f_c = safe_func(x_c_new)
                    
                    if self.calls >= self.budget or time.time() > deadline:
                        break
                    
                    if f_c <= min(f_r, f_worst):
                        simplex[-1] = x_c_new
                        simplex_f[-1] = f_c
                    else:
                        # Shrinkage
                        for i in range(1, self.dim + 1):
                            simplex[i] = x_best + sigma * (simplex[i] - x_best)
                            simplex_f[i] = safe_func(simplex[i])
                            if self.calls >= self.budget or time.time() > deadline:
                                break
                        
                        if self.calls >= self.budget or time.time() > deadline:
                            break
                        
                # Re-sort simplex
                order = np.argsort(simplex_f)
                simplex = simplex[order]
                simplex_f = simplex_f[order]
                
                # Check convergence criteria (optional, but helps avoid infinite loops in flat areas)
                # If the spread of function values is very small, we might be converged
                if np.std(simplex_f) < 1e-10:
                    break

        return self.best_x, self.best_f
