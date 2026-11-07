import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        timeout = 290.0  # Safety margin against 330s limit
        
        rng = np.random.default_rng(0)
        dim = self.dim
        budget = self.budget
        
        best_x = None
        best_f = np.inf
        eval_count = 0
        
        def time_check():
            return (time.time() - start_time) > timeout

        # We will use a multi-start Nelder-Mead approach.
        # Budget 500, Dim 5.
        # Nelder-Mead requires (dim+1) points for the simplex.
        # Each iteration of NM uses 1 new evaluation (reflection, expansion, contraction, shrink).
        # With dim=5, simplex size is 6.
        
        # Strategy:
        # 1. Generate a few diverse starting points.
        # 2. For each start, run Nelder-Mead until budget is exhausted or convergence/stagnation.
        # 3. Keep the global best.
        
        # How many restarts?
        # If we use 100 evals per restart, we can do 5 restarts.
        # NM is very fast in low dimensions.
        
        n_restarts = 5
        budget_per_restart = budget // n_restarts
        
        for r in range(n_restarts):
            if eval_count >= budget or time_check():
                break
            
            # Initialize a new simplex
            # Start point: random in [-5, 5]^dim
            x0 = rng.uniform(-5.0, 5.0, dim)
            
            # Initial simplex size
            delta = 0.5
            
            # Create simplex: x0, and x0 + delta * e_i for i=1..dim
            simplex = np.zeros((dim + 1, dim))
            simplex[0] = x0
            for i in range(dim):
                simplex[i + 1] = x0.copy()
                simplex[i + 1][i] += delta
            
            # Evaluate initial simplex
            f_simplex = np.zeros(dim + 1)
            for i in range(dim + 1):
                if eval_count >= budget or time_check():
                    break
                x_i = simplex[i]
                f_i = func(x_i)
                f_simplex[i] = f_i
                eval_count += 1
                
                if f_i < best_f:
                    best_f = f_i
                    best_x = x_i.copy()
            
            if eval_count >= budget or time_check():
                break
            
            # Nelder-Mead Loop
            # Parameters
            alpha = 1.0    # Reflection
            gamma = 2.0    # Expansion
            rho = 0.5      # Contraction
            sigma = 0.5    # Shrinkage
            
            # Track stagnation to decide if we should restart early
            prev_best_f = best_f
            stagnant_iters = 0
            
            while eval_count < budget and not time_check():
                # Sort simplex by fitness
                idx = np.argsort(f_simplex)
                simplex = simplex[idx]
                f_simplex = f_simplex[idx]
                
                # Current best in this simplex
                f_best_local = f_simplex[0]
                
                # Update global best
                if f_best_local < best_f:
                    best_f = f_best_local
                    best_x = simplex[0].copy()
                    stagnant_iters = 0
                else:
                    stagnant_iters += 1
                    
                # If we are stagnating significantly, break and restart
                # This allows us to use the remaining budget for a new start
                if stagnant_iters > 20:
                    break
                
                # Centroid of all points except the worst
                centroid = np.mean(simplex[:-1], axis=0)
                
                # Worst point
                x_worst = simplex[-1]
                f_worst = f_simplex[-1]
                
                # 1. Reflection
                x_r = centroid + alpha * (centroid - x_worst)
                f_r = func(x_r)
                eval_count += 1
                
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                if f_r < f_simplex[0]:
                    # 2. Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    f_e = func(x_e)
                    eval_count += 1
                    
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simplex[-1] = x_e
                        f_simplex[-1] = f_e
                    else:
                        simplex[-1] = x_r
                        f_simplex[-1] = f_r
                        
                elif f_r < f_simplex[-2]:
                    # Accept reflection
                    simplex[-1] = x_r
                    f_simplex[-1] = f_r
                else:
                    # 3. Contraction
                    # Check if contraction is towards best or away (outside)
                    if f_r < f_worst:
                        # Inside contraction
                        x_c = centroid + rho * (x_r - centroid)
                        f_c = func(x_c)
                        eval_count += 1
                        
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                        
                        if f_c < f_r:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # 4. Shrinkage
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if eval_count < budget and not time_check():
                                    f_simplex[i] = func(simplex[i])
                                    eval_count += 1
                                    if f_simplex[i] < best_f:
                                        best_f = f_simplex[i]
                                        best_x = simplex[i].copy()
                                else:
                                    break
                    else:
                        # Outside contraction
                        x_c = centroid - rho * (x_worst - centroid)
                        f_c = func(x_c)
                        eval_count += 1
                        
                        if f_c < best_f:
                            best_f = f_c
                            best_x = x_c.copy()
                        
                        if f_c < f_worst:
                            simplex[-1] = x_c
                            f_simplex[-1] = f_c
                        else:
                            # 4. Shrinkage
                            for i in range(1, dim + 1):
                                simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                                if eval_count < budget and not time_check():
                                    f_simplex[i] = func(simplex[i])
                                    eval_count += 1
                                    if f_simplex[i] < best_f:
                                        best_f = f_simplex[i]
                                        best_x = simplex[i].copy()
                                else:
                                    break

        if best_x is None:
            # Fallback if no evaluations were made (should not happen)
            best_x = np.zeros(dim)
            best_f = func(best_x)
            
        return best_x, best_f
