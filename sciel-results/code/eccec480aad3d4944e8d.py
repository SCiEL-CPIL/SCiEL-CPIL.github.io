import numpy as np
import time

class Optimizer:
    """Multi-start Nelder-Mead optimizer for Katsuura function."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 330s limit, stop at 290s
        deadline = start_time + 290.0 
        rng = np.random.default_rng(42)
        
        best_x = None
        best_f = float('inf')
        
        # Strategy: Multi-start Nelder-Mead
        # Allocate budget to multiple independent runs
        # Katsuura is multimodal, so diverse starts are crucial.
        # Let's use 20 restarts.
        num_restarts = 20
        budget_per_run = self.budget // num_restarts
        
        # If budget_per_run is too small to even form a simplex (dim+1), adjust
        min_budget_for_simplex = self.dim + 1
        if budget_per_run < min_budget_for_simplex:
            num_restarts = max(1, self.budget // min_budget_for_simplex)
            budget_per_run = self.budget // num_restarts

        for r in range(num_restarts):
            if time.time() > deadline:
                break
                
            # 1. Initialize Simplex
            # Random center
            x0 = rng.uniform(-5, 5, self.dim)
            # Simplex points: x0 and x0 + alpha * e_i
            alpha = 0.5
            simplex_points = [x0.copy()]
            for i in range(self.dim):
                xi = x0.copy()
                xi[i] += alpha
                simplex_points.append(xi)
            
            # Evaluate initial simplex
            simplex_vals = []
            eval_count = 0
            for p in simplex_points:
                if eval_count >= budget_per_run:
                    break
                # Clip to be safe, though func does it
                p_clipped = np.clip(p, -5, 5)
                fp = func(p_clipped)
                simplex_vals.append(fp)
                eval_count += 1
                if fp < best_f:
                    best_f = fp
                    best_x = p_clipped.copy()
            
            # If we couldn't evaluate full simplex, skip to next restart
            if len(simplex_vals) < self.dim + 1:
                continue

            # Sort simplex by function value
            # List of (value, point)
            simp = sorted(zip(simplex_vals, simplex_points), key=lambda t: t[0])
            
            # 2. Nelder-Mead Iteration
            # Parameters
            rho = 1.0  # Reflection
            chi = 2.0  # Expansion
            gamma = 0.5 # Contraction
            sigma = 0.5 # Shrink
            
            while eval_count < budget_per_run:
                if time.time() > deadline:
                    break
                    
                f_worst, x_worst = simp[-1]
                f_best, x_best = simp[0]
                
                # Centroid of all points except worst
                centroid = np.mean([x for f, x in simp[:-1]], axis=0)
                
                # Reflection
                x_r = centroid - rho * (x_worst - centroid)
                x_r = np.clip(x_r, -5, 5)
                f_r = func(x_r)
                eval_count += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                
                if f_r < simp[0][0] and f_r >= simp[1][0]:
                    # Accept reflection
                    simp[-1] = (f_r, x_r)
                    simp.sort(key=lambda t: t[0])
                elif f_r < simp[0][0]:
                    # Expansion
                    x_e = centroid - chi * (x_worst - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    f_e = func(x_e)
                    eval_count += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                    
                    if f_e < f_r:
                        simp[-1] = (f_e, x_e)
                    else:
                        simp[-1] = (f_r, x_r)
                    simp.sort(key=lambda t: t[0])
                elif f_r < f_worst:
                    # Accept reflection
                    simp[-1] = (f_r, x_r)
                    simp.sort(key=lambda t: t[0])
                else:
                    # Contraction
                    if f_r < f_worst:
                        # Outside contraction
                        x_c = centroid + gamma * (x_r - centroid)
                    else:
                        # Inside contraction
                        x_c = centroid - gamma * (x_worst - centroid)
                    
                    x_c = np.clip(x_c, -5, 5)
                    f_c = func(x_c)
                    eval_count += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                    
                    if f_c < min(f_r, f_worst):
                        simp[-1] = (f_c, x_c)
                        simp.sort(key=lambda t: t[0])
                    else:
                        # Shrink
                        # Shrink requires evaluating dim points.
                        # Check if we have enough budget
                        if eval_count + self.dim > budget_per_run:
                            break
                            
                        for i in range(1, len(simp)):
                            # Move point i towards best point
                            new_p = simp[0][1] + sigma * (simp[i][1] - simp[0][1])
                            new_p = np.clip(new_p, -5, 5)
                            new_f = func(new_p)
                            eval_count += 1
                            if new_f < best_f:
                                best_f = new_f
                                best_x = new_p.copy()
                            simp[i] = (new_f, new_p)
                        
                        simp.sort(key=lambda t: t[0])
                        
        return best_x, best_f
