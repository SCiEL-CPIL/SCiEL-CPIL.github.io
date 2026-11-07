import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        n = self.dim
        budget = self.budget
        calls_made = 0
        
        # Phase 1: Global Exploration with Coordinate Descent refinement
        # Use fewer initial points but refine each one with coordinate descent
        # to exploit ridge structure early on
        n_explore = 15
        points = rng.uniform(-5, 5, (n_explore, n))
        
        # For each point, do a quick coordinate descent to find local ridge
        # This uses budget more efficiently by leveraging the ridge structure
        # Each coordinate descent step uses 1 evaluation per coordinate
        
        best_x = None
        best_f = np.inf
        
        # Reserve budget for refinement
        # Use 10 points with ~25 evaluations each for initial coordinate descent
        # That's 250 evaluations, leaving 250 for final refinement
        evals_per_cd = 25
        
        for i in range(n_explore):
            if calls_made >= budget - 50:  # Keep 50 for final refinement
                break
            if time.time() > deadline:
                break
            
            x_curr = points[i].copy()
            f_curr = func(x_curr)
            calls_made += 1
            
            if f_curr < best_f:
                best_f = f_curr
                best_x = x_curr.copy()
            
            # Quick coordinate descent from this point
            # Cycle through coordinates, trying +/- steps
            step_sizes = [2.0, 1.0, 0.5, 0.25, 0.125]
            
            for step in step_sizes:
                if calls_made >= budget - 50:
                    break
                if time.time() > deadline:
                    break
                
                improved = True
                while improved and calls_made < budget - 50:
                    improved = False
                    for j in range(n):
                        if calls_made >= budget - 50:
                            break
                        if time.time() > deadline:
                            break
                        
                        # Try positive step
                        x_test = x_curr.copy()
                        x_test[j] = np.clip(x_test[j] + step, -5, 5)
                        f_test = func(x_test)
                        calls_made += 1
                        
                        if f_test < f_curr:
                            x_curr = x_test
                            f_curr = f_test
                            improved = True
                            if f_curr < best_f:
                                best_f = f_curr
                                best_x = x_curr.copy()
                        
                        # Try negative step
                        if calls_made < budget - 50:
                            x_test = x_curr.copy()
                            x_test[j] = np.clip(x_test[j] - step, -5, 5)
                            f_test = func(x_test)
                            calls_made += 1
                            
                            if f_test < f_curr:
                                x_curr = x_test
                                f_curr = f_test
                                improved = True
                                if f_curr < best_f:
                                    best_f = f_curr
                                    best_x = x_curr.copy()
                
                # Halve step size for next iteration
                # Already done by iterating through step_sizes list
        
        if best_x is None:
            # Fallback
            x0 = np.zeros(n)
            f0 = func(x0)
            calls_made += 1
            return x0, f0
        
        x_curr = best_x.copy()
        f_curr = best_f
        
        # Phase 2: Fine-grained coordinate descent for final refinement
        # Use remaining budget for very fine coordinate search
        remaining = budget - calls_made
        if remaining > 0:
            # Adaptive step sizes starting small
            step = 0.1
            min_step = 1e-4
            
            while calls_made < budget:
                if time.time() > deadline:
                    break
                
                improved = False
                for j in range(n):
                    if calls_made >= budget:
                        break
                    if time.time() > deadline:
                        break
                    
                    # Try positive step
                    x_test = x_curr.copy()
                    x_test[j] = np.clip(x_test[j] + step, -5, 5)
                    f_test = func(x_test)
                    calls_made += 1
                    
                    if f_test < f_curr:
                        x_curr = x_test
                        f_curr = f_test
                        improved = True
                    
                    # Try negative step
                    if calls_made < budget:
                        x_test = x_curr.copy()
                        x_test[j] = np.clip(x_test[j] - step, -5, 5)
                        f_test = func(x_test)
                        calls_made += 1
                        
                        if f_test < f_curr:
                            x_curr = x_test
                            f_curr = f_test
                            improved = True
                
                if not improved:
                    # No improvement, reduce step size
                    step *= 0.5
                    if step < min_step:
                        break
        
        return x_curr, f_curr
