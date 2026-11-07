import numpy as np
import time

class Optimizer:
    """
    Hybrid Multi-Start Adaptive Coordinate Descent Optimizer.
    
    Uses a global multi-start strategy with random restarts and local perturbation.
    Each start point is polished with an Adaptive Coordinate Descent (ACD) local search.
    ACD adapts step sizes per dimension, which is effective for anisotropic landscapes 
    like the Sharp Ridge (f13) where curvature varies significantly across axes.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        # Safety margin: 300s limit, use 280s to avoid timeout
        deadline = start_time + 280.0
        
        best_x = None
        best_f = np.inf
        calls_made = 0
        max_calls = self.budget
        
        def local_search(x, f, remaining_calls):
            """
            Adaptive Coordinate Descent.
            Returns (x, f, calls_used).
            """
            n = self.dim
            steps = np.ones(n)
            calls_used = 0
            
            # Perform a few sweeps of coordinate descent
            # Each sweep tries to improve each dimension
            for sweep in range(5):
                if calls_used >= remaining_calls or time.time() > deadline:
                    break
                improved_this_sweep = False
                for i in range(n):
                    if calls_used >= remaining_calls or time.time() > deadline:
                        break
                    
                    # Try positive step
                    x_trial = x.copy()
                    x_trial[i] = np.clip(x[i] + steps[i], -5.0, 5.0)
                    f_trial = func(x_trial)
                    calls_used += 1
                    
                    if f_trial < f:
                        x, f = x_trial, f_trial
                        steps[i] = min(steps[i] * 1.5, 10.0)
                        improved_this_sweep = True
                    else:
                        # Try negative step
                        x_trial = x.copy()
                        x_trial[i] = np.clip(x[i] - steps[i], -5.0, 5.0)
                        f_trial = func(x_trial)
                        calls_used += 1
                        
                        if f_trial < f:
                            x, f = x_trial, f_trial
                            steps[i] = min(steps[i] * 1.5, 10.0)
                            improved_this_sweep = True
                        else:
                            # No improvement, shrink step size for this dimension
                            steps[i] = max(steps[i] * 0.5, 1e-4)
                            
                if not improved_this_sweep:
                    # If no dimension improved in this sweep, shrink all steps 
                    # to allow for finer adjustments or to detect convergence
                    steps *= 0.5
                    if np.all(steps < 1e-5):
                        break

            return x, f, calls_used

        # 1. Initial random point
        x = np.random.uniform(-5.0, 5.0, self.dim)
        f = func(x)
        calls_made += 1
        
        if f < best_f:
            best_f = f
            best_x = x.copy()

        # 2. Main loop: Multi-start with local polish
        while calls_made < max_calls and time.time() < deadline:
            # Decide next start point
            # 10% chance of fully random restart to escape local minima
            if np.random.rand() < 0.1:
                x = np.random.uniform(-5.0, 5.0, self.dim)
            else:
                # Perturb the current best solution
                # Use a moderate sigma for exploration around the best
                sigma = 1.0
                x = best_x + sigma * np.random.randn(self.dim)
                x = np.clip(x, -5.0, 5.0)
            
            # Evaluate the start point of the local search
            f_start = func(x)
            calls_made += 1
            
            if f_start < best_f:
                best_f = f_start
                best_x = x.copy()

            # Run local search from x
            # Allocate up to 15 evaluations for local search
            local_budget = min(max_calls - calls_made, 15)
            if local_budget <= 0:
                break
                
            x_new, f_new, used = local_search(x, f_start, local_budget)
            calls_made += used
            
            if f_new < best_f:
                best_f = f_new
                best_x = x_new.copy()

        return best_x, best_f
