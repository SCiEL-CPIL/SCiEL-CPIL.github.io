import numpy as np

class Optimizer:
    """Nelder-Mead simplex optimizer with adaptive restarts for BBOB f5."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        calls_used = 0
        
        # Nelder-Mead parameters
        alpha = 1.0   # reflection
        gamma = 2.0   # expansion
        rho = 0.5     # contraction
        sigma = 0.5   # shrink
        min_step = 1e-8
        
        def nm_step(x0, step_size, max_calls):
            """Run Nelder-Mead from x0 with initial step size."""
            nonlocal calls_used, f_best, x_best
            
            # Initialize simplex: d+1 vertices
            simplex = np.zeros((d + 1, d))
            simplex[0] = x0
            for i in range(1, d + 1):
                simplex[i] = x0.copy()
                simplex[i][i-1] += step_size
            
            # Evaluate initial simplex
            f_values = np.zeros(d + 1)
            for i in range(d + 1):
                if calls_used >= budget:
                    return
                x_i = np.clip(simplex[i], -5, 5)
                f_values[i] = func(x_i)
                calls_used += 1
                if f_values[i] < f_best:
                    f_best = f_values[i]
                    x_best = x_i.copy()
            
            # Nelder-Mead iterations
            max_iter = max(1, (budget - calls_used) // (d + 1))
            for _ in range(max_iter):
                if calls_used >= budget:
                    break
                
                # Sort simplex by function value
                order = np.argsort(f_values)
                simplex = simplex[order]
                f_values = f_values[order]
                
                # Check convergence
                if np.max(f_values) - np.min(f_values) < min_step:
                    break
                
                # Centroid of all but worst point
                centroid = np.mean(simplex[:d], axis=0)
                
                # Reflection
                x_ref = centroid + alpha * (centroid - simplex[-1])
                x_ref_clipped = np.clip(x_ref, -5, 5)
                if calls_used >= budget:
                    break
                f_ref = func(x_ref_clipped)
                calls_used += 1
                if f_ref < f_best:
                    f_best = f_ref
                    x_best = x_ref_clipped.copy()
                
                if f_ref < f_values[0]:
                    # Try expansion
                    x_exp = centroid + gamma * (x_ref - centroid)
                    x_exp_clipped = np.clip(x_exp, -5, 5)
                    if calls_used >= budget:
                        simplex[-1] = x_ref_clipped
                        f_values[-1] = f_ref
                        continue
                    f_exp = func(x_exp_clipped)
                    calls_used += 1
                    if f_exp < f_best:
                        f_best = f_exp
                        x_best = x_exp_clipped.copy()
                    if f_exp < f_ref:
                        simplex[-1] = x_exp_clipped
                        f_values[-1] = f_exp
                    else:
                        simplex[-1] = x_ref_clipped
                        f_values[-1] = f_ref
                elif f_ref < f_values[-2]:
                    # Accept reflection
                    simplex[-1] = x_ref_clipped
                    f_values[-1] = f_ref
                else:
                    # Contraction
                    if f_ref < f_values[-1]:
                        # Outside contraction
                        x_con = centroid + rho * (x_ref - centroid)
                    else:
                        # Inside contraction
                        x_con = centroid - rho * (simplex[-1] - centroid)
                    
                    x_con_clipped = np.clip(x_con, -5, 5)
                    if calls_used >= budget:
                        break
                    f_con = func(x_con_clipped)
                    calls_used += 1
                    if f_con < f_best:
                        f_best = f_con
                        x_best = x_con_clipped.copy()
                    
                    if f_con < min(f_ref, f_values[-1]):
                        simplex[-1] = x_con_clipped
                        f_values[-1] = f_con
                    else:
                        # Shrink
                        for i in range(1, d + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            x_shr = np.clip(simplex[i], -5, 5)
                            if calls_used >= budget:
                                break
                            f_shr = func(x_shr)
                            calls_used += 1
                            if f_shr < f_best:
                                f_best = f_shr
                                x_best = x_shr.copy()
                            f_values[i] = f_shr
                        if calls_used >= budget:
                            break
        
        # Track best solution
        x_best = None
        f_best = np.inf
        
        # Phase 1: Coordinate descent to find a good starting point
        x = np.zeros(d)
        x = np.clip(x, -5, 5)
        f_best = func(x)
        calls_used += 1
        x_best = x.copy()
        
        # Adaptive coordinate descent
        step_sizes = np.ones(d) * 1.0
        coord_dir = 1
        no_improve = 0
        
        while calls_used < budget * 0.3:
            improved = False
            for i in range(d):
                if calls_used >= budget:
                    break
                
                # Try positive and negative directions
                for direction in [1, -1]:
                    x_trial = x.copy()
                    x_trial[i] += direction * step_sizes[i]
                    x_trial = np.clip(x_trial, -5, 5)
                    
                    f_trial = func(x_trial)
                    calls_used += 1
                    
                    if f_trial < f_best:
                        x = x_trial
                        f_best = f_trial
                        x_best = x.copy()
                        step_sizes[i] *= 1.5
                        improved = True
                        break
                    else:
                        step_sizes[i] *= 0.5
                
                if calls_used >= budget:
                    break
            
            if not improved:
                no_improve += 1
                if no_improve > 3:
                    break
            else:
                no_improve = 0
        
        # Phase 2: Nelder-Mead from best point found
        step_size = 0.5
        nm_step(x_best.copy(), step_size, budget - calls_used)
        
        # Phase 3: If budget remains, do more Nelder-Mead with smaller step
        if calls_used < budget:
            nm_step(x_best.copy(), 0.1, budget - calls_used)
        
        # Phase 4: Final local search
        if calls_used < budget:
            step_size = 0.01
            nm_step(x_best.copy(), step_size, budget - calls_used)
        
        return x_best, f_best
