import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit
        
        dim = self.dim
        budget = self.budget
        calls_made = 0
        rng = np.random.default_rng(42)
        
        # Initial point: center of the box is a good start for rotated functions
        x = np.zeros(dim)
        fx = func(x)
        calls_made += 1
        
        best_x = x.copy()
        best_fx = fx
        
        sigma = 2.0  # Initial step size
        # 1/5 success rule parameters
        success_window_size = 10
        successes = 0
        trials = 0
        
        # Phase 1: (1+1)-ES with 1/5 success rule
        # Use 85% of budget for global search
        es_budget = int(budget * 0.85)
        
        while calls_made < es_budget:
            if time.time() > deadline - 10:
                break
            
            # Generate candidate
            z = rng.standard_normal(dim)
            y = np.clip(x + sigma * z, -5.0, 5.0)
            
            fy = func(y)
            calls_made += 1
            
            # Update best
            if fy < best_fx:
                best_fx = fy
                best_x = y.copy()
            
            # Update success counter
            trials += 1
            if fy < fx:
                # Accept
                x = y
                fx = fy
                successes += 1
            else:
                # Reject
                pass
            
            # Adapt sigma based on 1/5 rule
            if trials >= success_window_size:
                success_rate = successes / trials
                if success_rate > 0.2:
                    sigma *= 1.1
                elif success_rate < 0.2:
                    sigma *= 0.9
                # Reset counters
                successes = 0
                trials = 0
            
            # Clip sigma to reasonable bounds
            sigma = np.clip(sigma, 1e-4, 10.0)
        
        # Phase 2: Nelder-Mead Polish
        # Use remaining budget for local refinement
        nm_budget = budget - calls_made
        
        if nm_budget > 0 and time.time() < deadline - 5:
            # Initialize simplex around best_x
            simplex = [best_x.copy()]
            simplex_f = [best_fx]
            
            # Create vertices by perturbing each dimension
            for i in range(dim):
                v = best_x.copy()
                # Perturbation size relative to current sigma or fixed small value
                pert = 0.1
                if v[i] + pert > 5.0:
                    v[i] -= pert
                else:
                    v[i] += pert
                simplex.append(v)
                fv = func(v)
                calls_made += 1
                simplex_f.append(fv)
                
                if calls_made >= budget:
                    break
            
            if len(simplex) == dim + 1:
                # Run Nelder-Mead
                alpha = 1.0
                gamma = 2.0
                rho = 0.5
                sigma_nm = 0.5 # Shrink factor for contraction
                
                while calls_made < budget:
                    if time.time() > deadline - 2:
                        break
                    
                    # Sort simplex by function value
                    order = np.argsort(simplex_f)
                    simplex = [simplex[i] for i in order]
                    simplex_f = [simplex_f[i] for i in order]
                    
                    # Centroid of all but the worst point
                    centroid = np.mean(simplex[:-1], axis=0)
                    
                    # Reflect
                    x_r = centroid + alpha * (centroid - simplex[-1])
                    x_r = np.clip(x_r, -5.0, 5.0)
                    f_r = func(x_r)
                    calls_made += 1
                    
                    if f_r < simplex_f[0]:
                        # Expand
                        x_e = centroid + gamma * (x_r - centroid)
                        x_e = np.clip(x_e, -5.0, 5.0)
                        if calls_made < budget:
                            f_e = func(x_e)
                            calls_made += 1
                            if f_e < f_r:
                                simplex[-1] = x_e
                                simplex_f[-1] = f_e
                            else:
                                simplex[-1] = x_r
                                simplex_f[-1] = f_r
                        else:
                            simplex[-1] = x_r
                            simplex_f[-1] = f_r
                    elif f_r < simplex_f[-2]:
                        # Accept reflection
                        simplex[-1] = x_r
                        simplex_f[-1] = f_r
                    else:
                        # Contract
                        if f_r < simplex_f[-1]:
                            # Inner contraction
                            x_c = centroid + rho * (x_r - centroid)
                        else:
                            # Outer contraction
                            x_c = centroid - rho * (simplex[-1] - centroid)
                        x_c = np.clip(x_c, -5.0, 5.0)
                        
                        if calls_made < budget:
                            f_c = func(x_c)
                            calls_made += 1
                            if f_c < simplex_f[-1]:
                                simplex[-1] = x_c
                                simplex_f[-1] = f_c
                            else:
                                # Shrink
                                for i in range(1, len(simplex)):
                                    simplex[i] = simplex[0] + sigma_nm * (simplex[i] - simplex[0])
                                    if calls_made < budget:
                                        simplex_f[i] = func(simplex[i])
                                        calls_made += 1
                                    else:
                                        break
                                if calls_made >= budget:
                                    break
                    
                    # Update best
                    current_best_idx = np.argmin(simplex_f)
                    if simplex_f[current_best_idx] < best_fx:
                        best_fx = simplex_f[current_best_idx]
                        best_x = simplex[current_best_idx].copy()
        
        return best_x, best_fx
