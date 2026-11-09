import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(42)
        
        best_f = np.inf
        best_x = np.zeros(dim)
        calls_made = 0
        
        # Nelder-Mead parameters
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        # We will run multiple restarts of Nelder-Mead if stuck
        # Each restart gets a portion of the budget or runs until stuck
        # Strategy: Run NM until it converges or uses up budget.
        # If it converges early, restart with a perturbation.
        
        max_restarts = 10
        
        for restart in range(max_restarts):
            if calls_made >= budget:
                break
            
            # Initialize simplex
            # First restart: centered at 0
            # Subsequent: centered at best_x with random perturbation
            if restart == 0:
                center = np.zeros(dim)
                step = 2.0
            else:
                center = best_x.copy()
                # Perturb center slightly
                center += rng.standard_normal(dim) * 0.1
                # Clip to box
                center = np.clip(center, -5, 5)
                step = 1.0 # Smaller step for local refinement? Or larger to escape?
                # Let's try a random direction with step 1.0
            
            simplex = np.zeros((dim + 1, dim))
            for i in range(dim):
                simplex[i] = center
                simplex[i, i] = step
            simplex[dim] = center
            
            # Clip simplex points to box
            simplex = np.clip(simplex, -5, 5)
            
            # Evaluate initial simplex
            fs = []
            for x in simplex:
                if calls_made >= budget:
                    return best_x, best_f
                f = func(x)
                calls_made += 1
                if f < best_f:
                    best_f = f
                    best_x = x.copy()
                fs.append(f)
            
            fs = np.array(fs)
            order = np.argsort(fs)
            simplex = simplex[order]
            fs = fs[order]
            
            # Main NM loop for this restart
            no_progress = 0
            while calls_made < budget:
                # Check convergence
                if fs[-1] - fs[0] < 1e-8 * (1 + abs(fs[0])):
                    break
                
                # Find centroid excluding worst
                centroid = np.sum(simplex[:-1], axis=0) / dim
                
                # Reflection
                x_r = centroid + alpha * (centroid - simplex[dim])
                x_r = np.clip(x_r, -5, 5)
                if calls_made >= budget:
                    return best_x, best_f
                f_r = func(x_r)
                calls_made += 1
                if f_r < best_f:
                    best_f = f_r
                    best_x = x_r.copy()
                    no_progress = 0
                else:
                    no_progress += 1
                    
                if fs[0] <= f_r < fs[dim - 1]:
                    simplex[dim] = x_r
                    fs[dim] = f_r
                elif f_r < fs[0]:
                    # Expansion
                    x_e = centroid + gamma * (x_r - centroid)
                    x_e = np.clip(x_e, -5, 5)
                    if calls_made >= budget:
                        return best_x, best_f
                    f_e = func(x_e)
                    calls_made += 1
                    if f_e < best_f:
                        best_f = f_e
                        best_x = x_e.copy()
                        no_progress = 0
                    else:
                        no_progress += 1
                    if f_e < f_r:
                        simplex[dim] = x_e
                        fs[dim] = f_e
                    else:
                        simplex[dim] = x_r
                        fs[dim] = f_r
                else:
                    # Contraction
                    if f_r < fs[dim]:
                        x_c = centroid + rho * (x_r - centroid)
                    else:
                        x_c = centroid - rho * (simplex[dim] - centroid)
                    x_c = np.clip(x_c, -5, 5)
                    if calls_made >= budget:
                        return best_x, best_f
                    f_c = func(x_c)
                    calls_made += 1
                    if f_c < best_f:
                        best_f = f_c
                        best_x = x_c.copy()
                        no_progress = 0
                    else:
                        no_progress += 1
                    if f_c < min(f_r, fs[dim]):
                        simplex[dim] = x_c
                        fs[dim] = f_c
                    else:
                        # Shrink
                        for i in range(1, dim + 1):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], -5, 5)
                            if calls_made >= budget:
                                return best_x, best_f
                            f_s = func(simplex[i])
                            calls_made += 1
                            if f_s < best_f:
                                best_f = f_s
                                best_x = simplex[i].copy()
                                no_progress = 0
                            else:
                                no_progress += 1
                            fs[i] = f_s
                        
                # Re-sort
                order = np.argsort(fs)
                simplex = simplex[order]
                fs = fs[order]
                
                # If no progress for many iterations, break to restart
                if no_progress > 2 * dim:
                    break

        return best_x, best_f
