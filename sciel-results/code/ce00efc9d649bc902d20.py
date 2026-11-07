import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        n = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 50
        if N > budget:
            N = budget
            
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, n))
        F = np.array([problem(x) for x in X])
        used = N
        
        # DE Parameters
        F_factor = 0.8  # Scaling factor
        CR = 0.9        # Crossover rate
        
        # Helper: Check if a dominates b
        def dominates(a, b):
            return all(a <= b) and any(a < b)
        
        # Main Loop
        while used < budget:
            # Select base individual
            base_idx = np.random.randint(N)
            
            # Select two distinct random individuals
            r1, r2 = np.random.choice(N, 2, replace=False)
            while r1 == base_idx or r2 == base_idx:
                r1, r2 = np.random.choice(N, 2, replace=False)
                
            x_base = X[base_idx]
            x_r1 = X[r1]
            x_r2 = X[r2]
            
            # Differential Evolution variation
            # v = x_base + F * (x_r1 - x_r2)
            v = x_base + F_factor * (x_r1 - x_r2)
            
            # Binomial crossover
            x_trial = x_base.copy()
            j_rand = np.random.randint(n)
            for j in range(n):
                if np.random.random() < CR or j == j_rand:
                    x_trial[j] = v[j]
                    
            # Clip to bounds
            x_trial = np.clip(x_trial, lo, hi)
            
            # Evaluate
            f_trial = problem(x_trial)
            used += 1
            
            # Selection
            f_base = F[base_idx]
            
            if dominates(f_trial, f_base):
                # Trial dominates base, replace
                X[base_idx] = x_trial
                F[base_idx] = f_trial
            elif dominates(f_base, f_trial):
                # Base dominates trial, discard
                pass
            else:
                # Non-dominated
                # Replace if trial has better diversity or randomly
                # Simple diversity: distance to nearest neighbor in objective space
                min_dist_base = float('inf')
                min_dist_trial = float('inf')
                
                for j in range(N):
                    if j == base_idx:
                        continue
                    dist_base = np.sum((f_base - F[j])**2)
                    if dist_base < min_dist_base:
                        min_dist_base = dist_base
                    dist_trial = np.sum((f_trial - F[j])**2)
                    if dist_trial < min_dist_trial:
                        min_dist_trial = dist_trial
                        
                # If trial is more isolated (larger min_dist) or random
                if min_dist_trial > min_dist_base or np.random.random() < 0.5:
                    X[base_idx] = x_trial
                    F[base_idx] = f_trial

        return X
