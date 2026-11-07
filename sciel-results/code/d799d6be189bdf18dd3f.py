import numpy as np

class Optimizer:
    """Differential Evolution optimizer for BBOB f5 (Linear Slope)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        d = self.dim
        budget = self.budget
        
        # DE Parameters
        # Population size: 10 * dim is a good start for low d
        N = 10 * d 
        # Mutation factor
        F = 0.8
        # Crossover rate
        CR = 0.5
        
        # Initialize population randomly in [-5, 5]
        # Ensure we don't exceed budget
        if N > budget:
            N = budget
            
        X = np.random.uniform(-5, 5, (N, d))
        
        # Evaluate initial population
        F_vals = np.zeros(N)
        calls_used = 0
        for i in range(N):
            if calls_used >= budget:
                break
            F_vals[i] = func(X[i])
            calls_used += 1
            
        # Track best
        best_idx = np.argmin(F_vals)
        x_best = X[best_idx].copy()
        f_best = F_vals[best_idx]
        
        # Main loop
        # Each generation consumes N evaluations
        # We have budget - calls_used remaining
        while calls_used < budget:
            # Check if we can do a full generation
            if budget - calls_used < N:
                # Do partial generation or just stop? 
                # Better to stop if not enough for a full update cycle to maintain DE logic integrity, 
                # or evaluate one by one.
                # For simplicity and safety, if remaining < N, we can still evaluate individuals one by one 
                # but it breaks the population structure. 
                # Given budget=500, N=50 (if d=5, N=50), 500/50 = 10 generations.
                # If d=5, N=50. 
                # Let's stick to N=50 for d=5.
                # If remaining < N, we break.
                break
            
            # Generate trial vectors
            for i in range(N):
                if calls_used >= budget:
                    break
                    
                # Select r1, r2, r3 distinct from i
                r1, r2, r3 = np.random.choice([j for j in range(N) if j != i], 3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F * (X[r2] - X[r3])
                
                # Crossover
                # Binomial crossover
                j_rand = np.random.randint(N) # Not used for binomial, but standard
                # For each dimension, decide whether to take from v or x_i
                cross_mask = np.random.rand(d) < CR
                cross_mask[j_rand] = True # Ensure at least one gene is from v
                
                u = np.where(cross_mask, v, X[i])
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluate
                f_u = func(u)
                calls_used += 1
                
                # Selection
                if f_u <= F_vals[i]:
                    X[i] = u
                    F_vals[i] = f_u
                    if f_u < f_best:
                        f_best = f_u
                        x_best = u.copy()
                        
        return x_best, f_best
