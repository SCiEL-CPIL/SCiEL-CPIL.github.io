import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(0)
        
        # Use a population size that divides the budget evenly or close to it
        # N=10 is a good balance for dim=5
        N = 10
        if self.budget < N:
            # Fallback for very small budgets, though not expected here
            x = rng.uniform(-5, 5, self.dim)
            fx = func(x)
            return x, fx
            
        # Number of generations
        # We use N evaluations for initialization, then N per generation
        gens = (self.budget - N) // N
        
        # Initialize population
        P = rng.uniform(-5, 5, (N, self.dim))
        fP = np.array([func(x) for x in P])
        
        best_idx = np.argmin(fP)
        best_x = P[best_idx].copy()
        best_f = fP[best_idx]
        
        F = 0.7  # Mutation factor
        CR = 0.8 # Crossover rate
        
        for g in range(gens):
            for i in range(N):
                # Select 3 distinct individuals different from i
                # Efficient selection
                while True:
                    a, b, c = rng.choice(N, 3, replace=False)
                    if a != i and b != i and c != i:
                        break
                
                # Mutation
                v = P[a] + F * (P[b] - P[c])
                
                # Crossover (Binomial)
                # j_rand ensures at least one dimension is taken from v
                j_rand = rng.integers(self.dim)
                mask = rng.random(self.dim) <= CR
                mask[j_rand] = True
                u = np.where(mask, v, P[i])
                
                # Clip to bounds
                u = np.clip(u, -5, 5)
                
                # Evaluation
                fu = func(u)
                
                # Selection
                if fu <= fP[i]:
                    P[i] = u
                    fP[i] = fu
                    if fu < best_f:
                        best_f = fu
                        best_x = u.copy()
                        
        return best_x, best_f
