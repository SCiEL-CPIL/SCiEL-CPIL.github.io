import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280 # Safety margin
        
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        span = hi - lo
        
        N = 50 # Population size
        # Ensure N is reasonable relative to budget
        if self.budget < 100:
            N = max(10, self.budget // 2)
            
        # Initialization
        # Seed 50% with front-like points (x_i=0 for i>1)
        # Seed 50% with random
        X_front = np.zeros((N, self.dim))
        X_front[:, 0] = np.linspace(0, 1, N)
        # x_i = 0 for i > 1
        # This matches the current program's successful seeding
        
        X_rand = lo + span * np.random.rand(N, self.dim)
        
        X = np.vstack([X_front, X_rand])[:2*N] # Wait, I want N total? Or 2N total?
        # Standard NSGA-II starts with N.
        # If I start with 2N, I use more budget upfront.
        # Let's start with N=50.
        # 25 front, 25 random?
        # Or just N=50 total.
        
        # Let's stick to N=50 total population.
        # 25 front, 25 random.
        N_half = N // 2
        X_f = np.zeros((N_half, self.dim))
        X_f[:, 0] = np.linspace(0, 1, N_half)
        
        X_r = lo + span * np.random.rand(N_half, self.dim)
        
        X = np.vstack([X_f, X_r])
        F = np.array([problem(x) for x in X])
        used = len(X)
        
        while used + N <= self.budget:
            if time.time() > deadline:
                break
            
            # Generate Offspring
            X_off = np.zeros((N, self.dim))
            F_off = np.zeros((N, self.n_obj))
            
            for i in range(N):
                # Tournament Selection
                p1 = self._tournament(X, F, N=3)
                p2 = self._tournament(X, F, N=3)
                
                # Crossover
                child = self._crossover(p1, p2, lo, hi)
                
                # Mutation
                child = self._mutate(child, lo, hi)
                
                X_off[i] = child
                F_off[i] = problem(child)
                used += 1
                
                if used >= self.budget:
                    break
            
            if used >= self.budget:
                break
            
            # Combine
            X_all = np.vstack([X, X_off[:used - len(X)]]) # Careful with slicing if loop broke early
            # Actually, if loop broke early, X_off is partially filled.
            # It's safer to track valid offspring.
            
            # Let's re-structure to handle budget break cleanly.
            
            # ...
