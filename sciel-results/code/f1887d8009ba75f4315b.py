"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Differential Evolution optimizer for BBOB f17."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin for 300s limit
        
        dim = self.dim
        total_budget = self.budget
        
        # DE Parameters
        # Population size: Standard is 5*dim to 10*dim. 
        # For dim=5, let's use 25 or 30. 
        # Smaller pop allows more generations (more exploitation) but risks premature convergence.
        # Larger pop allows better global coverage.
        # Let's try N=20 (4*dim) to balance.
        N = 20 
        
        # Mutation factor F and Crossover rate CR
        F = 0.8
        CR = 0.9
        
        # Scaling: The box is [-5, 5]. 
        # Schaffers F7 is multimodal.
        
        best_x = None
        best_f = np.inf
        
        # We will run a single DE instance with a large population and many generations,
        # but monitor progress. If stuck, we could restart, but with N=20 and 500 budget,
        # that's only 25 generations. That's quite few for DE to converge if far away.
        
        # Alternative: Multi-start DE?
        # Or just a very robust single DE?
        
        # Let's stick to a single robust DE run but with a slightly larger population 
        # to ensure diversity, or smaller to allow more steps?
        # 500 / 20 = 25 generations.
        # 500 / 10 = 50 generations.
        # 500 / 15 = 33 generations.
        
        # Let's use N = 15. It's a good compromise.
        N = 15
        
        rng = np.random.default_rng(42)
        
        # Initialize population uniformly in [-5, 5]
        # To improve early progress (AOCC weights early calls heavily), 
        # we should ensure the first few calls are diverse and cover the space well.
        
        # Generate initial population
        pop = rng.uniform(-5, 5, size=(N, dim))
        
        # Evaluate initial population
        fs = np.empty(N)
        calls = 0
        
        for i in range(N):
            if calls >= total_budget:
                break
            if time.time() > deadline - 5.0:
                break
                
            x = pop[i]
            fx = func(x)
            fs[i] = fx
            calls += 1
            
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
                
        if calls < N:
            # If we ran out of budget during init, just return best
            return best_x, best_f
            
        # Sort initial population
        idx = np.argsort(fs)
        pop = pop[idx]
        fs = fs[idx]
        
        # DE Loop
        gen = 0
        while calls < total_budget:
            if time.time() > deadline - 5.0:
                break
                
            # Check if we have enough budget for a full generation
            if calls + N > total_budget:
                # Do a partial generation or just stop?
                # Partial generation is tricky with DE because it needs the whole pop.
                # We can just break and return best.
                break
                
            # Generate offspring
            for i in range(N):
                if calls >= total_budget:
                    break
                if time.time() > deadline - 5.0:
                    break
                    
                # Select two distinct parents different from i
                r1, r2 = i
                while r1 == i:
                    r1 = rng.integers(N)
                while r2 == i or r2 == r1:
                    r2 = rng.integers(N)
                    
                # Mutation: v = x_r1 + F * (x_r2 - x_i)
                v = pop[r1] + F * (pop[r2] - pop[i])
                
                # Crossover: binomial
                j_rand = rng.integers(dim)
                trial = np.empty(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = v[j]
                    else:
                        trial[j] = pop[i][j]
                        
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                ft = func(trial)
                calls += 1
                
                # Selection
                if ft <= fs[i]:
                    pop[i] = trial
                    fs[i] = ft
                    if ft < best_f:
                        best_f = ft
                        best_x = trial.copy()
            
            # Sort population for next generation (optional for standard DE, but helps if we use ranked selection)
            # Standard DE uses tournament or random. We used random distinct.
            # Sorting helps if we want to bias selection towards better individuals.
            # Let's keep it simple: random selection is standard.
            
            gen += 1
            
            # Optional: Check for stagnation
            # If best_f hasn't improved in X generations, we might restart or perturb.
            # But with 500 budget and N=15, we only get ~33 gens. 
            # Stagnation is a risk.
            
        return best_x, best_f
# EVOLVE-BLOCK-END
