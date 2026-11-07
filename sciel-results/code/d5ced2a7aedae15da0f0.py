# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

class Optimizer:
    """
    Hybrid optimizer for BBOB f7 (Step Ellipsoidal).
    Uses a multi-start strategy with:
    1. Random sampling to get a rough idea of the landscape.
    2. Differential Evolution (DE) for global exploration with robustness against flat areas.
    3. (1+1) CMA-ES or Simple ES for local polishing.
    
    The Step Ellipsoidal function has a flat half-space. DE is good at escaping
    flat areas if the population is diverse, but can stagnate if the 'step'
    is missed. We use a multi-start approach with different initializations.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 295.0  # Safety margin (limit is 330s)
        
        dim = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        calls = 0
        best_x = np.zeros(dim)
        best_f = np.inf
        
        def evaluate(x):
            nonlocal calls, best_x, best_f
            if calls >= budget or time.time() > deadline:
                return np.inf
            fx = func(x)
            calls += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Initialize with a few random points to avoid starting in a bad spot
        # and to have some baseline.
        init_count = 10
        for _ in range(init_count):
            x = rng.uniform(-5, 5, size=dim)
            evaluate(x)
        
        if calls >= budget or time.time() > deadline:
            return best_x, best_f

        # Phase 1: Differential Evolution (DE)
        # DE parameters
        pop_size = 12
        F = 0.8
        CR = 0.5
        
        # Initialize population around the current best, but with some spread
        # to ensure we don't get stuck in the flat region if the best is on the boundary.
        # However, for f7, the flat region is usually 'above' the minimum in one direction.
        # Let's start with a diverse population.
        
        # We will run DE for a fixed number of generations or until budget/time runs out.
        # We reserve 20% of the budget for local polishing.
        de_budget = int(budget * 0.8)
        
        # Create initial population
        # Mix of points near best and random points
        pop = np.empty((pop_size, dim))
        pop[0] = best_x
        for i in range(1, pop_size):
            # Some near best, some random
            if i % 3 == 0:
                pop[i] = np.clip(best_x + rng.normal(0, 2.0, size=dim), -5, 5)
            else:
                pop[i] = rng.uniform(-5, 5, size=dim)
        
        pop_f = np.array([evaluate(x) for x in pop])
        
        # DE Loop
        while calls < de_budget and time.time() < deadline:
            # Sort population
            order = np.argsort(pop_f)
            pop = pop[order]
            pop_f = pop_f[order]
            
            # Update global best
            if pop_f[0] < best_f:
                best_f = pop_f[0]
                best_x = pop[0].copy()
            
            # Generate new population
            new_pop = np.empty((pop_size, dim))
            new_pop[0] = pop[0] # Keep best
            new_f = np.empty(pop_size)
            new_f[0] = pop_f[0]
            
            for i in range(1, pop_size):
                # Select 3 distinct indices different from i
                a, b, c = rng.choice([j for j in range(pop_size) if j != i], 3, replace=False)
                
                # DE/rand/1
                mutant = pop[a] + F * (pop[b] - pop[c])
                
                # Crossover
                trial = pop[i].copy()
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        trial[j] = mutant[j]
                
                # Clip
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                ft = evaluate(trial)
                
                # Selection
                if ft <= pop_f[i]:
                    new_pop[i] = trial
                    new_f[i] = ft
                else:
                    new_pop[i] = pop[i]
                    new_f[i] = pop_f[i]
            
            pop = new_pop
            pop_f = new_f
            
            # Stagnation check: if no improvement in last 5 generations, restart DE with new init
            # Simple check: if best_f hasn't changed in a while
            # We track this loosely by checking if pop_f[0] is same as previous best
            # For simplicity, we just let it run. DE is robust.
            
            # Check if we should break early if converged
            if np.std(pop_f[:5]) < 1e-10:
                break

        # Phase 2: Local Polish with (1+1)-ES
        # This is crucial for the final drop in f7 where the step creates a sharp valley.
        if calls < budget and time.time() < deadline:
            x = best_x.copy()
            fx = best_f
            sigma = 0.1
            
            # Adaptive (1+1)-ES
            while calls < budget and time.time() < deadline:
                y = np.clip(x + sigma * rng.standard_normal(dim), -5, 5)
                fy = evaluate(y)
                
                if fy < fx:
                    x = y
                    fx = fy
                    sigma *= 1.1 # Increase step size on success (cautious)
                else:
                    sigma *= 0.9 # Decrease step size on failure
                    
                if sigma < 1e-8:
                    break

        return best_x, best_f

# EVOLVE-BLOCK-END
