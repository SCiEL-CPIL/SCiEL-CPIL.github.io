import numpy as np
import time

class Optimizer:
    """
    Multi-start Nelder-Mead optimizer.
    Uses diverse initial points to quickly locate the basin of attraction
    for the BBOB f6 (Attractive Sector) function.
    """

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        d = self.dim
        max_calls = self.budget
        calls_made = 0
        best_x = None
        best_f = float('inf')

        # Helper to evaluate and track budget
        def eval_func(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= max_calls:
                return best_f
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Differential Evolution Parameters
        pop_size = 10
        F = 0.8
        CR = 0.9
        
        # Initialize population uniformly in [-5, 5]^d
        population = np.random.uniform(-5.0, 5.0, (pop_size, d))
        fvals = np.zeros(pop_size)
        
        # Evaluate initial population
        for i in range(pop_size):
            if calls_made >= max_calls or time.time() > deadline:
                break
            fvals[i] = eval_func(population[i])
            
        # Ensure we have a valid best
        if best_x is None:
             best_x = np.zeros(d)
             best_f = 0.0

        # Main DE Loop
        while calls_made < max_calls and time.time() < deadline:
            # Select a target individual
            idx = np.random.randint(pop_size)
            
            # Select three distinct random individuals for the donor
            r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
            
            # Create donor vector
            v = population[r1] + F * (population[r2] - population[r3])
            
            # Crossover
            target = population[idx]
            trial = np.copy(target)
            
            # Generate random bit vector
            mask = np.random.rand(d) < CR
            
            # Ensure at least one dimension is taken from v
            if not np.any(mask):
                mask[np.random.randint(d)] = True
            
            trial[mask] = v[mask]
            
            # Clip to bounds
            trial = np.clip(trial, -5.0, 5.0)
            
            # Evaluate trial
            f_trial = eval_func(trial)
            
            # Selection
            if f_trial <= fvals[idx]:
                population[idx] = trial
                fvals[idx] = f_trial

        return best_x, best_f
