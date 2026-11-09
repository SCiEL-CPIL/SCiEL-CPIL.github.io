import numpy as np

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        rng = np.random.default_rng(42)
        
        # DE Parameters
        N = 10  # Population size
        CR = 0.9  # Crossover rate
        F = 0.8   # Scaling factor
        
        # Initialize population uniformly in [-5, 5]
        pop = rng.uniform(-5, 5, size=(N, self.dim))
        
        # Evaluate initial population
        fitnesses = np.array([func(x) for x in pop])
        evals = N
        
        # Track global best
        global_best_idx = np.argmin(fitnesses)
        global_best = pop[global_best_idx].copy()
        global_best_fitness = fitnesses[global_best_idx]
        
        # Main loop
        while evals < self.budget:
            remaining = self.budget - evals
            
            if remaining < N:
                # If not enough for a full generation, evaluate the remaining individuals one by one
                for _ in range(remaining):
                    # Pick a random individual to mutate
                    i1 = rng.integers(0, N)
                    i2 = rng.integers(0, N)
                    i3 = rng.integers(0, N)
                    while i2 == i1: i2 = rng.integers(0, N)
                    while i3 == i1 or i3 == i2: i3 = rng.integers(0, N)
                    
                    # Mutation
                    v = pop[i1] + F * (pop[i2] - pop[i3])
                    
                    # Crossover
                    trial = np.copy(pop[i1])
                    j_rand = rng.integers(0, self.dim)
                    for j in range(self.dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluate
                    f_trial = func(trial)
                    evals += 1
                    
                    if f_trial < fitnesses[i1]:
                        fitnesses[i1] = f_trial
                        pop[i1] = trial
                        if f_trial < global_best_fitness:
                            global_best_fitness = f_trial
                            global_best = trial.copy()
                break
            else:
                # Perform a full generation
                new_pop = pop.copy()
                new_fitnesses = fitnesses.copy()
                
                for i in range(N):
                    # Select three distinct individuals for mutation
                    i1 = rng.integers(0, N)
                    i2 = rng.integers(0, N)
                    i3 = rng.integers(0, N)
                    while i2 == i1: i2 = rng.integers(0, N)
                    while i3 == i1 or i3 == i2: i3 = rng.integers(0, N)
                    
                    # Mutation: v = x_i1 + F * (x_i2 - x_i3)
                    v = pop[i1] + F * (pop[i2] - pop[i3])
                    
                    # Crossover
                    trial = np.copy(pop[i])
                    j_rand = rng.integers(0, self.dim)
                    for j in range(self.dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                    trial = np.clip(trial, -5, 5)
                    
                    # Evaluate
                    f_trial = func(trial)
                    evals += 1
                    
                    # Selection
                    if f_trial < fitnesses[i]:
                        new_pop[i] = trial
                        new_fitnesses[i] = f_trial
                        if f_trial < global_best_fitness:
                            global_best_fitness = f_trial
                            global_best = trial.copy()
                
                pop = new_pop
                fitnesses = new_fitnesses

        return global_best, global_best_fitness
