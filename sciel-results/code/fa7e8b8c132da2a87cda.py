import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        eval_count = 0
        best_x = np.zeros(dim)
        best_f = np.inf

        def time_check():
            return (time.time() - start_time) > deadline

        def update_best(x, f):
            nonlocal best_x, best_f
            if f < best_f:
                best_f = f
                best_x = x.copy()

        def wrapped_func(x):
            nonlocal eval_count
            if time_check():
                return 1e100
            eval_count += 1
            val = func(x)
            update_best(x, val)
            return val

        # Initialize with origin and a few random points
        rng = np.random.default_rng(123)
        init_points = [np.zeros(dim)]
        for _ in range(5):
            init_points.append(rng.uniform(-5, 5, dim))

        for p in init_points:
            if eval_count >= budget or time_check():
                break
            wrapped_func(p)

        # Differential Evolution parameters
        # For Rastrigin, a larger population and higher crossover rate help
        pop_size = 15
        F = 0.8  # Scale factor
        CR = 0.9  # Crossover rate

        # Initialize population
        # Use the best few points from init as part of the population
        # But we need to track their values. Let's just re-randomize for DE population
        # or use the init points.
        # Let's create a fresh population centered around the current best to exploit,
        # but keep some diversity.
        
        # Actually, a standard DE approach:
        # 1. Generate initial population
        # 2. Loop: Select 3 distinct, create trial, evaluate, select
        
        # We already have 6 points. Let's add more to form a population of size pop_size
        # We need to store the population members and their fitness.
        
        # Reset population to include our best knowns and randoms
        # But we don't have a list of (x, f) for all init points easily without re-evaluating or storing.
        # Let's just start a fresh DE population but seed it with the current best and some randoms.
        
        # To be efficient, let's just generate a new population of size pop_size
        # and evaluate them.
        
        pop = []
        # Add current best
        pop.append(best_x.copy())
        
        # Add random points
        while len(pop) < pop_size:
            # Sample around best with some spread, or globally?
            # Global is safer for multimodal.
            p = rng.uniform(-5, 5, dim)
            pop.append(p)
            
        # Evaluate new population members (except best which is already evaluated)
        # Note: best_x might have been evaluated in init.
        # We need a list of (x, f) for the population.
        # Let's just re-evaluate all to be safe and simple, or track.
        # Re-evaluating costs budget. 
        # Let's assume the init points were evaluated. 
        # We'll just generate a NEW population and evaluate it.
        # This wastes the init evals but ensures clean state for DE.
        # However, init evals helped find a good starting point.
        # Let's keep the best_x and generate the rest of the population randomly.
        
        # Population storage: list of [x, f]
        pop_data = []
        
        # First member is current best
        pop_data.append([best_x.copy(), best_f])
        
        for i in range(1, pop_size):
            if eval_count >= budget or time_check():
                break
            x = rng.uniform(-5, 5, dim)
            f = wrapped_func(x)
            pop_data.append([x, f])
            
        # Sort population by fitness
        pop_data.sort(key=lambda x: x[1])
        
        # DE Loop
        while eval_count < budget and not time_check():
            # Check if we have a full population
            if len(pop_data) < pop_size:
                break
                
            # Shuffle for diversity in selection? Standard DE selects 3 distinct from current pop.
            # Let's iterate through the population
            for i in range(len(pop_data)):
                if eval_count >= budget or time_check():
                    break
                    
                # Select 3 distinct individuals a, b, c different from i
                # Simple approach: pick random indices
                idxs = list(range(len(pop_data)))
                idxs.remove(i)
                a, b, c = rng.choice(idxs, 3, replace=False)
                
                x_a = pop_data[a][0]
                x_b = pop_data[b][0]
                x_c = pop_data[c][0]
                x_i = pop_data[i][0]
                
                # Create trial vector
                # x_trial = x_a + F * (x_b - x_c)
                # With binomial crossover
                trial = x_a + F * (x_b - x_c)
                
                # Crossover
                # For each dimension, with prob CR take from x_i, else from mutant
                # Ensure at least one dimension comes from mutant
                j_rand = rng.integers(0, dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        pass # Keep mutant value
                    else:
                        trial[j] = x_i[j]
                        
                # Clip to bounds
                trial = np.clip(trial, -5, 5)
                
                # Evaluate
                f_trial = wrapped_func(trial)
                
                # Selection
                if f_trial <= pop_data[i][1]:
                    pop_data[i] = [trial, f_trial]
                else:
                    # Keep original
                    pass
            
            # Re-sort population after a full pass?
            # Optional, but helps if we use ranking.
            # For standard DE, we don't strictly need to sort every generation,
            # but it helps if we want to restart or check stagnation.
            # Let's sort occasionally.
            if eval_count % (pop_size * 2) < pop_size: # Roughly every 2 gens
                 pop_data.sort(key=lambda x: x[1])
            
            # Stagnation check: if best_f hasn't improved in last X evals, restart?
            # Simple restart: if no improvement in 50 evals, perturb best and continue?
            # Let's just let DE run. It's robust.
            
            # If time is running out, break
            if time_check():
                 break

        return best_x, best_f
