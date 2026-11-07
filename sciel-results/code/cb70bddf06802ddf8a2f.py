import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin for 300s limit
        dim = self.dim
        budget = self.budget

        best_x = None
        best_f = float('inf')
        calls_made = 0

        def clip(x):
            return np.clip(x, -5, 5)

        def remaining_time():
            return deadline - time.time()

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            x = clip(np.asarray(x, dtype=float))
            fx = float(func(x))
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        rng = np.random.default_rng(42)

        # Island Model Setup
        n_islands = 5
        pop_size = 4  # 20 total individuals
        population = []
        
        # Phase 1: Initialization
        # Use diverse stratified sampling for the initial population
        for _ in range(pop_size * n_islands):
            if calls_made >= budget or remaining_time() < 10.0:
                break
            # Simple random initialization within the box
            x = rng.uniform(-5, 5, dim)
            fx = evaluate(x)
            population.append((fx, x.copy()))
        
        if not population:
             x0 = np.zeros(dim)
             evaluate(x0)
             population.append((best_f, best_x.copy()))

        # Split into islands
        # Sort by fitness and distribute
        population.sort(key=lambda t: t[0])
        islands = [[] for _ in range(n_islands)]
        for i, ind in enumerate(population):
            islands[i % n_islands].append(ind)

        # Phase 2: Evolutionary Loop
        # We will run generations until budget or time runs out.
        # Each generation:
        # 1. Local refinement (Hill Climbing / (1+1)-ES) on the best individual of each island
        # 2. Migration: Exchange best individuals between islands (random ring)
        # 3. Crossover: Create new individuals by crossing over the best of two different islands
        
        sigma = 1.0 # Global step size for local search
        max_local_steps = 10 # Max function calls per local refinement per island per generation
        
        while calls_made < budget and remaining_time() > 5.0:
            # 1. Local Refinement
            for i in range(n_islands):
                if calls_made >= budget or remaining_time() < 2.0:
                    break
                
                # Get the best individual in this island
                # Islands are lists of (fx, x)
                if not islands[i]:
                    continue
                
                # Sort island to keep best at front
                islands[i].sort(key=lambda t: t[0])
                best_fx, best_x_island = islands[i][0]
                
                # Perform (1+1)-ES on the best individual
                x = best_x_island.copy()
                fx = best_fx
                local_steps = 0
                
                while local_steps < max_local_steps and calls_made < budget and remaining_time() > 1.0:
                    y = x + sigma * rng.standard_normal(dim)
                    y = clip(y)
                    fy = evaluate(y)
                    local_steps += 1
                    
                    if fy < fx:
                        x = y
                        fx = fy
                        # Adapt sigma based on success (simple 1/5 rule or similar)
                        # For simplicity, we keep sigma constant per generation but could adapt
                        # Let's use a simple adaptation: if success, sigma *= 1.1, else 0.9
                        # But we need to be careful not to change sigma too much globally.
                        # Let's just update the island's best.
                    else:
                        # Failure, we don't move x, but we could adapt sigma locally?
                        # Let's keep it simple: fixed sigma per generation for now, 
                        # or adapt sigma globally based on average success rate?
                        # Let's try a small adaptation per step.
                        pass
                
                # Update the best individual in the island with the result of local search
                # Replace the worst individual in the island with the new best (or just update the best if improved)
                # Actually, (1+1)-ES replaces the parent only if child is better.
                # So x, fx is the best found.
                islands[i][0] = (fx, x.copy())
                
                # Optionally, we could insert x into the island if it's better than the worst, 
                # but since it started as the best and only improved, it remains the best.
                # The rest of the island population stays static unless we do more operations.
                # To maintain diversity, we should probably replace the worst individual in the island 
                # with a mutated version of the best, or just keep the best and discard the rest?
                # Standard island model: each island maintains a population.
                # Let's implement a simple replacement:
                # The local search improves the best. The other individuals in the island are "stagnant".
                # We will replace the worst individual in the island with a mutation of the best.
                if len(islands[i]) > 1:
                    # Find the index of the worst individual
                    # Since islands[i] is sorted, the worst is at the end
                    worst_idx = len(islands[i]) - 1
                    # Mutate the best
                    y = x + sigma * rng.standard_normal(dim)
                    y = clip(y)
                    fy = evaluate(y)
                    # Replace the worst
                    islands[i][worst_idx] = (fy, y.copy())
                    # Re-sort
                    islands[i].sort(key=lambda t: t[0])

            # 2. Migration
            # Exchange the best individual from each island with the best of the next island (ring topology)
            for i in range(n_islands):
                if calls_made >= budget or remaining_time() < 2.0:
                    break
                if not islands[i] or not islands[(i + 1) % n_islands]:
                    continue
                # Swap best individuals
                best_i = islands[i][0]
                best_j = islands[(i + 1) % n_islands][0]
                
                # To avoid duplicate evaluation, we just swap the references in our internal list
                # But wait, if we swap, we need to ensure we don't lose the best.
                # Standard migration: copy the best from i to j, replacing the worst in j.
                # Let's do that.
                # Insert best_i into island j
                islands[(i + 1) % n_islands].append(best_i)
                # Remove the worst from island j to keep size constant
                islands[(i + 1) % n_islands].sort(key=lambda t: t[0])
                islands[(i + 1) % n_islands] = islands[(i + 1) % n_islands][:pop_size]
                
            # 3. Crossover (The specific idea target)
            # Create a new individual by crossing over the best of two random distinct islands
            # This tests if recombination helps.
            if n_islands > 1:
                i1 = rng.integers(0, n_islands)
                i2 = rng.integers(0, n_islands)
                while i2 == i1:
                    i2 = rng.integers(0, n_islands)
                
                if islands[i1] and islands[i2]:
                    x1 = islands[i1][0][1]
                    x2 = islands[i2][0][1]
                    
                    # Uniform Crossover
                    mask = rng.random(dim) > 0.5
                    child = np.where(mask, x1, x2)
                    child = clip(child)
                    
                    # Evaluate the child
                    f_child = evaluate(child)
                    
                    # Insert the child into one of the islands (e.g., i1), replacing the worst
                    islands[i1].append((f_child, child.copy()))
                    islands[i1].sort(key=lambda t: t[0])
                    islands[i1] = islands[i1][:pop_size]
                    
                    # Also, if the child is the global best, update global best
                    # (evaluate already does this)

        return best_x, best_f
