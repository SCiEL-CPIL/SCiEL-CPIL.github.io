"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Island Model Evolutionary Strategy with Crossover for BBOB Sphere."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        dim = self.dim
        budget = self.budget
        deadline = time.time() + 280  # Safety margin before 300s limit
        
        # Fixed seed for reproducibility
        rng = np.random.default_rng(42)
        
        # Island Model Parameters
        n_islands = 4
        pop_size_per_island = 5
        total_pop = n_islands * pop_size_per_island
        
        # Initialize islands with diverse starting points
        # Spread them across the box to cover different regions
        islands = []
        for i in range(n_islands):
            # Create a diverse set of initial points
            # Use a grid-like distribution plus some randomness
            base_points = [
                np.zeros(dim),
                np.ones(dim) * 2,
                -np.ones(dim) * 2,
                np.array([3, -3, 3, -3, 3])[:dim],
                np.array([-3, 3, -3, 3, -3])[:dim],
                np.array([2, 2, -2, -2, 2])[:dim],
                np.array([-2, -2, 2, 2, -2])[:dim],
                np.array([1, -1, 1, -1, 1])[:dim],
            ]
            # Select a subset for this island
            start_idx = i % len(base_points)
            island_pop = []
            for j in range(pop_size_per_island):
                if j == 0:
                    x = base_points[(start_idx + j) % len(base_points)].copy()
                else:
                    # Add noise to the base point
                    x = base_points[(start_idx + j) % len(base_points)].copy() + rng.standard_normal(dim) * 1.0
                x = np.clip(x, -5, 5)
                fx = func(x)
                island_pop.append((x, fx))
            # Sort by fitness
            island_pop.sort(key=lambda p: p[1])
            islands.append(island_pop)
            
        # Track global best
        all_points = [p for island in islands for p in island]
        all_points.sort(key=lambda p: p[1])
        best_x, best_f = all_points[0]
        
        calls = len(all_points)
        
        # Adaptive sigma
        sigma = 1.0
        
        # Crossover probability
        p_cross = 0.5
        
        # Migration interval
        migrate_every = 20
        migration_rate = 0.2  # Fraction of population to migrate
        
        generation = 0
        
        while calls < budget and time.time() < deadline:
            generation += 1
            
            # Update sigma based on success rate (1/5 rule)
            # Count successes across all islands in last generation
            # For simplicity, use a global success counter
            # In a more robust implementation, we'd track per-island
            
            # Evaluate offspring for each island
            for i in range(n_islands):
                if calls >= budget or time.time() > deadline:
                    break
                
                island = islands[i]
                new_island = []
                
                # Elitism: keep the best individual
                new_island.append(island[0])
                
                # Generate offspring
                for j in range(pop_size_per_island - 1):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Parent selection: tournament of size 2
                    p1_idx = rng.integers(0, len(island))
                    p2_idx = rng.integers(0, len(island))
                    if island[p1_idx][1] <= island[p2_idx][1]:
                        p1 = island[p1_idx][0]
                    else:
                        p1 = island[p2_idx][0]
                    
                    # Crossover
                    if rng.random() < p_cross:
                        # Select second parent
                        p2_idx = rng.integers(0, len(island))
                        p2 = island[p2_idx][0]
                        
                        # Uniform crossover
                        mask = rng.random(dim) < 0.5
                        child = np.where(mask, p1, p2)
                    else:
                        child = p1.copy()
                    
                    # Mutation
                    child = child + rng.standard_normal(dim) * sigma
                    child = np.clip(child, -5, 5)
                    
                    # Evaluate
                    fx = func(child)
                    calls += 1
                    new_island.append((child, fx))
                
                # Sort new island by fitness
                new_island.sort(key=lambda p: p[1])
                islands[i] = new_island[:pop_size_per_island]
            
            # Migration
            if generation % migrate_every == 0:
                for i in range(n_islands):
                    if calls >= budget or time.time() > deadline:
                        break
                    
                    # Select best from a random other island
                    other_islands = [j for j in range(n_islands) if j != i]
                    other_idx = other_islands[rng.integers(0, len(other_islands))]
                    donor = islands[other_idx][0]  # Best from other island
                    
                    # Replace worst individual in current island with donor (with mutation)
                    migrant = donor + rng.standard_normal(dim) * sigma * 0.5
                    migrant = np.clip(migrant, -5, 5)
                    fx = func(migrant)
                    calls += 1
                    
                    # Insert migrant into island
                    islands[i].append((migrant, fx))
                    islands[i].sort(key=lambda p: p[1])
                    islands[i] = islands[i][:pop_size_per_island]
            
            # Update global best
            for island in islands:
                for x, fx in island:
                    if fx < best_f:
                        best_f = fx
                        best_x = x.copy()
            
            # Adapt sigma
            # Simple adaptation: increase if many improvements, decrease if few
            # Count improvements in this generation
            improvements = 0
            for i in range(n_islands):
                for j in range(1, len(islands[i])):
                    if islands[i][j][1] < islands[i][0][1]:
                        improvements += 1
            
            # 1/5 success rule approximation
            if improvements > total_pop / 5:
                sigma *= 1.2
            elif improvements < total_pop / 10:
                sigma *= 0.8
            
            # Keep sigma within reasonable bounds
            sigma = np.clip(sigma, 0.01, 5.0)
        
        return best_x, best_f
# EVOLVE-BLOCK-END
