import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        n = self.dim
        budget = self.budget
        
        rng = np.random.default_rng(42)
        
        best_x = np.zeros(n)
        best_f = np.inf
        calls_made = 0

        def evaluate(x):
            nonlocal calls_made, best_x, best_f
            if calls_made >= budget:
                return np.inf
            x = np.clip(x, -5.0, 5.0)
            fx = func(x)
            calls_made += 1
            if fx < best_f:
                best_f = fx
                best_x = x.copy()
            return fx

        # Phase 1: Coarse Global Scan (Halton-like systematic scan)
        # The Step Ellipsoidal function has a flat plateau. 
        # A systematic scan ensures we cross the discontinuity early.
        scan_budget = int(budget * 0.15)  # 75 calls
        
        # Use a stratified sampling approach for better coverage
        # We'll create a grid and sample from it
        grid_points = []
        
        # Create a stratified grid
        # For 5D, we can use a 3x3x3x3x3 grid = 243 points, but we only have ~75
        # Let's use a more efficient sampling strategy
        # Use Latin Hypercube Sampling approximation
        
        # Actually, let's use a simpler approach: systematic scan with jitter
        n_per_dim = 3  # 3^5 = 243, but we'll sample a subset
        
        # Generate all grid points
        grid_indices = np.meshgrid(*[np.linspace(-4.5, 4.5, n_per_dim) for _ in range(n)])
        grid_points = np.array([np.concatenate([g.ravel() for g in grid_indices], axis=1).T])
        
        # Shuffle and take first scan_budget points
        shuffle_idx = rng.permutation(len(grid_points))
        scan_points = grid_points[shuffle_idx[:scan_budget]]
        
        for p in scan_points:
            if calls_made >= scan_budget:
                break
            if (time.time() - start_time) > deadline:
                break
            evaluate(p)
        
        # Phase 2: Differential Evolution (DE) for population-based search
        # DE is more robust for flat landscapes than (1+lambda)-ES
        if calls_made < budget and (time.time() - start_time) < deadline:
            pop_size = 10
            F = 0.8  # Mutation scale factor
            CR = 0.9  # Crossover rate
            
            # Initialize population around the best point found
            center = best_x.copy()
            
            # Create initial population
            population = [center.copy()]
            fitness = [best_f]
            
            for i in range(pop_size - 1):
                if calls_made >= budget:
                    break
                if (time.time() - start_time) > deadline:
                    break
                # Sample around center with increasing radius
                radius = 0.5 + i * 0.3
                x_trial = center + radius * rng.normal(size=n)
                x_trial = np.clip(x_trial, -5.0, 5.0)
                f_trial = evaluate(x_trial)
                population.append(x_trial)
                fitness.append(f_trial)
            
            # Run DE for remaining budget
            generation = 0
            while calls_made < budget and (time.time() - start_time) < deadline:
                generation += 1
                
                # For each individual, create a trial vector
                new_population = []
                new_fitness = []
                
                for i in range(pop_size):
                    if calls_made >= budget:
                        break
                    if (time.time() - start_time) > deadline:
                        break
                    
                    # Select two random distinct indices different from i
                    r1, r2 = np.random.choice([j for j in range(pop_size) if j != i], 2, replace=False)
                    
                    # Mutation: v = x_r1 + F * (x_r2 - x_i)
                    v = population[r1] + F * (population[r2] - population[i])
                    
                    # Crossover: trial vector
                    trial = np.zeros(n)
                    j_rand = rng.integers(n)
                    for j in range(n):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = population[i][j]
                    
                    trial = np.clip(trial, -5.0, 5.0)
                    f_trial = evaluate(trial)
                    
                    # Selection
                    if f_trial <= fitness[i]:
                        new_population.append(trial)
                        new_fitness.append(f_trial)
                    else:
                        new_population.append(population[i])
                        new_fitness.append(fitness[i])
                
                # Update population
                if new_population:
                    population = new_population
                    fitness = new_fitness
                    
                    # Update global best
                    idx_best = np.argmin(fitness)
                    if fitness[idx_best] < best_f:
                        best_f = fitness[idx_best]
                        best_x = population[idx_best].copy()
                    
                    # Adaptive F: reduce F if no improvement for several generations
                    if generation % 10 == 0:
                        F = max(0.3, F * 0.9)
        
        # Phase 3: Local Refinement (Coordinate Descent)
        if calls_made < budget and (time.time() - start_time) < deadline:
            step_sizes = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
            
            current_x = best_x.copy()
            current_f = best_f
            
            for step in step_sizes:
                if calls_made >= budget:
                    break
                if (time.time() - start_time) > deadline:
                    break
                    
                improved = True
                max_iter = 3
                iter_count = 0
                
                while improved and calls_made < budget and iter_count < max_iter:
                    improved = False
                    iter_count += 1
                    for d_idx in range(n):
                        if calls_made >= budget:
                            break
                        if (time.time() - start_time) > deadline:
                            break
                            
                        # Try positive direction
                        x_pos = current_x.copy()
                        x_pos[d_idx] += step
                        x_pos = np.clip(x_pos, -5, 5)
                        f_pos = evaluate(x_pos)
                        
                        if f_pos < current_f:
                            current_x = x_pos
                            current_f = f_pos
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                            improved = True
                            continue
                            
                        # Try negative direction
                        x_neg = current_x.copy()
                        x_neg[d_idx] -= step
                        x_neg = np.clip(x_neg, -5, 5)
                        f_neg = evaluate(x_neg)
                        
                        if f_neg < current_f:
                            current_x = x_neg
                            current_f = f_neg
                            if current_f < best_f:
                                best_f = current_f
                                best_x = current_x.copy()
                            improved = True
                            continue

        return best_x, best_f
