"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for WFG7 using MOEA/D-DE strategy."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Parameters for MOEA/D-DE
        # Budget is 200. We need to be careful with overhead.
        # A population size of 20-30 is reasonable for 200 evals.
        # Let's use pop_size = 20. 
        # Initial 20 evals. Remaining 180.
        # Each generation evaluates pop_size children.
        # 180 / 20 = 9 generations. This is very few.
        # Maybe smaller pop size? Or larger?
        # If pop_size = 10, 190/10 = 19 generations.
        # If pop_size = 20, 180/20 = 9 generations.
        # MOEA/D usually benefits from larger populations, but we are budget constrained.
        # Let's try pop_size = 20 for better coverage initially.
        
        pop_size = 20
        self.pop_size = pop_size
        
        # 1. Initialization
        # Random uniform initialization
        X = lo + range_ * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Store all evaluated points for hypervolume calculation
        all_points = list(zip(X, F))
        
        # 2. Weight vectors for Tchebycheff decomposition
        # Generate weight vectors uniformly on the simplex
        # For 3 objectives, we can generate points on a grid or randomly.
        # Since we want coverage, random weights are often good, 
        # but structured weights ensure better distribution.
        # Let's use a simple grid for 3 objectives.
        # Number of weights should match pop_size.
        # We can generate random weights and normalize them.
        raw_weights = np.random.random((pop_size, self.n_obj))
        W = raw_weights / np.sum(raw_weights, axis=1, keepdims=True)
        
        # Ideal point (min values so far)
        ideal_point = np.min(F, axis=0)
        
        # 3. Evolutionary Loop
        # We will run until budget is exhausted or time runs out.
        # Each "generation" we generate pop_size children.
        # To be efficient, we can generate children in batches.
        
        # Adaptive parameters
        # CR (crossover rate) and F (mutation factor)
        # Standard DE: F ~ 0.5-0.8, CR ~ 0.5
        # For multi-objective, sometimes lower CR is better for diversity?
        # Let's start with F=0.5, CR=0.5 and see.
        
        # Neighborhood structure: 
        # In MOEA/D, each subproblem has a neighborhood.
        # With pop_size=20, we can define neighbors by index distance.
        # Let's assume a small neighborhood, e.g., 2 neighbors on each side.
        T = 4  # Neighborhood size
        
        # Pre-calculate neighbor indices
        neighbors = []
        for i in range(pop_size):
            # Simple circular neighborhood
            nbs = []
            for offset in range(-T//2, T//2 + 1):
                if offset == 0:
                    continue
                nbs.append((i + offset) % pop_size)
            neighbors.append(nbs)
        
        # Main Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Determine how many children we can evaluate in this batch
            # We want to do a full generation if possible, 
            # but if we have few evals left, we do partial.
            batch_size = min(pop_size, self.budget - used)
            
            # If batch_size is small, we might just do random search or 
            # mutate the best individuals.
            # But let's stick to the DE loop structure.
            
            # For each individual in the batch, generate a child
            # We process them one by one to respect the budget strictly.
            
            for i in range(batch_size):
                if used >= self.budget:
                    break
                if time.time() > deadline:
                    break
                
                # Select parents for DE
                # Standard DE: x_new = x_r1 + F * (x_r2 - x_r3)
                # In MOEA/D, we often select from the neighborhood.
                
                # Pick r1, r2, r3 from neighborhood of i
                # Ensure they are distinct
                nbs = neighbors[i]
                # Add i itself? Usually DE uses distinct individuals.
                # Let's pick 3 distinct from nbs + [i]
                candidates = list(set(nbs + [i]))
                # If not enough candidates (shouldn't happen with T=4, pop=20), pad
                while len(candidates) < 3:
                    candidates.append(np.random.randint(pop_size))
                
                r1, r2, r3 = np.random.choice(candidates, 3, replace=False)
                
                # Mutation
                # F is mutation factor. Let's use a fixed F=0.5 initially.
                # Or adaptive? 
                # Let's try F = 0.5
                F_val = 0.5
                v = X[r1] + F_val * (X[r2] - X[r3])
                
                # Crossover
                # Binomial crossover
                CR = 0.5
                u = np.random.random(self.dim)
                j_rand = np.random.randint(self.dim)
                
                child = np.empty(self.dim)
                for j in range(self.dim):
                    if u[j] <= CR or j == j_rand:
                        child[j] = v[j]
                    else:
                        child[j] = X[i][j]
                
                # Clip to bounds
                child = np.clip(child, lo, hi)
                
                # Evaluate
                fc = problem(child)
                used += 1
                all_points.append((child, fc))
                
                # Update population using Tchebycheff fitness
                # Fitness = max_k ( w_k * |f_k - z_k| )
                # z_k is the ideal point (best value for objective k)
                
                # Update ideal point
                ideal_point = np.minimum(ideal_point, fc)
                
                # Calculate fitness for child and current individual i
                # Note: We need to calculate fitness for all individuals if ideal point changes?
                # In MOEA/D, we usually keep a fitness matrix or recalculate.
                # For efficiency, let's recalculate fitness for i and child.
                # But wait, the decision of which individual to replace depends on comparing
                # child's fitness vs i's fitness.
                
                # Calculate Tchebycheff fitness for child
                # f_child = max_k ( W[i,k] * | fc[k] - ideal_point[k] | )
                # f_i = max_k ( W[i,k] * | F[i,k] - ideal_point[k] | )
                
                # Note: If ideal_point changed, F[i] fitness also changes.
                # This is a dynamic fitness.
                
                # Let's compute:
                diff_child = np.abs(fc - ideal_point) * W[i]
                fit_child = np.max(diff_child)
                
                diff_i = np.abs(F[i] - ideal_point) * W[i]
                fit_i = np.max(diff_i)
                
                # Replace if child is better (lower fitness)
                if fit_child <= fit_i:
                    X[i] = child
                    F[i] = fc
        
        return X
# EVOLVE-BLOCK-END
