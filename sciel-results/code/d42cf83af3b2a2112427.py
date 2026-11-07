import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        max_time = 280  # Safety margin against 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        all_points = []
        
        # Strategy: Multi-Objective Differential Evolution (MOEA/D style or simple DE/rand/1)
        # DE is known for good exploration in low dimensions.
        # We will maintain an archive of non-dominated solutions.
        # To ensure coverage, we use a simple approach:
        # 1. Initialize population randomly.
        # 2. Iteratively generate new points using DE mutation.
        # 3. Update archive if the new point is non-dominated and improves the hypervolume (or simply if it's non-dominated).
        #    Actually, to maximize hypervolume, we should keep points that contribute to the front.
        #    A simpler heuristic: Keep the non-dominated set.
        #    But with a fixed budget, we need to be efficient.
        #    
        # Let's try a specific MOEA/D-like decomposition or just a robust DE with an archive.
        # Population size: 20.
        # Budget: 200.
        # Init: 20 evals.
        # Remaining: 180.
        # 180 / 20 = 9 generations.
        
        pop_size = 20
        max_iter = (budget - pop_size) // pop_size
        
        # Initialize population
        X = lo + np.random.random((pop_size, dim)) * (hi - lo)
        F = np.zeros((pop_size, n_obj))
        
        # Archive of non-dominated points
        # We will store the points in a list and check dominance.
        # To be efficient, we can just keep the list and prune it.
        # Since dim=4, n_obj=3, checking dominance is O(N) per point, where N is archive size.
        # Archive size will likely be small (Pareto front of 4D 3-obj problem is a curve/surface).
        
        archive = []
        
        def is_dominated(a, b):
            # Returns True if a dominates b
            # a <= b in all objectives and a < b in at least one
            return np.all(a <= b) and np.any(a < b)

        def update_archive(f_new, x_new):
            nonlocal archive
            # Check if new point dominates any in archive
            dominated_indices = []
            for i, f_old in enumerate(archive):
                if is_dominated(f_new, f_old):
                    dominated_indices.append(i)
            
            if dominated_indices:
                # Remove dominated points
                for idx in reversed(dominated_indices):
                    archive.pop(idx)
            
            # Check if any point in archive dominates the new point
            for f_old in archive:
                if is_dominated(f_old, f_new):
                    return # New point is dominated, don't add
            
            # Add new point
            archive.append(f_new)
        
        # Evaluate initial population
        for i in range(pop_size):
            if used >= budget or time.time() - start_time > max_time:
                break
            f = problem(X[i])
            F[i] = f
            all_points.append(f)
            used += 1
            update_archive(f, X[i])
            
        # DE Parameters
        F_factor = 0.8  # Scaling factor
        CR = 0.9         # Crossover rate
        
        for it in range(max_iter):
            if used >= budget or time.time() - start_time > max_time:
                break
            
            # Shuffle population indices for DE
            perm = np.random.permutation(pop_size)
            
            for i in range(pop_size):
                if used >= budget or time.time() - start_time > max_time:
                    break
                
                # Pick 3 distinct random indices
                # Ensure they are distinct from each other and ideally from i
                # Simple way: pick 3 random indices
                p1 = perm[i]
                p2 = perm[(i + 1) % pop_size]
                p3 = perm[(i + 2) % pop_size]
                
                # If any are the same as i or each other, re-pick or just use them (DE is robust)
                # Let's just pick 3 random indices distinct from each other
                while len(set([p1, p2, p3])) < 3:
                    p1 = np.random.randint(pop_size)
                    p2 = np.random.randint(pop_size)
                    p3 = np.random.randint(pop_size)
                
                # Mutation
                v = X[p1] + F_factor * (X[p2] - X[p3])
                
                # Crossover
                # Binary crossover
                j_rand = np.random.randint(dim)
                u = np.zeros(dim)
                for j in range(dim):
                    if np.random.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = X[i][j]
                
                # Clip to bounds
                u = np.clip(u, lo, hi)
                
                # Evaluate
                f_u = problem(u)
                all_points.append(f_u)
                used += 1
                
                # Update archive
                update_archive(f_u, u)
                
                # Update population
                # Simple selection: if u is non-dominated by X[i] (or better in some sense), replace?
                # For multi-objective, we usually keep the non-dominated set.
                # Here, we can replace X[i] with u if u is not dominated by X[i].
                # This helps exploration.
                if not is_dominated(F[i], f_u):
                    X[i] = u
                    F[i] = f_u
                else:
                    # If X[i] dominates u, keep X[i]. 
                    # If they are non-dominated, we might keep both? 
                    # But population size is fixed. 
                    # Let's keep the one that is "better" in a scalarized sense?
                    # Or just keep X[i] to maintain diversity?
                    # Actually, if they are non-dominated, replacing X[i] with u is fine as long as we don't lose the best.
                    # Since we have an archive, we can afford to be more aggressive in exploration.
                    # Let's replace if u is not dominated by X[i].
                    pass

        return all_points
