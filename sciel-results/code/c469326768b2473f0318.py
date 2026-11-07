import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0 # 20s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Use a fixed seed for internal RNG if we want reproducibility in the algorithm logic, 
        # but the problem seed is set by the evaluator. 
        # The prompt says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # This affects np.random functions. 
        # Using np.random.default_rng(0) inside creates a separate stream, which is safer 
        # to ensure the algorithm behaves consistently regardless of external np.random state 
        # if the problem function uses np.random. 
        # However, if the problem function relies on the global seed, and we use a local RNG, 
        # we are independent. This is usually good.
        
        rng = np.random.default_rng(42)
        
        # Initial population
        pop_size = 10
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Store as list of (x, f) for easier manipulation
        pop = list(zip(X, F))
        
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 and f1 != f2
            return np.all(f1 <= f2) and np.any(f1 < f2)
            
        def is_non_dominated(f_new, pop_F):
            # Check if f_new is dominated by any in pop_F
            for f_old in pop_F:
                if dominates(f_old, f_new):
                    return False
            return True
            
        def crowding_distance(points):
            # points: list of (x, f)
            # returns list of distances
            if len(points) <= 2:
                return [float('inf')] * len(points)
            
            # Sort by first objective
            sorted_idx = np.argsort([p[1][0] for p in points])
            sorted_points = [points[i] for i in sorted_idx]
            
            cd = np.zeros(len(points))
            # Initialize boundaries
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            
            # Calculate for middle points
            for i in range(1, len(sorted_points) - 1):
                prev_f = sorted_points[i-1][1]
                curr_f = sorted_points[i][1]
                next_f = sorted_points[i+1][1]
                
                # Distance in f1
                d1 = next_f[0] - prev_f[0]
                # Distance in f2 (need to sort by f2 separately for this part? 
                # Standard crowding distance sums distances in each objective space independently)
                
                # To do it correctly, we should calculate distance in f2 space separately
                # and add them.
                pass # Placeholder, will implement full CD below
                
            # Full CD Implementation
            cd = np.zeros(len(points))
            for m in range(n_obj):
                # Sort by objective m
                idx = np.argsort([p[1][m] for p in points])
                for i in range(1, len(idx) - 1):
                    cd[idx[i]] += (points[idx[i+1]][1][m] - points[idx[i-1]][1][m])
                # Boundaries are inf
                cd[idx[0]] = float('inf')
                cd[idx[-1]] = float('inf')
                
            # If any objective is constant, CD might be 0 for all, which is bad.
            # If all points have same f[m], the term is 0.
            # We should normalize? The task says "exact hypervolume", but for pruning, 
            # raw distance is usually fine if scales are similar. 
            # RE21 objectives might have different scales.
            # Let's normalize by range to be safe.
            
            # Recalculate with normalization
            cd = np.zeros(len(points))
            for m in range(n_obj):
                vals = np.array([p[1][m] for p in points])
                range_m = vals.max() - vals.min()
                if range_m > 1e-10:
                    idx = np.argsort(vals)
                    for i in range(1, len(idx) - 1):
                        cd[idx[i]] += (vals[idx[i+1]] - vals[idx[i-1]]) / range_m
                    cd[idx[0]] = float('inf')
                    cd[idx[-1]] = float('inf')
                else:
                    # If all same, distance is 0, which is fine, 
                    # but if all objectives are same, CD is 0. 
                    # This shouldn't happen for non-dominated sets usually.
                    pass
                    
            return cd

        max_pop_size = 20
        
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Select 3 parents
            # Ensure we have at least 3 points
            if len(pop) < 3:
                # If we somehow lost points, add random
                x_new = lo + (hi - lo) * rng.random(dim)
                f_new = problem(x_new)
                evals += 1
                pop.append((x_new, f_new))
                continue
                
            i = rng.integers(len(pop))
            j = rng.integers(len(pop))
            k = rng.integers(len(pop))
            # Ensure distinct indices
            if i == j or i == k or j == k:
                continue
                
            x1 = pop[i][0]
            x2 = pop[j][0]
            x3 = pop[k][0]
            
            # DE Mutation
            F_factor = 0.8
            v = x1 + F_factor * (x2 - x3)
            v = np.clip(v, lo, hi)
            
            f_v = problem(v)
            evals += 1
            
            # Check if v is dominated
            dominated = False
            for _, f_old in pop:
                if dominates(f_old, f_v):
                    dominated = True
                    break
                    
            if not dominated:
                # Remove points dominated by v
                new_pop = []
                for x_old, f_old in pop:
                    if not dominates(f_v, f_old):
                        new_pop.append((x_old, f_old))
                new_pop.append((v, f_v))
                pop = new_pop
                
                # Prune if too large
                if len(pop) > max_pop_size:
                    cds = crowding_distance(pop)
                    # Find index with min CD
                    min_idx = np.argmin(cds)
                    pop.pop(min_idx)
        
        return pop
