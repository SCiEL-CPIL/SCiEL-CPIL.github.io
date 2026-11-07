import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Leave 10s margin for safety
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        pop_size = min(100, max(20, self.budget // 2))
        
        # Initialize population
        X = lo + range_ * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # DE Parameters
        F_factor = 0.8
        CR = 0.9
        
        # Helper to check dominance
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def is_non_dominated(F_set):
            nd = []
            for i in range(len(F_set)):
                dominated = False
                for j in range(len(F_set)):
                    if i != j and dominates(F_set[j], F_set[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return np.array(nd)
        
        # Main DE Loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Get non-dominated set
            nd_indices = is_non_dominated(F)
            if len(nd_indices) == 0:
                nd_indices = [np.argmin(np.sum(F, axis=1))]
            
            # Select parents from non-dominated set
            p1_idx = nd_indices[np.random.randint(len(nd_indices))]
            p2_idx = nd_indices[np.random.randint(len(nd_indices))]
            p3_idx = nd_indices[np.random.randint(len(nd_indices))]
            
            x1 = X[p1_idx]
            x2 = X[p2_idx]
            x3 = X[p3_idx]
            
            # Mutation
            v = x1 + F_factor * (x2 - x3)
            
            # Crossover
            j_rand = np.random.randint(self.dim)
            trial = np.empty_like(x1)
            for j in range(self.dim):
                if np.random.rand() < CR or j == j_rand:
                    trial[j] = v[j]
                else:
                    trial[j] = x1[j]
            
            # Clip
            trial = np.clip(trial, lo, hi)
            
            # Evaluate
            f_trial = problem(trial)
            used += 1
            
            # Selection: Replace worst dominated or use tournament
            # Simple approach: If trial dominates p1, replace p1.
            # Or replace the worst individual in the population.
            # To maintain diversity, we should replace a dominated individual.
            
            # Find an individual to replace
            # Prefer replacing a dominated individual that is worse than trial
            replaced = False
            # Sort population by dominance rank (approximate by sum)
            # For efficiency, just try to replace p1 if trial is better
            if dominates(f_trial, F[p1_idx]):
                X[p1_idx] = trial
                F[p1_idx] = f_trial
                replaced = True
            
            if not replaced:
                # Find worst individual (max sum of objectives)
                worst_idx = np.argmax(np.sum(F, axis=1))
                # Replace if trial is better than worst
                if np.sum(f_trial) < np.sum(F[worst_idx]):
                    X[worst_idx] = trial
                    F[worst_idx] = f_trial

        # Local Search Phase
        # Use remaining budget to polish the best solution
        if used < self.budget:
            # Find best solution (lowest sum)
            best_idx = np.argmin(np.sum(F, axis=1))
            best_x = X[best_idx].copy()
            best_f = F[best_idx].copy()
            
            # Coordinate descent / local random search
            step = 0.1
            while used < self.budget and step > 0.001:
                if time.time() > deadline:
                    break
                
                improved = False
                for dim_idx in range(self.dim):
                    if used >= self.budget:
                        break
                    
                    # Try +step
                    trial = best_x.copy()
                    trial[dim_idx] += step
                    trial = np.clip(trial, lo, hi)
                    f_trial = problem(trial)
                    used += 1
                    
                    if np.sum(f_trial) < np.sum(best_f):
                        best_x = trial
                        best_f = f_trial
                        improved = True
                    
                    # Try -step
                    if used < self.budget:
                        trial = best_x.copy()
                        trial[dim_idx] -= step
                        trial = np.clip(trial, lo, hi)
                        f_trial = problem(trial)
                        used += 1
                        
                        if np.sum(f_trial) < np.sum(best_f):
                            best_x = trial
                            best_f = f_trial
                            improved = True
                
                if improved:
                    step *= 0.5
                else:
                    step *= 0.5
                    # If stuck, try a random jump
                    if step > 0.01 and used < self.budget:
                        trial = best_x + 0.5 * step * np.random.randn(self.dim)
                        trial = np.clip(trial, lo, hi)
                        f_trial = problem(trial)
                        used += 1
                        if np.sum(f_trial) < np.sum(best_f):
                            best_x = trial
                            best_f = f_trial
                            improved = True

            # Update population with best found
            X[best_idx] = best_x
            F[best_idx] = best_f

        return X
