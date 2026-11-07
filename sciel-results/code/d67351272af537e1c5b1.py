"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II-style multi-objective optimizer for DTLZ4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin before 330s limit
        rng = np.random.default_rng(0)
        lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
        
        # Population size: use a larger fraction of budget for better exploration
        # DTLZ4 with 12 vars, 3 objectives. Budget is 200.
        # Let's use a population size that allows multiple generations.
        # If pop_size = 50, we have 4 generations of 50 offspring (100 evals) + 50 init = 150, leaving 50 for extra?
        # Or just run until budget is exhausted.
        # Standard NSGA-II uses pop_size. Let's try pop_size = 50.
        pop_size = 50
        if pop_size > self.budget:
            pop_size = self.budget // 2  # Ensure we can at least do one generation if budget is small
            
        # Initialize population
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions for NSGA-II
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            front = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_count[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_count[j] += 1
                        
                if n_count[i] == 0:
                    front[0].append(i)
                    
            k = 0
            while len(front[k]) > 0:
                next_front = []
                for i in front[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    front.append(next_front)
                else:
                    break
            return front

        def crowding_distance(F, indices):
            if len(indices) == 0:
                return []
            if len(indices) == 1:
                return [float('inf')]
            
            n = len(indices)
            cd = np.zeros(n)
            m = F.shape[1]
            
            for k in range(m):
                obj = F[indices, k]
                idx_sorted = np.argsort(obj)
                sorted_indices = [indices[i] for i in idx_sorted]
                cd_sorted = np.zeros(n)
                
                min_val = obj[idx_sorted[0]]
                max_val = obj[idx_sorted[-1]]
                
                if max_val - min_val == 0:
                    cd_sorted[:] = 1.0
                else:
                    cd_sorted[0] = 1.0
                    cd_sorted[-1] = 1.0
                    for i in range(1, n - 1):
                        cd_sorted[i] = (obj[idx_sorted[i+1]] - obj[idx_sorted[i-1]]) / (max_val - min_val)
                
                # Map back to original indices
                for i, orig_idx in enumerate(sorted_indices):
                    pos_in_indices = np.where(indices == orig_idx)[0][0]
                    cd[pos_in_indices] += cd_sorted[i]
            
            return cd

        def tournament_select(F, indices, k=2):
            # Pick k random individuals from indices, return the best one
            chosen = rng.choice(indices, size=k, replace=False)
            # Find the non-dominated one among chosen, or use rank
            # For simplicity in tournament, we can just pick the one with better crowding distance
            # But we need ranks. Let's precompute ranks.
            # This helper is called inside the loop, so passing ranks is better.
            # Let's refactor to pass ranks and cd.
            pass

        # Main loop
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Combine parents and offspring? No, standard NSGA-II creates offspring from parents.
            # Current X is the parent population.
            
            # 1. Fast Non-Dominated Sort to get ranks
            fronts = fast_non_dominated_sort(F)
            
            # 2. Calculate Crowding Distance for each front
            cd_by_front = []
            for front in fronts:
                if not front:
                    cd_by_front.append([])
                    continue
                cd_vals = crowding_distance(F, front)
                cd_by_front.append(cd_vals)
            
            # Create a list of (rank, crowding, index) for selection
            # We need a way to compare individuals.
            # Rank 0 is best. Higher crowding is better.
            
            # Create offspring
            offspring_count = pop_size
            offspring_X = []
            offspring_F = []
            
            # We will use binary tournament selection
            # Precompute ranks for each individual
            ranks = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            # For crowding, we need to access it quickly.
            # Let's create a dictionary or array mapping index to cd.
            cd_arr = np.zeros(len(F))
            for i, front in enumerate(fronts):
                if not front:
                    continue
                cd_vals = cd_by_front[i]
                for j, idx in enumerate(front):
                    cd_arr[idx] = cd_vals[j]
            
            def tournament_select_idx():
                i1 = rng.integers(len(F))
                i2 = rng.integers(len(F))
                # Compare i1 and i2
                # 1. Lower rank is better
                if ranks[i1] != ranks[i2]:
                    return i1 if ranks[i1] < ranks[i2] else i2
                # 2. Higher crowding distance is better
                return i1 if cd_arr[i1] > cd_arr[i2] else i2
            
            for _ in range(offspring_count):
                if used + 1 > self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                p1 = tournament_select_idx()
                p2 = tournament_select_idx()
                
                # Crossover: Simulated Binary Crossover (SBX) or simple blend?
                # For DTLZ, variables are in [0,1].
                # Let's use a simple blend crossover + mutation.
                # Or just mutate one parent?
                # Standard NSGA-II uses SBX. Let's implement a simple version.
                # eta_c = 20 for crossover
                # eta_m = 20 for mutation
                
                # To keep it simple and robust, let's use a mix of:
                # 1. Copy one parent (if rng < 0.5)
                # 2. Blend crossover
                # 3. Mutation
                
                # Let's try a simple approach:
                # child = p1 + 0.1 * noise
                # This is basically what the previous code did, but with proper selection.
                
                # Let's try SBX-like crossover:
                # child = (1-alpha)*p1 + alpha*p2
                # where alpha is random?
                
                # Let's stick to a robust simple strategy:
                # Pick a parent, mutate it.
                # But use the tournament selection to pick the parent.
                
                parent_idx = p1 # Just use one parent for now? Or mix?
                # Let's do a simple blend:
                # child = 0.5 * p1 + 0.5 * p2 + noise
                # This might be too aggressive.
                
                # Let's use: child = p1 + sigma * noise
                # Adaptive sigma?
                # Let's use a fixed sigma = 0.1 * (hi - lo)
                
                child = X[p1].copy()
                # Add Gaussian noise
                sigma = 0.1
                child += sigma * (hi - lo) * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                offspring_X.append(child)
                offspring_F.append(fc)
                
                if len(offspring_X) == offspring_count:
                    break
            
            if not offspring_X:
                break
            
            # Combine parents and offspring
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, np.array(offspring_F)])
            
            # Non-dominated sort on combined population
            fronts_comb = fast_non_dominated_sort(F_comb)
            
            # Select next generation of size pop_size
            X_new = []
            F_new = []
            i = 0
            while len(X_new) < pop_size and i < len(fronts_comb):
                front = fronts_comb[i]
                if len(X_new) + len(front) <= pop_size:
                    # Add all
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    i += 1
                else:
                    # Need to select from this front based on crowding distance
                    need = pop_size - len(X_new)
                    cd_vals = crowding_distance(F_comb, front)
                    # Sort by cd descending
                    sorted_indices = np.argsort(cd_vals)[::-1]
                    for j in range(need):
                        idx = front[sorted_indices[j]]
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
            
        return X
# EVOLVE-BLOCK-END
