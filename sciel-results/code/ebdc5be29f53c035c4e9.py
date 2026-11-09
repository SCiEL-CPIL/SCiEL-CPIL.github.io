"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective evolutionary optimizer with hypervolume-aware selection."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use fixed seed for reproducibility within the run, but allow for variation if needed
        # However, the task says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)"
        # So we should probably use the global numpy random state or a local one seeded by it.
        # The previous code used np.random.default_rng(0). Let's stick to that or use np.random directly.
        # Since the evaluator seeds np.random, using np.random functions will inherit that seed.
        # Using default_rng(0) ignores the evaluator's seed. The task says "For seeds 0 and 1...".
        # This implies the results should differ or depend on the seed.
        # Let's use np.random directly to respect the evaluator's seeding.
        
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [-1] * n
            front = [0]
            
            for i in range(n):
                for j in range(n):
                    if i != j:
                        # i dominates j?
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            S[i].append(j)
                            n_count[i] += 1
                        elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            n_count[j] += 1
            
            current_front = []
            for i in range(n):
                if n_count[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
                    
            k = 0
            while len(current_front) > 0:
                front.append(k)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return rank, front

        def crowding_distance(F, indices):
            if len(indices) <= 1:
                return {i: float('inf') for i in indices}
            
            distances = {i: 0.0 for i in indices}
            for m in range(n_obj):
                vals = [F[i][m] for i in indices]
                min_val = min(vals)
                max_val = max(vals)
                if max_val == min_val:
                    continue
                # Sort indices by objective m
                sorted_indices = sorted(indices, key=lambda i: F[i][m])
                distances[sorted_indices[0]] = float('inf')
                distances[sorted_indices[-1]] = float('inf')
                for k in range(1, len(sorted_indices) - 1):
                    i = sorted_indices[k]
                    distances[i] += (F[sorted_indices[k+1]][m] - F[sorted_indices[k-1]][m]) / (max_val - min_val)
            return distances

        # Initial population
        pop_size = min(50, budget)
        if pop_size < 2:
            pop_size = 2
            
        X = lo + (hi - lo) * np.random.random((pop_size, dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II loop
        while used < budget:
            if time.time() > deadline - 5.0:
                break
                
            # Combine parents and offspring? 
            # Standard NSGA-II: 
            # 1. Evaluate offspring
            # 2. Combine P + Q
            # 3. Non-dominated sort
            # 4. Select next P from combined using ranks and crowding
            
            # For simplicity and to fit budget, let's do a simpler approach:
            # Select parents from current non-dominated front
            # Create offspring
            # Replace worst members of current population
            
            # Non-dominated sort current F
            rank, front = non_dominated_sort(F)
            
            # Select parents from first front
            if front[0] > 0:
                parents_indices = front[0]
            else:
                parents_indices = list(range(len(F)))
                
            # Tournament selection
            def tournament_select(pop_X, pop_F, pop_rank, k=2):
                candidates = [np.random.randint(len(pop_X)) for _ in range(k)]
                best = min(candidates, key=lambda i: pop_rank[i])
                return pop_X[best]
                
            # Create offspring
            n_offspring = min(pop_size, budget - used)
            if n_offspring <= 0:
                break
                
            offspring_X = []
            for _ in range(n_offspring):
                p1 = tournament_select(X, F, rank)
                p2 = tournament_select(X, F, rank)
                
                # SBX Crossover (Simplified: uniform crossover or blend)
                # Let's use uniform crossover
                mask = np.random.random(dim) < 0.5
                child = np.where(mask, p1, p2)
                
                # Mutation
                if np.random.random() < 0.5:
                    # Gaussian mutation
                    sigma = 0.1
                    child = child + sigma * (hi - lo) * np.random.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                else:
                    # Polynomial mutation or just keep as is
                    pass
                    
                offspring_X.append(child)
                
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += n_offspring
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated sort combined
            rank_comb, front_comb = non_dominated_sort(F_comb)
            
            # Select next population
            next_X = []
            next_F = []
            i = 0
            while len(next_X) < pop_size:
                if i >= len(front_comb):
                    break
                front_idx = front_comb[i]
                if len(next_X) + front_idx <= pop_size:
                    # Take all from this front
                    for idx in range(front_idx):
                        if len(next_X) < pop_size:
                            # Find the index in the original array corresponding to this rank
                            # This is tricky because non_dominated_sort returns ranks, not indices
                            # We need to map rank back to indices
                            pass
                i += 1
                
            # The above selection logic is flawed because non_dominated_sort doesn't return indices per front.
            # Let's rewrite non_dominated_sort to return fronts as lists of indices.

            # Let's restart the selection part properly.
            
            # Re-implement non_dominated_sort to return fronts as lists of indices
            def non_dominated_sort_indices(F):
                n = len(F)
                S = [[] for _ in range(n)]
                n_count = [0] * n
                rank = [-1] * n
                fronts = []
                
                for i in range(n):
                    for j in range(n):
                        if i != j:
                            if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                                S[i].append(j)
                                n_count[i] += 1
                            elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                                n_count[j] += 1
                                
                current_front = []
                for i in range(n):
                    if n_count[i] == 0:
                        rank[i] = 0
                        current_front.append(i)
                        
                k = 0
                while len(current_front) > 0:
                    fronts.append(current_front.copy())
                    next_front = []
                    for i in current_front:
                        for j in S[i]:
                            n_count[j] -= 1
                            if n_count[j] == 0:
                                rank[j] = k + 1
                                next_front.append(j)
                    current_front = next_front
                    k += 1
                return rank, fronts

            # Recalculate for combined population
            rank_comb, fronts_comb = non_dominated_sort_indices(F_comb)
            
            next_X = []
            next_F = []
            for front in fronts_comb:
                if len(next_X) + len(front) <= pop_size:
                    for idx in front:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need to select from this front using crowding distance
                    cd = crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    for idx in sorted_front:
                        if len(next_X) < pop_size:
                            next_X.append(X_comb[idx])
                            next_F.append(F_comb[idx])
                    break
                    
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
# EVOLVE-BLOCK-END
