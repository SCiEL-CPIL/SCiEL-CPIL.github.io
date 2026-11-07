# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Helper: Check if a dominates b (minimization)
        def dominates(a, b):
            return np.all(a <= b) and np.any(a < b)

        # Helper: Fast Non-Dominated Sort (simplified for small N)
        # Returns list of ranks (indices of points in each front)
        def fast_non_dominated_sort(F):
            n = len(F)
            if n == 0:
                return []
            S = [[] for _ in range(n)]  # S[i] = set of points dominated by i
            n_count = np.zeros(n, dtype=int) # n[i] = number of points dominating i
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        S[i].append(j)
                        n_count[j] += 1
                    elif dominates(F[j], F[i]):
                        S[j].append(i)
                        n_count[i] += 1
            
            front = []
            current_front = [i for i in range(n) if n_count[i] == 0]
            for i in current_front:
                rank[i] = 0
            front.append(current_front)
            
            k = 0
            while front[k]:
                next_front = []
                for i in front[k]:
                    for j in S[i]:
                        n_count[j] -= 1
                        if n_count[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if not next_front:
                    break
                front.append(next_front)
                k += 1
            return front

        # Helper: Crowding Distance
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return np.ones(len(front_indices)) * np.inf
            
            cd = np.zeros(len(front_indices))
            n_obj = self.n_obj
            
            for m in range(n_obj):
                # Sort front by m-th objective
                sorted_indices = sorted(front_indices, key=lambda idx: F[idx, m])
                cd[front_indices.index(sorted_indices[0])] = np.inf
                cd[front_indices.index(sorted_indices[-1])] = np.inf
                
                f_min = F[sorted_indices[0], m]
                f_max = F[sorted_indices[-1], m]
                range_m = f_max - f_min
                if range_m == 0:
                    continue
                    
                for i in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i-1]
                    next_idx = sorted_indices[i+1]
                    cd[front_indices.index(idx)] += (F[next_idx, m] - F[prev_idx, m]) / range_m
            
            return cd

        # Helper: Tournament Selection
        def tournament_select(population, ranks, cds, size=2):
            # population: list of indices
            # ranks: array of rank for each index
            # cds: dict mapping index to crowding distance
            best = None
            for _ in range(size):
                cand = np.random.choice(population)
                if best is None:
                    best = cand
                else:
                    if ranks[cand] < ranks[best]:
                        best = cand
                    elif ranks[cand] == ranks[best]:
                        if cds.get(cand, 0) > cds.get(best, 0):
                            best = cand
            return best

        # Phase 1: Initialization with LHS
        pop_size = 50
        if budget < pop_size:
            pop_size = budget
            
        X = np.zeros((pop_size, dim))
        for i in range(dim):
            perm = np.random.permutation(pop_size)
            for k in range(pop_size):
                start = perm[k] / pop_size
                end = (perm[k] + 1) / pop_size
                X[k, i] = lo[i] + (hi[i] - lo[i]) * (start + np.random.random() * (end - start))
                
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Main Evolutionary Loop
        # We maintain a population of size pop_size
        # Each iteration:
        # 1. Evaluate children (pop_size of them)
        # 2. Combine parents + children (2 * pop_size)
        # 3. Non-dominated sort
        # 4. Select next generation (pop_size)
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Check if we have enough budget for a full generation
            if used + pop_size > budget:
                # If not enough for full gen, do single-step improvements or break
                # For simplicity and safety, if remaining < pop_size, we might do random sampling
                # or just break. Given low budget, breaking is safer to avoid complex partial handling.
                # However, we can use the remaining budget for local refinement of best points.
                remaining = budget - used
                # Perform random local search with remaining budget
                for _ in range(remaining):
                    if time.time() > deadline: break
                    # Pick a random non-dominated point
                    fronts = fast_non_dominated_sort(F)
                    if fronts:
                        first_front = fronts[0]
                        parent_idx = np.random.choice(first_front)
                    else:
                        parent_idx = 0
                        
                    parent = X[parent_idx]
                    sigma = 0.1 * (hi - lo)
                    child = parent + np.random.normal(0, sigma, dim)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    X = np.vstack([X, child])
                    F = np.vstack([F, fc])
                break

            # Generate offspring
            offspring_X = np.zeros((pop_size, dim))
            for i in range(pop_size):
                # Select two parents
                p1_idx = tournament_select(list(range(len(X))), ranks, cds)
                p2_idx = tournament_select(list(range(len(X))), ranks, cds)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # BLX-Alpha Crossover
                # alpha = 0.5
                # c1 = min(p1, p2) - alpha * |p1 - p2|
                # c2 = max(p1, p2) + alpha * |p1 - p2|
                # Then sample uniformly in [c1, c2]
                alpha = 0.5
                min_p = np.minimum(p1, p2)
                max_p = np.maximum(p1, p2)
                diff = max_p - min_p
                c1 = min_p - alpha * diff
                c2 = max_p + alpha * diff
                
                # Sample 2 children
                # Actually, standard BLX produces 2 children from 2 parents. 
                # Here we generate 1 child per loop iteration to match pop_size offspring count.
                # So we pick one of the two ranges or just one child.
                # Let's just do uniform crossover + mutation for simplicity and robustness.
                
                # Simple Uniform Crossover
                mask = np.random.random(dim) < 0.5
                child = np.where(mask, p1, p2)
                
                # Polynomial/Gaussian Mutation
                # Adaptive sigma
                sigma = 0.1 * (hi - lo)
                child = child + np.random.normal(0, sigma, dim)
                child = np.clip(child, lo, hi)
                
                offspring_X[i] = child

            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine parents and offspring
            total_X = np.vstack([X, offspring_X])
            total_F = np.vstack([F, offspring_F])
            total_n = len(total_X)
            
            # Non-dominated Sort
            fronts = fast_non_dominated_sort(total_F)
            
            # Prepare next generation
            next_X = []
            next_F = []
            next_indices = []
            
            # Ranks for selection
            ranks = np.zeros(total_n, dtype=int)
            for r, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = r
            
            # Calculate crowding distances for fronts
            cds = {}
            for front in fronts:
                if len(next_indices) + len(front) > pop_size:
                    # We need to select only part of this front
                    cd = crowding_distance(total_F, front)
                    # Sort by cd descending
                    sorted_front = sorted(front, key=lambda idx: cd[idx], reverse=True)
                    needed = pop_size - len(next_indices)
                    selected = sorted_front[:needed]
                    for idx in selected:
                        next_indices.append(idx)
                    break
                else:
                    # Add whole front
                    cd = crowding_distance(total_F, front)
                    for idx in front:
                        cds[idx] = cd[idx]
                        next_indices.append(idx)
                        
            if len(next_indices) < pop_size:
                 # If we ran out of fronts (unlikely if pop_size is small), fill with best remaining
                 # This case is handled by the break above usually.
                 pass

            # Update population
            X = total_X[next_indices]
            F = total_F[next_indices]
            
            # Update ranks and cds for next iteration's selection
            # We need ranks and cds for the NEW population
            new_ranks = np.zeros(len(X), dtype=int)
            new_cds = {}
            new_fronts = fast_non_dominated_sort(F)
            for r, front in enumerate(new_fronts):
                for idx in front:
                    new_ranks[idx] = r
                cd = crowding_distance(F, front)
                for idx in front:
                    new_cds[idx] = cd[idx]
            
            ranks = new_ranks
            cds = new_cds

        # Return all evaluated points? 
        # The task says "Every returned objective vector is recorded".
        # The current code returns X. But X is just the final population.
        # The evaluator records "Every returned objective vector". 
        # Wait, the contract says: "Every returned objective vector is recorded and the return value of __call__ is ignored."
        # This implies the `problem` function calls are recorded by the evaluator wrapper, not by my return value.
        # "The return value of __call__ is ignored."
        # So I don't need to return anything specific, but I should probably return something valid.
        # The current code returns X. I will return the final population or all points.
        # Since return is ignored, I'll just return the final population X.
        
        return X
# EVOLVE-BLOCK-END
