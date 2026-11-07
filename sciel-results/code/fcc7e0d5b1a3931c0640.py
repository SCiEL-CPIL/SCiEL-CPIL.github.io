# EVOLVE-BLOCK-START
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        all_X = []
        all_F = []
        used = 0
        
        def evaluate(x):
            nonlocal used
            x = np.clip(np.asarray(x, dtype=float), lo, hi)
            f = np.asarray(problem(x), dtype=float)
            all_X.append(x.copy())
            all_F.append(f.copy())
            used += 1
            return f

        # Determine population size
        # Keep it small to allow more generations within budget
        # Standard NSGA-II uses N=100, but for budget=200, we need smaller N
        # Let's use N=20. 
        # Initial pop: 20 evals
        # Each gen: 20 offspring (20 evals) -> selection from 40
        # Total evals = 20 + k*20.
        # 20 + 9*20 = 200. So 9 generations.
        N = 20
        
        # 1. Initialize population
        pop = []
        for _ in range(N):
            x = lo + (hi - lo) * rng.random(self.dim)
            f = evaluate(x)
            pop.append((x, f))
        
        # Helper: Fast Non-Dominated Sorting
        def fast_nondominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)] # Dominance relation
            n_dominating = [0] * n
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q:
                        continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dominating[p] += 1
                        
                if n_dominating[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dominating[q] -= 1
                        if n_dominating[q] == 0:
                            Q.append(q)
                i += 1
                fronts.append(Q)
            
            # Remove last empty front if any
            if len(fronts) > 0 and len(fronts[-1]) == 0:
                fronts.pop()
                
            return fronts

        # Helper: Crowding Distance Assignment
        def assign_crowding_distances(front, F):
            if len(front) <= 2:
                return [float('inf')] * len(front)
            
            m = len(F[front[0]])
            crowd = [0.0] * len(front)
            
            for k in range(m):
                # Sort front by k-th objective
                sorted_front = sorted(front, key=lambda idx: F[idx][k])
                crowd[sorted_front[0]] = float('inf')
                crowd[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0]][k]
                f_max = F[sorted_front[-1]][k]
                
                if f_max - f_min == 0:
                    continue
                    
                for i in range(1, len(sorted_front) - 1):
                    idx = sorted_front[i]
                    crowd[idx] += (F[sorted_front[i+1]][k] - F[sorted_front[i-1]][k]) / (f_max - f_min)
            
            return crowd

        # Helper: Binary Tournament Selection
        def tournament_select(pop, fronts, crowd_dict):
            # pop is list of (x, f)
            # We need to select based on fronts and crowding
            # Map index in pop to front number and crowding
            # Since we recompute fronts every generation, we can just pass the sorted order
            
            # Create a list of indices sorted by (front_rank, -crowding)
            # front_rank: 0 is best
            # We need to map population indices to their rank in the current non-dominated sorting
            
            # This is slightly tricky because `pop` indices change.
            # Let's create a list of (front_rank, crowd, pop_idx)
            candidates = []
            for i, (x, f) in enumerate(pop):
                # Find which front i is in
                # This is O(N^2) per selection if done naively.
                # Better to pass the sorted list of indices.
                pass
            
            # Actually, let's just use the `fronts` and `crowd_dict` which maps pop_idx -> crowd
            # But `crowd_dict` is computed per front.
            # Let's create a global rank score.
            scores = {}
            for rank, front in enumerate(fronts):
                # Compute crowding for this front
                front_indices = front
                # Get F values for these indices
                F_subset = [pop[idx][1] for idx in front_indices]
                # Wait, `assign_crowding_distances` expects F array and front indices relative to that F array?
                # My helper `assign_crowding_distances` takes `front` (list of indices into F) and `F`.
                # Here `front` is list of indices into `pop`.
                # So we can pass `front` and `[p[1] for p in pop]`? No, that's inefficient.
                
                # Let's refactor helpers to work with the population list directly.
                pass
            
            # Simplified: Just pick two random individuals from pop, compare their non-domination rank and crowding.
            # To make it fast, we precompute rank and crowding for all current pop members.
            
            # This logic is getting complex to inline. Let's write a cleaner selection.
            
            # Precompute rank and crowding for all N individuals in `pop`
            # `fronts` is list of lists of indices into `pop`
            # We need crowding distance for each individual in `pop`.
            
            # Recompute crowding for all fronts
            crowd_dist = [0.0] * N
            for front in fronts:
                if len(front) <= 2:
                    for idx in front:
                        crowd_dist[idx] = float('inf')
                else:
                    # Get F values for this front
                    F_front = np.array([pop[idx][1] for idx in front])
                    # Indices are 0..len(front)-1 relative to F_front
                    # We need to map back to pop indices
                    # My `assign_crowding_distances` returns crowd distances for the `front` list provided.
                    # If I pass `front` (list of pop indices) and `F_front`, it returns distances for positions in `front`.
                    # So `crowd_vals[i]` corresponds to `front[i]`.
                    
                    # Wait, my helper `assign_crowding_distances(front, F)` assumes `front` contains indices into `F`.
                    # If I pass `front` (indices into pop) and `F_front` (array of F values for those indices),
                    # then `F[idx]` in the helper would be wrong because `idx` is a pop index, not an index into `F_front`.
                    
                    # Let's fix the helper usage.
                    # I will create a wrapper that correctly maps.
                    
                    pass
            
            # Let's just implement a simple binary tournament that compares two individuals directly.
            # Comparison:
            # 1. If a dominates b, a wins.
            # 2. If b dominates a, b wins.
            # 3. If incomparable, compare front rank (lower is better).
            # 4. If same front, compare crowding distance (higher is better).
            
            # To do this efficiently, we need the rank and crowding distance for every individual in the current population.
            
            # Let's compute ranks and crowding distances once per generation for the current `pop`.
            
            # 1. Fast Non-Dominated Sorting on current pop
            F_pop = np.array([p[1] for p in pop])
            fronts = fast_nondominated_sort(F_pop)
            
            # 2. Crowding Distances
            # We need to map pop index to its crowding distance.
            crowd_dist = [0.0] * N
            for front in fronts:
                if len(front) <= 2:
                    for idx in front:
                        crowd_dist[idx] = float('inf')
                else:
                    # Extract F values for this front
                    F_subset = F_pop[front]
                    # Call assign_crowding_distances with indices 0..len(front)-1
                    # The helper expects `front` to be indices into `F`.
                    # So we pass `list(range(len(front)))` and `F_subset`.
                    # It returns distances for these local indices.
                    local_crowd = assign_crowding_distances(list(range(len(front))), F_subset)
                    for local_idx, dist in enumerate(local_crowd):
                        pop_idx = front[local_idx]
                        crowd_dist[pop_idx] = dist
            
            # 3. Binary Tournament Selection
            i1 = rng.integers(0, N)
            i2 = rng.integers(0, N)
            
            # Determine rank for i1 and i2
            rank1 = next(i for i, f in enumerate(fronts) if i1 in f)
            rank2 = next(i for i, f in enumerate(fronts) if i2 in f)
            
            if rank1 < rank2:
                return pop[i1]
            elif rank2 < rank1:
                return pop[i2]
            else:
                # Same front, compare crowding
                if crowd_dist[i1] > crowd_dist[i2]:
                    return pop[i1]
                else:
                    return pop[i2]

        # NSGA-II Main Loop
        while used < self.budget:
            # Generate offspring
            offspring = []
            for _ in range(N):
                if used >= self.budget:
                    break
                # Binary Tournament Selection for parents
                p1 = tournament_select(pop, None, None) # Note: tournament_select now handles its own sorting? 
                # Wait, my tournament_select above was a mess. 
                # Let's refactor `tournament_select` to take `pop`, `fronts`, `crowd_dist`.
                
                # I need to pass the precomputed fronts and crowd_dist.
                # Let's rewrite the loop to compute fronts/crowd once per generation.
                pass
            
            # The logic above inside `tournament_select` was recomputing fronts every time it was called, which is O(N^3) per generation.
            # Let's restructure.
            
            # 1. Compute fronts and crowding for current `pop`
            F_pop = np.array([p[1] for p in pop])
            fronts = fast_nondominated_sort(F_pop)
            
            crowd_dist = [0.0] * N
            for front in fronts:
                if len(front) <= 2:
                    for idx in front:
                        crowd_dist[idx] = float('inf')
                else:
                    F_subset = F_pop[front]
                    local_crowd = assign_crowding_distances(list(range(len(front))), F_subset)
                    for local_idx, dist in enumerate(local_crowd):
                        pop_idx = front[local_idx]
                        crowd_dist[pop_idx] = dist
            
            # 2. Generate offspring
            offspring = []
            for _ in range(N):
                if used >= self.budget:
                    break
                
                # Select Parent 1
                i1 = rng.integers(0, N)
                i2 = rng.integers(0, N)
                rank1 = next(i for i, f in enumerate(fronts) if i1 in f)
                rank2 = next(i for i, f in enumerate(fronts) if i2 in f)
                if rank1 < rank2:
                    p1_idx = i1
                elif rank2 < rank1:
                    p1_idx = i2
                else:
                    if crowd_dist[i1] > crowd_dist[i2]:
                        p1_idx = i1
                    else:
                        p1_idx = i2
                p1 = pop[p1_idx][0]
                
                # Select Parent 2
                i3 = rng.integers(0, N)
                i4 = rng.integers(0, N)
                rank3 = next(i for i, f in enumerate(fronts) if i3 in f)
                rank4 = next(i for i, f in enumerate(fronts) if i4 in f)
                if rank3 < rank4:
                    p2_idx = i3
                elif rank4 < rank3:
                    p2_idx = i4
                else:
                    if crowd_dist[i3] > crowd_dist[i4]:
                        p2_idx = i3
                    else:
                        p2_idx = i4
                p2 = pop[p2_idx][0]
                
                # SBX Crossover (Simulated Binary Crossover)
                if rng.random() < 0.9: # Crossover probability
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            # Apply SBX
                            u = rng.random()
                            if u <= 0.5:
                                beta = (2 * u) ** (1 / (2 + 20)) # eta_c = 20
                            else:
                                beta = (2 * (1 - u)) ** (1 / (2 + 20))
                            
                            x1 = p1[d]
                            x2 = p2[d]
                            if x1 == x2:
                                child1[d] = x1
                                child2[d] = x2
                            else:
                                if x1 < x2:
                                    child1[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                                    child2[d] = 0.5 * ((1 + beta) * x2 + (1 - beta) * x1)
                                else:
                                    child1[d] = 0.5 * ((1 + beta) * x2 + (1 - beta) * x1)
                                    child2[d] = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                        else:
                            child1[d] = p1[d]
                            child2[d] = p2[d]
                    
                    # Mutation (Polynomial Mutation)
                    for d in range(self.dim):
                        if rng.random() < 1.0 / self.dim: # Mutation rate
                            # Simple random mutation within bounds
                            child1[d] = lo[d] + (hi[d] - lo[d]) * rng.random()
                        if rng.random() < 1.0 / self.dim:
                            child2[d] = lo[d] + (hi[d] - lo[d]) * rng.random()
                    
                    # Clip to bounds
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    f1 = evaluate(child1)
                    offspring.append((child1, f1))
                    
                    if used < self.budget:
                        f2 = evaluate(child2)
                        offspring.append((child2, f2))
                else:
                    # No crossover, just copy
                    child1 = p1.copy()
                    f1 = evaluate(child1)
                    offspring.append((child1, f1))
                    
                    if used < self.budget:
                        child2 = p2.copy()
                        f2 = evaluate(child2)
                        offspring.append((child2, f2))
            
            # Combine pop and offspring
            combined = pop + offspring
            
            # Sort combined to select next pop
            F_combined = np.array([c[1] for c in combined])
            fronts_combined = fast_nondominated_sort(F_combined)
            
            new_pop = []
            for front in fronts_combined:
                if len(new_pop) + len(front) <= N:
                    for idx in front:
                        new_pop.append(combined[idx])
                else:
                    # Need to truncate this front based on crowding distance
                    # Calculate crowding distances for this front within the combined set
                    F_front = F_combined[front]
                    local_crowd = assign_crowding_distances(list(range(len(front))), F_front)
                    
                    # Sort front by crowding distance (descending)
                    sorted_front = sorted(range(len(front)), key=lambda i: local_crowd[i], reverse=True)
                    
                    needed = N - len(new_pop)
                    for i in range(needed):
                        idx = front[sorted_front[i]]
                        new_pop.append(combined[idx])
                    break
            
            pop = new_pop

        return all_X[-1] if all_X else lo.copy()

# EVOLVE-BLOCK-END
