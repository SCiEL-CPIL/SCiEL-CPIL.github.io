"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II multi-objective optimizer with stratified initialization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        pop_size = 40
        
        # Stratified initialization for better initial coverage
        X = np.zeros((pop_size, dim))
        for i in range(pop_size):
            # Use Latin hypercube-like stratification
            for d in range(dim):
                # Divide range into pop_size strata and pick one point per stratum
                # With replacement to allow some overlap, but ensure coverage
                stratum = np.random.randint(0, pop_size)
                low_stratum = lo[d] + stratum * (hi[d] - lo[d]) / pop_size
                high_stratum = lo[d] + (stratum + 1) * (hi[d] - lo[d]) / pop_size
                X[i, d] = np.random.uniform(low_stratum, high_stratum)
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II helpers
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = [0] * n
            rank = [0] * n
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n_count[p] += 1
            
            front = []
            current_front = []
            for p in range(n):
                if n_count[p] == 0:
                    rank[p] = 0
                    current_front.append(p)
            front.append(current_front)
            
            i = 0
            while current_front:
                i += 1
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            rank[q] = i
                            next_front.append(q)
                front.append(next_front)
                current_front = next_front
                
            return front, rank
        
        def crowding_distance(F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return [float('inf')] * n
            dist = np.zeros(n)
            for m in range(n_obj):
                vals = F[front_indices, m]
                min_val = np.min(vals)
                max_val = np.max(vals)
                if max_val - min_val < 1e-12:
                    continue
                # Sort indices by objective m
                sorted_idx = np.argsort(vals)
                dist[sorted_idx[0]] = float('inf')
                dist[sorted_idx[-1]] = float('inf')
                for i in range(1, n - 1):
                    dist[sorted_idx[i]] += (vals[sorted_idx[i+1]] - vals[sorted_idx[i-1]]) / (max_val - min_val)
            return dist
        
        def tournament_select(pop_indices, ranks, dists, k=2):
            idx1 = np.random.randint(len(pop_indices))
            idx2 = np.random.randint(len(pop_indices))
            i1 = pop_indices[idx1]
            i2 = pop_indices[idx2]
            if ranks[i1] < ranks[i2]:
                return i1
            elif ranks[i2] < ranks[i1]:
                return i2
            else:
                if dists[i1] > dists[i2]:
                    return i1
                else:
                    return i2

        def sbx_crossover(x1, x2, eta=20.0):
            x = x1.copy()
            for i in range(dim):
                if np.random.random() < 0.5:
                    if abs(x1[i] - x2[i]) < 1e-12:
                        continue
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    x[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
            return x

        def poly_mutation(x, eta=20.0):
            x = x.copy()
            for i in range(dim):
                if np.random.random() < 1.0 / dim:
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1.0 / (eta + 1.0)) - 1.0
                    else:
                        delta = 1.0 - (2 * (1.0 - u)) ** (1.0 / (eta + 1.0))
                    x[i] += delta * (hi[i] - lo[i])
            x = np.clip(x, lo, hi)
            return x

        # Main loop
        while used < self.budget:
            # Check time
            if time.time() > deadline:
                break
            
            # Non-dominated sort and crowding distance for current population
            fronts, ranks = fast_non_dominated_sort(F)
            
            dists = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 0:
                    front_dists = crowding_distance(F, front)
                    for j, idx in enumerate(front):
                        dists[idx] = front_dists[j]
            
            # Generate offspring
            offspring_X = np.zeros((pop_size, dim))
            for i in range(pop_size):
                p1 = tournament_select(range(pop_size), ranks, dists)
                p2 = tournament_select(range(pop_size), ranks, dists)
                child = sbx_crossover(X[p1], X[p2])
                child = poly_mutation(child)
                offspring_X[i] = child
            
            F_off = np.array([problem(x) for x in offspring_X])
            used += pop_size
            
            # Combine parents and offspring
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, F_off])
            
            # Non-dominated sort of combined population
            fronts_all, ranks_all = fast_non_dominated_sort(F_all)
            
            # Calculate crowding distance for combined population
            dists_all = np.zeros(len(X_all))
            for front in fronts_all:
                if len(front) > 0:
                    front_dists = crowding_distance(F_all, front)
                    for j, idx in enumerate(front):
                        dists_all[idx] = front_dists[j]

            # Select next population
            X_new = np.zeros((pop_size, dim))
            F_new = np.zeros((pop_size, n_obj))
            count = 0
            for front in fronts_all:
                if count + len(front) <= pop_size:
                    for idx in front:
                        X_new[count] = X_all[idx]
                        F_new[count] = F_all[idx]
                        count += 1
                else:
                    # Need to select from this front based on crowding distance
                    remaining = pop_size - count
                    # Sort indices by crowding distance (descending)
                    sorted_indices = np.argsort(dists_all[front])[::-1]
                    for k in range(remaining):
                        idx = front[sorted_indices[k]]
                        X_new[count] = X_all[idx]
                        F_new[count] = F_all[idx]
                        count += 1
                    break
            
            X = X_new
            F = F_new
        
        return X
# EVOLVE-BLOCK-END
