import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 290  # Safety margin from 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        # 40 individuals, 5 generations for 200 budget
        pop_size = 40
        if self.budget < pop_size:
            pop_size = self.budget
            
        # Initialize
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Helper functions for NSGA-II
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                        n_dom[p] += 1
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_dom[q] += 1
                        
                if n_dom[p] == 0:
                    fronts[0].append(p)
                    
            i = 0
            while fronts[i]:
                next_front = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_dom[q] -= 1
                        if n_dom[q] == 0:
                            next_front.append(q)
                i += 1
                fronts.append(next_front)
            return fronts[:-1] # Remove last empty front if any

        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            
            m = len(front_indices)
            cd = np.zeros(m)
            for k in range(self.n_obj):
                f_k = F[front_indices, k]
                min_f = np.min(f_k)
                max_f = np.max(f_k)
                if max_f == min_f:
                    continue
                sorted_idx = np.argsort(f_k)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                for i in range(1, m - 1):
                    cd[sorted_idx[i]] += (f_k[sorted_idx[i+1]] - f_k[sorted_idx[i-1]]) / (max_f - min_f)
            return cd

        def tournament_selection(X, F, ranks, cds):
            i1, i2 = np.random.randint(0, len(X), 2)
            if ranks[i1] < ranks[i2]:
                return i1
            elif ranks[i2] < ranks[i1]:
                return i2
            else:
                return i1 if cds[i1] > cds[i2] else i2

        # Initial sorting for selection
        fronts = fast_non_dominated_sort(F)
        ranks = np.zeros(len(X), dtype=int)
        for i, front in enumerate(fronts):
            for idx in front:
                ranks[idx] = i
        cds = np.zeros(len(X))
        for front in fronts:
            if len(front) > 1:
                front_F = F[front]
                cd_vals = crowding_distance(front_F, np.arange(len(front)))
                for j, idx in enumerate(front):
                    cds[idx] = cd_vals[j]
            else:
                for idx in front:
                    cds[idx] = np.inf

        # Evolution Loop
        while used < self.budget:
            if time.time() > deadline:
                break
                
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # Determine how many offspring to create
            num_offspring = min(pop_size, self.budget - used)
            
            for _ in range(num_offspring):
                # Selection
                p1 = tournament_selection(X, F, ranks, cds)
                p2 = tournament_selection(X, F, ranks, cds)
                
                # Crossover (Simple Arithmetic)
                alpha = np.random.rand(self.dim)
                child = alpha * X[p1] + (1 - alpha) * X[p2]
                
                # Mutation (Gaussian)
                sigma = 0.1 * (hi - lo)
                child = child + np.random.normal(0, sigma)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                offspring_X.append(child)
                offspring_F.append(fc)
                used += 1
                
                if used >= self.budget or time.time() > deadline:
                    break

            if not offspring_X:
                break

            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine populations
            X_combined = np.vstack([X, offspring_X])
            F_combined = np.vstack([F, offspring_F])
            
            # Non-dominated sorting
            fronts = fast_non_dominated_sort(F_combined)
            
            # Assign ranks
            ranks_combined = np.zeros(len(X_combined), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks_combined[idx] = i
                    
            # Calculate crowding distance for selection
            cds_combined = np.zeros(len(X_combined))
            for front in fronts:
                if len(front) > 1:
                    front_F = F_combined[front]
                    cd_vals = crowding_distance(front_F, np.arange(len(front)))
                    for j, idx in enumerate(front):
                        cds_combined[idx] = cd_vals[j]
                else:
                    for idx in front:
                        cds_combined[idx] = np.inf

            # Select next generation
            new_X = []
            new_F = []
            remaining = pop_size
            
            for i, front in enumerate(fronts):
                if remaining <= 0:
                    break
                if len(front) <= remaining:
                    # Take all
                    for idx in front:
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    remaining -= len(front)
                else:
                    # Need to select some from this front based on crowding distance
                    front_idx = front
                    # Sort front indices by crowding distance descending
                    cd_vals = cds_combined[front_idx]
                    sorted_local = np.argsort(-cd_vals)
                    for j in range(remaining):
                        idx = front_idx[sorted_local[j]]
                        new_X.append(X_combined[idx])
                        new_F.append(F_combined[idx])
                    remaining = 0
                    break

            if not new_X:
                break

            X = np.array(new_X)
            F = np.array(new_F)
            
            # Recalculate ranks and cds for next generation selection
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(len(X), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            cds = np.zeros(len(X))
            for front in fronts:
                if len(front) > 1:
                    front_F = F[front]
                    cd_vals = crowding_distance(front_F, np.arange(len(front)))
                    for j, idx in enumerate(front):
                        cds[idx] = cd_vals[j]
                else:
                    for idx in front:
                        cds[idx] = np.inf

        return X
