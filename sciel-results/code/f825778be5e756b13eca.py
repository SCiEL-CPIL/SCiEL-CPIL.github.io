import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
    
    def __call__(self, problem, bounds):
        # Use a fixed seed for reproducibility if needed, but evaluator sets np.random.seed
        # We should use np.random directly or a local RNG. 
        # The prompt says "For seeds 0 and 1 the evaluator calls np.random.seed(seed)".
        # So we should use np.random functions to respect the global seed.
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        N = 20  # Population size
        # Budget is 200. Init 20. Remaining 180.
        # Each generation produces N offspring. 180 // 20 = 9 generations.
        n_gen = (self.budget - N) // N
        
        # Initialize population
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        def dominates(f1, f2):
            # f1 dominates f2 if f1 <= f2 in all objectives and f1 < f2 in at least one
            return np.all(f1 <= f2) and np.any(f1 < f2)
        
        def non_dominated_sort(F):
            n = len(F)
            fronts = []
            domination_count = np.zeros(n, dtype=int)
            dominated_set = [[] for _ in range(n)]
            
            for i in range(n):
                for j in range(i + 1, n):
                    if dominates(F[i], F[j]):
                        dominated_set[i].append(j)
                        domination_count[j] += 1
                    elif dominates(F[j], F[i]):
                        dominated_set[j].append(i)
                        domination_count[i] += 1
            
            current_front = [i for i in range(n) if domination_count[i] == 0]
            fronts.append(current_front)
            
            while current_front:
                next_front = []
                for i in current_front:
                    for j in dominated_set[i]:
                        domination_count[j] -= 1
                        if domination_count[j] == 0:
                            next_front.append(j)
                if next_front:
                    fronts.append(next_front)
                current_front = next_front
                
            return fronts
        
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            distances = np.zeros(len(indices))
            n_obj = self.n_obj
            
            for m in range(n_obj):
                # Sort indices by m-th objective
                sorted_indices = sorted(indices, key=lambda i: F[i, m])
                # Assign infinite distance to best and worst
                distances[sorted_indices[0]] = np.inf
                distances[sorted_indices[-1]] = np.inf
                
                f_min = F[sorted_indices[0], m]
                f_max = F[sorted_indices[-1], m]
                range_val = f_max - f_min
                
                if range_val == 0:
                    continue
                    
                for k in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[k]
                    prev_idx = sorted_indices[k-1]
                    next_idx = sorted_indices[k+1]
                    distances[idx] += (F[next_idx, m] - F[prev_idx, m]) / range_val
                    
            return distances

        def tournament_select(pop_idx, F, fronts, k=2):
            # Randomly pick k indices from pop_idx
            selected = np.random.choice(pop_idx, size=k, replace=False)
            best = selected[0]
            for i in range(1, k):
                cand = selected[i]
                # Compare ranks (fronts)
                # We need a mapping from index to front rank
                # This is inefficient to compute every time. 
                # Let's precompute rank for the combined population.
                pass
            
            # Better approach: Pass rank array
            return best # Placeholder

        # Refactor to use rank array for efficiency
        
        def sbx_crossover(x1, x2, lo, hi, eta=15.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if np.random.random() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        if x1[i] < x2[i]:
                            u = np.random.random()
                            if u <= 1.0 / (eta + 1):
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                            c1 = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                            c2 = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
                        else:
                            u = np.random.random()
                            if u <= 1.0 / (eta + 1):
                                beta = (2 * u) ** (1.0 / (eta + 1))
                            else:
                                beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                            c1 = 0.5 * ((1 + beta) * x2[i] + (1 - beta) * x1[i])
                            c2 = 0.5 * ((1 - beta) * x2[i] + (1 + beta) * x1[i])
                        
                        if c1 < lo[i]:
                            c1 = lo[i]
                            c2 = 2 * x2[i] - c1
                        elif c1 > hi[i]:
                            c1 = hi[i]
                            c2 = 2 * x2[i] - c1
                            
                        if c2 < lo[i]:
                            c2 = lo[i]
                            c1 = 2 * x1[i] - c2
                        elif c2 > hi[i]:
                            c2 = hi[i]
                            c1 = 2 * x1[i] - c2
                            
                        child1[i] = c1
                        child2[i] = c2
            return child1, child2

        def polynomial_mutation(x, lo, hi, eta=20.0):
            child = x.copy()
            for i in range(self.dim):
                if np.random.random() < 1.0 / self.dim:
                    if x[i] < lo[i] + 1e-14 or x[i] > hi[i] - 1e-14:
                        child[i] = lo[i] + (hi[i] - lo[i]) * np.random.random()
                    else:
                        u = np.random.random()
                        if u < 0.5:
                            delta1 = (2 * u + (x[i] - lo[i]) / (hi[i] - lo[i])) ** (1.0 / (eta + 1)) - 1.0
                        else:
                            delta1 = 1.0 - (2 * (1 - u) + (hi[i] - x[i]) / (hi[i] - lo[i])) ** (1.0 / (eta + 1))
                        
                        child[i] = x[i] + delta1 * (hi[i] - lo[i])
                        child[i] = np.clip(child[i], lo[i], hi[i])
            return child

        # Main Loop
        for gen in range(n_gen):
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            # We need ranks for the current population X to select parents
            fronts_X = non_dominated_sort(F)
            rank_X = np.zeros(N, dtype=int)
            for rank, front in enumerate(fronts_X):
                for idx in front:
                    rank_X[idx] = rank
            
            # Calculate crowding distance for all fronts to store in an array
            cd_X = np.zeros(N)
            for front in fronts_X:
                cd_X[front] = crowding_distance(F, front)
                
            for i in range(N):
                # Select two parents using tournament
                # Parent 1
                p1_idx = np.random.choice(N, size=2, replace=False)
                if rank_X[p1_idx[0]] < rank_X[p1_idx[1]]:
                    p1 = p1_idx[0]
                elif rank_X[p1_idx[1]] < rank_X[p1_idx[0]]:
                    p1 = p1_idx[1]
                else:
                    if cd_X[p1_idx[0]] > cd_X[p1_idx[1]]:
                        p1 = p1_idx[0]
                    else:
                        p1 = p1_idx[1]
                        
                # Parent 2
                p2_idx = np.random.choice(N, size=2, replace=False)
                if rank_X[p2_idx[0]] < rank_X[p2_idx[1]]:
                    p2 = p2_idx[0]
                elif rank_X[p2_idx[1]] < rank_X[p2_idx[0]]:
                    p2 = p2_idx[1]
                else:
                    if cd_X[p2_idx[0]] > cd_X[p2_idx[1]]:
                        p2 = p2_idx[0]
                    else:
                        p2 = p2_idx[1]
                        
                # Crossover
                c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi)
                
                # Mutation
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                # Evaluate
                f1 = problem(c1)
                f2 = problem(c2)
                used += 2
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                offspring_F.append(f1)
                offspring_F.append(f2)
                
                # If we exceed budget, break? 
                # Budget check:
                # We loop N times, adding 2 evaluations each. Total 2*N per gen.
                # If used > budget, we should stop.
                if used >= self.budget:
                    break
                    
            if used >= self.budget:
                break

            # Combine
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.array([F, np.array(offspring_F)])
            
            # Sort and select
            fronts_comb = non_dominated_sort(F_comb)
            rank_comb = np.zeros(len(X_comb), dtype=int)
            for rank, front in enumerate(fronts_comb):
                for idx in front:
                    rank_comb[idx] = rank
            
            cd_comb = np.zeros(len(X_comb))
            for front in fronts_comb:
                cd_comb[front] = crowding_distance(F_comb, front)
                
            # Select N best
            selected_indices = []
            for front in fronts_comb:
                if len(selected_indices) + len(front) <= N:
                    selected_indices.extend(front)
                else:
                    # Sort front by crowding distance descending
                    front_sorted = sorted(front, key=lambda i: cd_comb[i], reverse=True)
                    needed = N - len(selected_indices)
                    selected_indices.extend(front_sorted[:needed])
                    break
                    
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]
            
        return X
