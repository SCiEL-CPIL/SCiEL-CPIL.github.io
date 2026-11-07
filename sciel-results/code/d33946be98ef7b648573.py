import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 280  # Safety margin (limit is 300s)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(7)
        
        # Population size
        # N=20 allows 10 generations.
        # Let's try N=20 with Crossover.
        N = 20
        if self.budget < N * 2:
            N = max(2, self.budget // 2)
            
        # Initialize population
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = F.shape[0]
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]  # S[i] = indices dominated by i
            n_dom = np.zeros(n, dtype=int) # n_dom[i] = number of solutions that dominate i
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
                        
            current_front = [i for i in range(n) if n_dom[i] == 0]
            rank = 0
            while current_front:
                next_front = []
                for i in current_front:
                    ranks[i] = rank
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                rank += 1
                current_front = next_front
            return ranks
        
        # Helper: Crowding distance
        def crowding_distance(F, ranks):
            n = F.shape[0]
            distances = np.zeros(n)
            if n <= 2:
                return distances
                
            # Group by rank
            max_rank = np.max(ranks)
            for r in range(max_rank + 1):
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    continue
                    
                for m in range(self.n_obj):
                    # Sort by m-th objective
                    sorted_idx = indices[np.argsort(F[indices, m])]
                    distances[sorted_idx[0]] = np.inf
                    distances[sorted_idx[-1]] = np.inf
                    
                    obj_min = F[sorted_idx[0], m]
                    obj_max = F[sorted_idx[-1], m]
                    obj_range = obj_max - obj_min
                    if obj_range == 0:
                        continue
                        
                    for i in range(1, len(sorted_idx) - 1):
                        idx = sorted_idx[i]
                        prev_idx = sorted_idx[i - 1]
                        next_idx = sorted_idx[i + 1]
                        distances[idx] += (F[next_idx, m] - F[prev_idx, m]) / obj_range
                        
            return distances

        # Helper: SBX Crossover
        def sbx_crossover(p1, p2, lo, hi, eta, rng):
            c1 = p1.copy()
            c2 = p2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    u = rng.random()
                    if u <= 0.5:
                        alpha = (2 * u) ** (1 / (eta + 1))
                        beta = (2 * (1 - u)) ** (1 / (eta + 1))
                    else:
                        alpha = (2 * (1 - u)) ** (1 / (eta + 1))
                        beta = (2 * u) ** (1 / (eta + 1))
                    
                    # c1 = 0.5 * ((1 + alpha) * p1 + (1 - alpha) * p2)
                    # c2 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c1[i] = 0.5 * ((1 + alpha) * p1[i] + (1 - alpha) * p2[i])
                    c2[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    
                    # Clip to bounds
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            return c1, c2

        # Helper: Gaussian Mutation
        def gaussian_mutation(x, lo, hi, sigma, rng):
            # sigma is relative to range
            mutation = sigma * (hi - lo) * rng.standard_normal(self.dim)
            return np.clip(x + mutation, lo, hi)

        # Main Loop
        eta_crossover = 15
        prob_crossover = 0.9
        sigma_mutation = 0.1
        
        while evals < self.budget:
            if time.time() - start_time > timeout:
                break
            
            # Select parents
            ranks = non_dominated_sort(F)
            cd = crowding_distance(F, ranks)
            
            # Select N parents (for N children, we need N/2 pairs if using 2-child crossover, 
            # or just select N parents and do single-parent mutation? 
            # Standard NSGA-II: Select N parents, do crossover/mutation to create N offspring.
            # Usually, we select N/2 pairs, do crossover to get N children, then mutate.
            
            offspring_X = []
            
            # We need to generate N offspring.
            # Loop N/2 times to generate 2 children per pair?
            # Or select N parents and mutate?
            # Let's do: Select N/2 pairs, crossover to get N children.
            
            num_pairs = N // 2
            if N % 2 != 0:
                num_pairs += 1 # Handle odd N, though N=20 is even.
                
            for _ in range(num_pairs):
                # Tournament selection for 2 parents
                parents = []
                for _ in range(2):
                    i, j = rng.integers(0, N, size=2)
                    if ranks[i] != ranks[j]:
                        p = i if ranks[i] < ranks[j] else j
                    else:
                        p = i if cd[i] > cd[j] else j
                    parents.append(X[p])
                
                if rng.random() < prob_crossover:
                    c1, c2 = sbx_crossover(parents[0], parents[1], lo, hi, eta_crossover, rng)
                    c1 = gaussian_mutation(c1, lo, hi, sigma_mutation, rng)
                    c2 = gaussian_mutation(c2, lo, hi, sigma_mutation, rng)
                    offspring_X.append(c1)
                    offspring_X.append(c2)
                else:
                    # If no crossover, just copy and mutate?
                    # Or pick one parent and mutate twice?
                    c1 = gaussian_mutation(parents[0].copy(), lo, hi, sigma_mutation, rng)
                    c2 = gaussian_mutation(parents[1].copy(), lo, hi, sigma_mutation, rng)
                    offspring_X.append(c1)
                    offspring_X.append(c2)
            
            # Ensure we have exactly N offspring
            # If num_pairs * 2 > N, truncate. If < N, pad?
            # With N=20, num_pairs=10, offspring=20. Perfect.
            offspring_X = np.array(offspring_X[:N])
            
            # Check if we have enough budget for all offspring
            if evals + N > self.budget:
                # Evaluate only as many as we can
                num_to_eval = self.budget - evals
                offspring_X = offspring_X[:num_to_eval]
            
            # Evaluate offspring
            if len(offspring_X) == 0:
                break
                
            if time.time() - start_time > timeout:
                break
                
            offspring_F = np.array([problem(x) for x in offspring_X])
            evals += len(offspring_F)
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Sort and Select
            ranks_comb = non_dominated_sort(F_comb)
            cd_comb = crowding_distance(F_comb, ranks_comb)
            
            # Create selection key: (rank, -crowding_distance)
            selection_keys = np.stack([ranks_comb, -cd_comb], axis=1)
            order = np.lexsort((selection_keys[:, 1], selection_keys[:, 0]))
            
            X = X_comb[order[:N]]
            F = F_comb[order[:N]]
            
        return X
