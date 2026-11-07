import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 290  # Safety margin

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        # Population size
        N = 20
        if self.budget < N * 2:
            N = max(2, self.budget // 2)

        # LHS Initialization for broad coverage
        X = np.zeros((N, self.dim))
        for d in range(self.dim):
            perm = np.random.permutation(N)
            for i in range(N):
                X[perm[i], d] = lo[d] + (hi[d] - lo[d]) * (i + 0.5) / N
        
        F = np.array([problem(x) for x in X])
        evals = N

        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            n = F.shape[0]
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)

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

            max_rank = np.max(ranks)
            for r in range(max_rank + 1):
                indices = np.where(ranks == r)[0]
                if len(indices) <= 2:
                    continue

                for m in range(self.n_obj):
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

        # Helper: DE Mutation (Differential Evolution)
        def de_mutation(x_best, x_rand1, x_rand2, F_param=0.8):
            return x_best + F_param * (x_rand1 - x_rand2)

        # Helper: Binomial Crossover
        def binomial_crossover(v, x, CR=0.9):
            j_rand = np.random.randint(self.dim)
            child = np.zeros(self.dim)
            for i in range(self.dim):
                if np.random.rand() < CR or i == j_rand:
                    child[i] = v[i]
                else:
                    child[i] = x[i]
            return np.clip(child, lo, hi)

        # Main Loop
        while evals < self.budget:
            if time.time() - start_time > timeout:
                break

            ranks = non_dominated_sort(F)
            cd = crowding_distance(F, ranks)

            # Find best individual (rank 0, highest crowding? No, usually rank 0 is best. 
            # For DE, we often use the global best or a random non-dominated individual.
            # Let's use a random individual from the first non-dominated front as the "best" for DE.
            # Or just pick a random index if front 0 is empty (shouldn't happen).
            front0_indices = np.where(ranks == 0)[0]
            if len(front0_indices) == 0:
                front0_indices = np.arange(N)
            idx_best = np.random.choice(front0_indices)
            
            offspring_X = []
            for _ in range(N):
                # Select base individual
                # Binary Tournament Selection for base
                i, j = np.random.choice(N, 2, replace=False)
                if ranks[i] != ranks[j]:
                    base_idx = i if ranks[i] < ranks[j] else j
                else:
                    base_idx = i if cd[i] > cd[j] else j
                
                # Select two other random individuals for DE
                idx1, idx2 = np.random.choice(N, 2, replace=False)
                # Ensure they are distinct from base and each other
                while idx1 == base_idx or idx2 == base_idx or idx1 == idx2:
                    idx1, idx2 = np.random.choice(N, 2, replace=False)

                # DE Mutation: v = x_best + F * (x1 - x2)
                # Using a "best" from front 0 to guide convergence, or just x_base?
                # Standard DE/rand/1: v = x_rand + F*(x1-x2)
                # DE/best/1: v = x_best + F*(x1-x2)
                # Let's use DE/best/1 with the global best (from front 0) to exploit, 
                # or DE/rand/1 for exploration.
                # Given the stagnation issue, maybe DE/rand/1 is better for diversity, 
                # but DE/best/1 converges faster.
                # Let's try a mix or just DE/rand/1 with high F.
                v = X[base_idx] + 0.8 * (X[idx1] - X[idx2])
                
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Binomial Crossover
                child = binomial_crossover(v, X[base_idx], CR=0.9)
                
                offspring_X.append(child)

            offspring_X = np.array(offspring_X)
            
            # Evaluate offspring
            if time.time() - start_time > timeout:
                break

            offspring_F = np.array([problem(x) for x in offspring_X])
            evals += N

            # Combine and Select
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            ranks_comb = non_dominated_sort(F_comb)
            cd_comb = crowding_distance(F_comb, ranks_comb)
            
            # Selection
            order = np.lexsort((-cd_comb, ranks_comb))
            
            X = X_comb[order[:N]]
            F = F_comb[order[:N]]

        return X
