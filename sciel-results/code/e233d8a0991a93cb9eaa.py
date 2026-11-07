import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        # Set population size and generations to fit budget
        # Reserve some for initial population
        self.pop_size = 20
        # Calculate generations
        # Total evals = pop_size + generations * pop_size
        # 200 = 20 + G * 20 => G = 9
        self.generations = max(1, (budget - self.pop_size) // self.pop_size)
        
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42) # Fixed seed for reproducibility within the run, though evaluator seeds np.random
        
        # 1. Initial Population
        X = lo + (hi - lo) * rng.random((self.pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = self.pop_size
        
        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_doms = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_doms[i] += 1
            
            current_front = []
            for i in range(n):
                if n_doms[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            k = 0
            while len(current_front) > 0:
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return rank
        
        # Helper: Crowding Distance
        def crowding_distance(F, rank):
            n = len(F)
            cd = np.zeros(n)
            for r in np.unique(rank):
                idx = np.where(rank == r)[0]
                m = len(idx)
                if m <= 2:
                    cd[idx] = np.inf
                    continue
                for i in range(self.n_obj):
                    f_vals = F[idx, i]
                    min_val = np.min(f_vals)
                    max_val = np.max(f_vals)
                    if max_val - min_val < 1e-10:
                        continue
                    # Sort indices by objective i
                    sorted_idx = idx[np.argsort(f_vals)]
                    cd[sorted_idx[0]] = np.inf
                    cd[sorted_idx[-1]] = np.inf
                    for k in range(1, m - 1):
                        prev_val = f_vals[sorted_idx[k-1]]
                        next_val = f_vals[sorted_idx[k+1]]
                        cd[sorted_idx[k]] += (next_val - prev_val) / (max_val - min_val)
            return cd

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    if abs(x1[i] - x2[i]) > 1e-14:
                        u = rng.random()
                        if u <= 0.5:
                            alpha = 2.0 * u
                            beta = ((alpha + 1.0) ** (1.0 / (eta + 1.0)))
                            child1[i] = 0.5 * ((1.0 + beta) * x1[i] + (1.0 - beta) * x2[i])
                            child2[i] = 0.5 * ((1.0 - beta) * x1[i] + (1.0 + beta) * x2[i])
                        else:
                            alpha = 2.0 * (1.0 - u)
                            beta = ((alpha + 1.0) ** (1.0 / (eta + 1.0)))
                            child1[i] = 0.5 * ((1.0 + beta) * x2[i] + (1.0 - beta) * x1[i])
                            child2[i] = 0.5 * ((1.0 - beta) * x2[i] + (1.0 + beta) * x1[i])
            return child1, child2

        # Helper: Polynomial Mutation
        def polynomial_mutation(x, eta=20.0):
            child = x.copy()
            for i in range(self.dim):
                if rng.random() < 0.5:
                    delta1 = (2.0 * rng.random()) ** (1.0 / (eta + 1.0)) - 1.0
                    delta2 = 1.0 - (2.0 * (1.0 - rng.random())) ** (1.0 / (eta + 1.0))
                    child[i] += 0.5 * (hi[i] - lo[i]) * (delta1 + delta2)
            return np.clip(child, lo, hi)

        # Main Evolutionary Loop
        for gen in range(self.generations):
            offspring = []
            for _ in range(self.pop_size):
                # Tournament Selection
                idx1 = rng.integers(self.pop_size)
                idx2 = rng.integers(self.pop_size)
                # Simple tournament: pick best of 2 based on rank and cd
                # We need ranks and cd from previous generation
                # Calculate ranks and cd for current X, F
                ranks = fast_non_dominated_sort(F)
                cds = crowding_distance(F, ranks)
                
                def compare(i, j):
                    if ranks[i] != ranks[j]:
                        return ranks[i] < ranks[j]
                    return cds[i] > cds[j]
                
                p1_idx = idx1 if compare(idx1, idx2) else idx2
                p2_idx = idx1 if compare(idx1, idx2) else idx2
                # Ensure distinct parents if possible, or just pick two
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                c1, c2 = sbx_crossover(p1, p2)
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                offspring.append(c1)
                if len(offspring) < self.pop_size:
                    offspring.append(c2)
            
            # Evaluate Offspring
            F_off = np.array([problem(x) for x in offspring])
            used += self.pop_size
            
            # Combine and Select
            X_combined = np.vstack([X, np.array(offspring)])
            F_combined = np.vstack([F, F_off])
            
            ranks_comb = fast_non_dominated_sort(F_combined)
            cds_comb = crowding_distance(F_combined, ranks_comb)
            
            # Sort by rank, then by crowding distance (descending)
            order = np.lexsort((cds_comb * -1, ranks_comb))
            next_X = X_combined[order[:self.pop_size]]
            next_F = F_combined[order[:self.pop_size]]
            
            X = next_X
            F = next_F
            
            if used >= self.budget:
                break
                
        return X
