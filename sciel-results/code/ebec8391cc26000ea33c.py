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
        
        pop_size = 20
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            front = [0]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
                    front[0] += 1
            
            i = 0
            while front[i] > 0:
                i += 1
                front.append(0)
                for a in range(n):
                    if rank[a] == i - 1:
                        for b in S[a]:
                            n_dom[b] -= 1
                            if n_dom[b] == 0:
                                rank[b] = i
                                front[i] += 1
            return rank
        
        def crowding_distance(F, rank, indices):
            # indices: list of indices in F that have the same rank
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            dist = np.zeros(len(indices))
            for m in range(self.n_obj):
                obj_vals = F[indices, m]
                min_val = np.min(obj_vals)
                max_val = np.max(obj_vals)
                if max_val - min_val < 1e-12:
                    continue
                # Sort indices by objective m
                sorted_idx = np.argsort(obj_vals)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for k in range(1, len(indices) - 1):
                    dist[sorted_idx[k]] += (obj_vals[sorted_idx[k+1]] - obj_vals[sorted_idx[k-1]]) / (max_val - min_val)
            return dist

        while used < self.budget:
            # Generate offspring
            offspring_X = np.zeros((pop_size, self.dim))
            offspring_F = np.zeros((pop_size, self.n_obj))
            
            for i in range(pop_size):
                # Binary tournament selection
                p1, p2 = rng.integers(0, pop_size, 2)
                # Simple domination check for selection
                if np.all(F[p1] <= F[p2]) and np.any(F[p1] < F[p2]):
                    p = p1
                elif np.all(F[p2] <= F[p1]) and np.any(F[p2] < F[p1]):
                    p = p2
                else:
                    p = p1 if rng.random() > 0.5 else p2
                
                # Mutation: Gaussian
                child = X[p] + 0.1 * (hi - lo) * rng.standard_normal(self.dim)
                child = np.clip(child, lo, hi)
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1
                if used >= self.budget:
                    break
            
            if used >= self.budget:
                break

            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            rank = non_dominated_sort(F_comb)
            
            # Select next population
            next_X = []
            next_F = []
            i = 0
            while len(next_X) < pop_size and i < len(F_comb):
                # Get indices of current front
                front_indices = np.where(rank == i)[0]
                
                if len(next_X) + len(front_indices) <= pop_size:
                    for idx in front_indices:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                else:
                    # Need to select by crowding distance
                    cd = crowding_distance(F_comb, i, front_indices)
                    # Sort front_indices by cd descending
                    sorted_front = front_indices[np.argsort(cd)[::-1]]
                    need = pop_size - len(next_X)
                    for idx in sorted_front[:need]:
                        next_X.append(X_comb[idx])
                        next_F.append(F_comb[idx])
                    break
                
                i += 1
            
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
