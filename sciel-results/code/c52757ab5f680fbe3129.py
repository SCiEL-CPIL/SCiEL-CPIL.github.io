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
        
        N = 50  # Population size
        # We can do budget // N generations
        # But we must not exceed budget.
        
        # Initialize population
        # x0 in [0, 0.2], others in [0, 1]
        X = np.zeros((N, self.dim))
        X[:, 0] = rng.uniform(0, 0.2, N)
        for i in range(1, self.dim):
            X[:, i] = rng.uniform(lo[i], hi[i], N)
            
        F = np.array([problem(x) for x in X])
        used = N
        
        # Non-dominated sorting
        def non_dominated_sort(F):
            n = len(F)
            S = [set() for _ in range(n)]  # Dominated by i
            n_dominates = np.zeros(n, dtype=int)
            front = [set()]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[i] += 1
                        
            for i in range(n):
                if n_dominates[i] == 0:
                    front[0].add(i)
                    
            k = 0
            while len(front[k]) > 0:
                next_front = set()
                for i in front[k]:
                    for j in S[i]:
                        n_dominates[j] -= 1
                        if n_dominates[j] == 0:
                            next_front.add(j)
                k += 1
                front.append(next_front)
                
            return front[:-1] # Remove empty last front

        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return np.full(len(front_indices), np.inf)
            
            dist = np.zeros(len(front_indices))
            n_obj = F.shape[1]
            for m in range(n_obj):
                fm = F[front_indices, m]
                min_f = fm.min()
                max_f = fm.max()
                if max_f == min_f:
                    dist[:] = np.inf
                    continue
                # Sort indices by objective m
                sorted_idx = np.argsort(fm)
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, len(front_indices)-1):
                    dist[sorted_idx[i]] += (fm[sorted_idx[i+1]] - fm[sorted_idx[i-1]]) / (max_f - min_f)
            return dist

        while used < self.budget:
            # Create offspring
            Q_X = np.zeros_like(X)
            for i in range(N):
                # Tournament selection
                parents = []
                for _ in range(2):
                    t1 = rng.integers(N)
                    t2 = rng.integers(N)
                    # Simple tournament: pick the one with better rank
                    # We need ranks. Let's compute ranks for current P
                    # To save time, we can approximate or just pick randomly from non-dominated
                    # For simplicity, let's pick randomly from the current population
                    # A better way: use the fronts from the last step
                    # But we need to update fronts each generation.
                    
                    # Let's just pick two random parents and keep the better one
                    # "Better" in MOO is tricky. Let's use a simple approach:
                    # Pick 2 random, if one dominates the other, pick it. Else pick random.
                    p1 = X[t1]
                    p2 = X[t2]
                    f1 = F[t1]
                    f2 = F[t2]
                    if np.all(f1 <= f2) and np.any(f1 < f2):
                        parents.append(p1)
                    elif np.all(f2 <= f1) and np.any(f2 < f1):
                        parents.append(p2)
                    else:
                        parents.append(p1 if rng.random() > 0.5 else p2)
                
                p1 = parents[0]
                p2 = parents[1]
                
                # Crossover (BLX)
                child = np.zeros(self.dim)
                for j in range(self.dim):
                    if rng.random() < 0.5:
                        child[j] = p1[j]
                    else:
                        child[j] = p2[j]
                
                # Mutation
                if rng.random() < 0.5: # Mutate each variable with prob 0.5
                    for j in range(self.dim):
                        if rng.random() < 0.5:
                            child[j] += 0.1 * (hi[j] - lo[j]) * rng.standard_normal()
                            child[j] = np.clip(child[j], lo[j], hi[j])
                
                Q_X[i] = child
            
            Q_F = np.array([problem(x) for x in Q_X])
            used += N
            
            # Combine
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # Sort
            fronts = non_dominated_sort(R_F)
            new_X = np.zeros((N, self.dim))
            new_F = np.zeros((N, self.n_obj))
            count = 0
            for front in fronts:
                if count + len(front) <= N:
                    for idx in front:
                        new_X[count] = R_X[idx]
                        new_F[count] = R_F[idx]
                        count += 1
                else:
                    # Need crowding distance
                    dist = crowding_distance(R_F, list(front))
                    sorted_front = list(front)[np.argsort(-dist)]
                    for idx in sorted_front:
                        if count < N:
                            new_X[count] = R_X[idx]
                            new_F[count] = R_F[idx]
                            count += 1
                    break
            
            X = new_X
            F = new_F
        
        return X
