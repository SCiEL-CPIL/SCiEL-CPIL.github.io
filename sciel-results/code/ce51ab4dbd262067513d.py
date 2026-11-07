import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        N = 40
        if N * 2 > self.budget:
            N = max(10, self.budget // 4)
            
        # Initialize
        X = lo + (hi - lo) * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper functions
        def non_dominated_sort(F):
            n = len(F)
            ranks = np.zeros(n, dtype=int)
            S = [[] for _ in range(n)]
            n_dominates = np.zeros(n, dtype=int)
            fronts = []
            front_index = 0
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dominates[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dominates[j] += 1
                        S[j].append(i)
                        
            # This is O(N^2 * N_obj). For N=80, it's manageable.
            # But let's optimize:
            # Actually, the standard algorithm is:
            # 1. Compute domination relationships
            # 2. Find first front
            # 3. Iteratively find next fronts
            
            # Let's use a simpler vectorized approach for small N
            # Dominance matrix
            # D[i,j] is True if i dominates j
            # F shape (N, M)
            # i dominates j if F[i] <= F[j] and F[i] != F[j]
            
            # This might be slow in Python loops.
            # Let's try to vectorize the domination check.
            
            def get_fronts(F):
                n = len(F)
                if n == 0: return []
                ranks = np.zeros(n, dtype=int)
                S = [[] for _ in range(n)]
                n_dominates = np.zeros(n, dtype=int)
                
                # Vectorized domination check
                # F[i] <= F[j] for all obj
                # F[i] < F[j] for at least one obj
                
                # This is tricky to fully vectorize in a single step for all pairs efficiently in pure numpy without loops over i.
                # Given N=80, a loop over i is fine.
                
                for i in range(n):
                    for j in range(n):
                        if i == j: continue
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            S[i].append(j)
                            n_dominates[i] += 1
                        elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            S[j].append(i)
                            n_dominates[j] += 1
                            
                front = []
                for i in range(n):
                    if n_dominates[i] == 0:
                        ranks[i] = 0
                        front.append(i)
                        
                if not front:
                    return [list(range(n))] # Fallback
            
                fronts = []
                current_front = front
                while current_front:
                    fronts.append(current_front)
                    next_front = []
                    for i in current_front:
                        for j in S[i]:
                            n_dominates[j] -= 1
                            if n_dominates[j] == 0:
                                ranks[j] = ranks[i] + 1
                                next_front.append(j)
                    current_front = next_front
                return fronts

            return get_fronts(F)

        def crowding_distance(F, front_indices):
            if len(front_indices) <= 2:
                return np.ones(len(front_indices))
            
            dist = np.zeros(len(front_indices))
            F_sub = F[front_indices]
            m = F_sub.shape[1]
            
            for k in range(m):
                # Sort by k-th objective
                sorted_idx = np.argsort(F_sub[:, k])
                # Map back to original indices in front_indices
                # dist[sorted_idx[0]] = inf
                # dist[sorted_idx[-1]] = inf
                # For i in 1..n-2: dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / (max - min)
                
                if F_sub[sorted_idx[-1], k] == F_sub[sorted_idx[0], k]:
                    continue
                
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                for i in range(1, len(sorted_idx) - 1):
                    dist[sorted_idx[i]] += (F_sub[sorted_idx[i+1], k] - F_sub[sorted_idx[i-1], k]) / (F_sub[sorted_idx[-1], k] - F_sub[sorted_idx[0], k])
                    
            return dist

        def tournament_select(X, F, N, k=2):
            # Select k individuals randomly, pick the best
            # "Best" is lower rank, then higher crowding distance
            # We need ranks and crowding distances for the current population
            
            # This function will be called inside the loop, so we should precompute ranks and crowding for the current population.
            pass

        # Main Loop
        while used + N <= self.budget:
            if time.time() > deadline:
                break
            
            # Get ranks and crowding for current X
            fronts = non_dominated_sort(F)
            ranks = np.zeros(len(X), dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            crowding = np.zeros(len(X))
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, idx in enumerate(front):
                    crowding[idx] = cd[i]
            
            # Generate offspring
            Q_X = np.zeros((N, dim))
            for i in range(N):
                # Select two parents
                p1_idx = self._tournament(ranks, crowding, N)
                p2_idx = self._tournament(ranks, crowding, N)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # SBX Crossover
                c1, c2 = self._sbx(p1, p2, lo, hi, eta=15)
                
                # Mutation
                c1 = self._polymut(c1, lo, hi, eta=20)
                c2 = self._polymut(c2, lo, hi, eta=20)
                
                Q_X[i] = c1
                # We only have budget for N offspring, so we discard c2? 
                # Or we can use a different strategy.
                # Standard NSGA-II generates 2 offspring from 1 pair.
                # If we generate 2N offspring, we need 2N budget.
                # Let's adjust: Generate N offspring.
                # To do this, we can select 1 parent and mutate, or select 2 parents and keep 1 child.
                # Let's keep 1 child from SBX to save budget.
                # Or, we can do:
                # for i in range(N//2):
                #   generate 2 children
                #   put in Q_X[2i], Q_X[2i+1]
                # This is better.
                
                # Let's restructure the offspring generation loop.
                
            # Let's rewrite the offspring generation properly.
            
            Q_X = np.zeros((N, dim))
            for i in range(0, N, 2):
                p1_idx = self._tournament(ranks, crowding, N)
                p2_idx = self._tournament(ranks, crowding, N)
                
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                c1, c2 = self._sbx(p1, p2, lo, hi, eta=15)
                
                c1 = self._polymut(c1, lo, hi, eta=20)
                c2 = self._polymut(c2, lo, hi, eta=20)
                
                Q_X[i] = c1
                if i + 1 < N:
                    Q_X[i+1] = c2
            
            # Evaluate Q_X
            Q_F = np.array([problem(x) for x in Q_X])
            used += N
            
            # Combine
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # Select next X
            # Non-dominated sort on R_F
            R_fronts = non_dominated_sort(R_F)
            
            new_X_indices = []
            for rank, front in enumerate(R_fronts):
                if len(new_X_indices) + len(front) <= N:
                    new_X_indices.extend(front)
                else:
                    # Need to select from this front based on crowding
                    cd = crowding_distance(R_F, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda idx: cd[front.index(idx)], reverse=True)
                    # Wait, crowding_distance returns array aligned with front_indices
                    # So cd[i] corresponds to front[i]
                    # We need to pick the top (N - len(new_X_indices)) from this front
                    remaining = N - len(new_X_indices)
                    # Sort front indices by cd value
                    front_cd_pairs = [(idx, cd[i]) for i, idx in enumerate(front)]
                    front_cd_pairs.sort(key=lambda x: x[1], reverse=True)
                    new_X_indices.extend([idx for idx, cd_val in front_cd_pairs[:remaining]])
                    break
            
            X = R_X[new_X_indices]
            F = R_F[new_X_indices]
            
        return X

    def _tournament(self, ranks, crowding, N, k=2):
        # Select k random indices, return the best
        # Best: lower rank, then higher crowding
        candidates = np.random.randint(0, N, k)
        best_idx = candidates[0]
        for idx in candidates[1:]:
            if ranks[idx] < ranks[best_idx]:
                best_idx = idx
            elif ranks[idx] == ranks[best_idx] and crowding[idx] > crowding[best_idx]:
                best_idx = idx
        return best_idx

    def _sbx(self, x1, x2, lo, hi, eta=15):
        # Simulated Binary Crossover
        c1 = np.copy(x1)
        c2 = np.copy(x2)
        
        for i in range(len(x1)):
            if np.random.rand() < 0.5:
                if abs(x1[i] - x2[i]) > 1e-14:
                    if x1[i] < x2[i]:
                        x1i, x2i = x1[i], x2[i]
                    else:
                        x1i, x2i = x2[i], x1[i]
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1))
                    
                    c1i = 0.5 * ((1 + beta) * x1i + (1 - beta) * x2i)
                    c2i = 0.5 * ((1 - beta) * x1i + (1 + beta) * x2i)
                    
                    # Clip
                    c1i = np.clip(c1i, lo[i], hi[i])
                    c2i = np.clip(c2i, lo[i], hi[i])
                    
                    if x1[i] < x2[i]:
                        c1[i] = c1i
                        c2[i] = c2i
                    else:
                        c1[i] = c2i
                        c2[i] = c1i
        return c1, c2

    def _polymut(self, x, lo, hi, eta=20):
        # Polynomial Mutation
        y = np.copy(x)
        for i in range(len(x)):
            if np.random.rand() < 0.5:
                delta = (hi[i] - lo[i])
                if delta > 1e-14:
                    u = np.random.rand()
                    if u < 0.5:
                        delta1 = (2 * u) ** (1.0 / (eta + 1)) - 1
                    else:
                        delta1 = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                    y[i] = x[i] + delta1 * delta
                    y[i] = np.clip(y[i], lo[i], hi[i])
        return y
