import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        N = 20
        
        # Initialize population
        X = lo + range_ * np.random.rand(N, dim)
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Non-dominated sorting
        def non_dominated_sort(F):
            m = len(F)
            S = [set() for _ in range(m)]
            n = [0] * m
            rank = [0] * m
            fronts = []
            front = set()
            
            for i in range(m):
                for j in range(i + 1, m):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].add(j)
                        n[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].add(i)
                        n[i] += 1
                
                if n[i] == 0:
                    rank[i] = 0
                    front.add(i)
            
            fronts.append(front)
            k = 0
            while front:
                next_front = set()
                for i in front:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            rank[j] = k + 1
                            next_front.add(j)
                k += 1
                front = next_front
                if front:
                    fronts.append(front)
            
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            m = len(indices)
            if m <= 2:
                return [float('inf')] * m
            
            cd = np.zeros(m)
            for k in range(n_obj):
                obj_vals = np.array([F[idx, k] for idx in indices])
                sorted_indices = np.argsort(obj_vals)
                min_val = obj_vals[sorted_indices[0]]
                max_val = obj_vals[sorted_indices[-1]]
                if max_val == min_val:
                    continue
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                for i in range(1, m - 1):
                    cd[sorted_indices[i]] += (obj_vals[sorted_indices[i + 1]] - obj_vals[sorted_indices[i - 1]]) / (max_val - min_val)
            return cd

        # DE Parameters
        F_mut = 0.8
        CR = 0.5
        
        while used + N <= budget:
            Q_X = []
            for i in range(N):
                # Choose 3 distinct random individuals different from i
                r1, r2, r3 = np.random.choice([j for j in range(N) if j != i], size=3, replace=False)
                
                # Mutation: v = x_r1 + F * (x_r2 - x_r3)
                v = X[r1] + F_mut * (X[r2] - X[r3])
                
                # Crossover
                j_rand = np.random.randint(dim)
                trial = X[i].copy()
                for j in range(dim):
                    if np.random.rand() < CR or j == j_rand:
                        trial[j] = v[j]
                
                # Clip to bounds
                trial = np.clip(trial, lo, hi)
                Q_X.append(trial)
            
            Q_F = np.array([problem(x) for x in Q_X])
            used += N
            
            # Combine
            R_X = np.vstack([X, Q_X])
            R_F = np.vstack([F, Q_F])
            
            # Non-dominated sorting
            fronts = non_dominated_sort(R_F)
            
            # Select next population
            P_next_X = []
            P_next_F = []
            
            for front in fronts:
                if len(P_next_X) + len(front) <= N:
                    for idx in front:
                        P_next_X.append(R_X[idx])
                        P_next_F.append(R_F[idx])
                else:
                    remaining = N - len(P_next_X)
                    cd = crowding_distance(R_F, list(front))
                    sorted_front = sorted(front, key=lambda i: cd[list(front).index(i)], reverse=True)
                    for idx in sorted_front[:remaining]:
                        P_next_X.append(R_X[idx])
                        P_next_F.append(R_F[idx])
                    break
            
            X = np.array(P_next_X)
            F = np.array(P_next_F)
            
            # If we have remaining budget, we can do single evaluations if N > 1
            # But standard DE works in generations. 
            # If budget is not multiple of N, we can just stop or do random perturbations.
            # Given 200 / 20 = 10, it fits perfectly.
            
        return X
