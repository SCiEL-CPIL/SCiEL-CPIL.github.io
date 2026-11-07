import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Use a smaller population to allow more generations and faster convergence
        # Budget 200. Init 10. 19 gens * 10 = 190. Total 200.
        pop_size = 10
        
        # 1. Initialization
        X = np.random.uniform(lo, hi, size=(pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            for i in current_front:
                rank[i] = 0
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, indices):
            n = len(indices)
            if n <= 2:
                return [float('inf')] * n
            
            F_sub = F[indices]
            cd = [0.0] * n
            m = self.n_obj
            
            for k in range(m):
                sorted_idx = sorted(range(n), key=lambda i: F_sub[i][k])
                
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                f_min = F_sub[sorted_idx[0]][k]
                f_max = F_sub[sorted_idx[-1]][k]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    dist = (F_sub[sorted_idx[i+1]][k] - F_sub[sorted_idx[i-1]][k]) / f_range
                    cd[sorted_idx[i]] += dist
            
            return cd

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate offspring
            offspring_X = []
            offspring_F = []
            
            for i in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # DE/rand/1
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                F_weight = 0.5 + 0.5 * np.random.rand()
                
                v = X[r1] + F_weight * (X[r2] - X[r3])
                
                C_rate = 0.5
                j_rand = np.random.randint(n_dim)
                mask = np.random.rand(n_dim) < C_rate
                mask[j_rand] = True
                
                child = np.where(mask, v, X[r1])
                child = np.clip(child, lo, hi)
                
                fv = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fv)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            fronts, ranks = non_dominated_sort(F_comb)
            
            selected_idx = []
            for front in fronts:
                if len(selected_idx) + len(front) <= pop_size:
                    selected_idx.extend(front)
                else:
                    remaining = pop_size - len(selected_idx)
                    if remaining <= 0:
                        break
                    
                    cd = crowding_distance(F_comb, front)
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    selected_idx.extend(sorted_front[:remaining])
                    break
            
            if not selected_idx:
                break
                
            X = X_comb[selected_idx]
            F = F_comb[selected_idx]
            
        return X
