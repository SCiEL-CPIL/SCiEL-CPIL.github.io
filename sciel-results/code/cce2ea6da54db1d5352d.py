import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        # Safety margin: The evaluator kills at 330s, task limit is 300s.
        # We stop at 295s to be safe.
        deadline = t_start + 295.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size 20 allows 10 generations (20 evals/gen for offspring + selection from 40)
        # Total evals: 20 (init) + 10 * 20 (offspring) = 220? 
        # Wait, budget is 200.
        # Init: 20.
        # Gen 1: 20 offspring. Total 40. Select 20.
        # ...
        # Gen 9: 20 offspring. Total 20 + 9*20 = 200.
        # So we can do exactly 9 full generations after init if we strictly stick to budget.
        # Or we can do 10 generations if we are careful.
        # Let's check:
        # Init: 20 used. Remaining: 180.
        # Gen 1: 20 used. Remaining: 160.
        # ...
        # Gen 9: 20 used. Remaining: 0.
        # So 9 generations of DE + Init = 200 evals.
        
        pop_size = 20
        
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
                    # Check if i dominates j
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    # Check if j dominates i
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
            
            # Map indices to local 0..n-1 for easier handling? 
            # Or just work with global indices and F.
            # Let's work with the subset F_sub = F[indices]
            F_sub = F[indices]
            cd = [0.0] * n
            m = self.n_obj
            
            for k in range(m):
                # Sort indices by k-th objective
                sorted_idx = sorted(range(n), key=lambda i: F_sub[i][k])
                
                # Extreme points get infinity
                cd[sorted_idx[0]] = float('inf')
                cd[sorted_idx[-1]] = float('inf')
                
                # Normalize the range
                f_min = F_sub[sorted_idx[0]][k]
                f_max = F_sub[sorted_idx[-1]][k]
                f_range = f_max - f_min
                
                if f_range == 0:
                    continue
                    
                for i in range(1, n - 1):
                    # Distance in objective k
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
                # Pick 3 distinct parents
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Weight F (0.5 to 1.0 is common for DE)
                F_weight = 0.5 + 0.5 * np.random.rand()
                
                # Mutation
                v = X[r1] + F_weight * (X[r2] - X[r3])
                
                # Crossover (Binomial)
                C_rate = 0.5
                j_rand = np.random.randint(n_dim)
                mask = np.random.rand(n_dim) < C_rate
                mask[j_rand] = True # Ensure at least one gene from v
                
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
            
            # Combine parents and offspring
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated Sort
            fronts, ranks = non_dominated_sort(F_comb)
            
            # Selection: Pick best pop_size individuals
            selected_idx = []
            for front in fronts:
                if len(selected_idx) + len(front) <= pop_size:
                    selected_idx.extend(front)
                else:
                    remaining = pop_size - len(selected_idx)
                    if remaining <= 0:
                        break
                    
                    # We need to pick 'remaining' from 'front'
                    # Use crowding distance
                    cd = crowding_distance(F_comb, front)
                    # Sort front by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd[front.index(i)], reverse=True)
                    selected_idx.extend(sorted_front[:remaining])
                    break
            
            if not selected_idx:
                break
                
            X = X_comb[selected_idx]
            F = F_comb[selected_idx]
            
        return X
