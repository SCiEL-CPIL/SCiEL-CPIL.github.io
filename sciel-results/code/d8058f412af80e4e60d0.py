import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin
        
        n = self.dim
        budget = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # Population size
        pop_size = 50
        if budget < pop_size * 2:
            pop_size = max(10, budget // 2)
            
        # Initialization
        # x0 is uniform [0, 1]
        # xi for i>0: mix of uniform and normal around 0
        X = np.zeros((pop_size, n))
        X[:, 0] = np.random.uniform(lo[0], hi[0], pop_size)
        
        # For i > 0, 50% uniform, 50% normal(0, 0.5) clipped
        mask = np.random.rand(pop_size) < 0.5
        for i in range(1, n):
            X[mask, i] = np.random.uniform(lo[i], hi[i], np.sum(mask))
            X[~mask, i] = np.clip(np.random.normal(0, 0.5, np.sum(~mask)), lo[i], hi[i])
            
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # NSGA-II helpers
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [-1] * n
            fronts = []
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[i] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[j] += 1
                if n_dom[i] == 0:
                    rank[i] = 0
                    fronts.append([i])
                    
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
            return fronts, rank

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.full(len(indices), np.inf)
            
            m = self.n_obj
            n = len(indices)
            cd = np.zeros(n)
            
            for k in range(m):
                # Sort by k-th objective
                sorted_idx = np.argsort(F[indices, k])
                obj_vals = F[indices, k]
                min_val = obj_vals[sorted_idx[0]]
                max_val = obj_vals[sorted_idx[-1]]
                
                if max_val == min_val:
                    continue
                    
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (obj_vals[sorted_idx[i + 1]] - obj_vals[sorted_idx[i - 1]]) / (max_val - min_val)
                    
            return cd

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
                
            # Create offspring
            offspring_X = np.zeros((pop_size, n))
            offspring_F = np.zeros((pop_size, self.n_obj))
            
            # Generate pairs
            # Use tournament selection
            def tournament_select(X, F, size=2):
                idxs = np.random.choice(len(F), size)
                best = idxs[0]
                for i in idxs[1:]:
                    # Check domination
                    if np.all(F[i] <= F[best]) and np.any(F[i] < F[best]):
                        best = i
                    elif not (np.all(F[best] <= F[i]) and np.any(F[best] < F[i])):
                        # If neither dominates, pick random or based on rank?
                        # Simple: keep current best or random
                        if np.random.rand() < 0.5:
                            best = i
                return best

            for i in range(pop_size):
                # Select parents
                p1_idx = tournament_select(X, F)
                p2_idx = tournament_select(X, F)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # SBX Crossover
                if np.random.rand() < 0.9:
                    beta = np.random.rand(n)
                    beta = np.where(beta < 0.5, (2 * beta) ** (1/21), (2 * (1 - beta)) ** (1/21))
                    
                    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
                    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
                else:
                    c1 = p1.copy()
                    c2 = p2.copy()
                    
                # Polynomial Mutation
                def poly_mutation(x):
                    x_mut = x.copy()
                    for j in range(n):
                        if np.random.rand() < 0.1: # Mutation rate 1/n? No, 1/n is standard, but 0.1 is high. Let's use 1/n.
                            pass
                    # Standard polynomial mutation
                    for j in range(n):
                        if np.random.rand() < 1.0 / n:
                            delta1 = (x[j] - lo[j]) / range_[j]
                            delta2 = (hi[j] - x[j]) / range_[j]
                            u = np.random.rand()
                            if u < 0.5:
                                xy = (2 * u + (1 - 2 * u) * (delta1 ** (21))) ** (1/20)
                                x_mut[j] = x[j] - (hi[j] - lo[j]) * (xy - 1)
                            else:
                                xy = (2 * (1 - u) + (2 * u - 1) * (delta2 ** (21))) ** (1/20)
                                x_mut[j] = x[j] + (hi[j] - lo[j]) * (xy - 1)
                    return np.clip(x_mut, lo, hi)
                
                c1 = poly_mutation(c1)
                c2 = poly_mutation(c2)
                
                offspring_X[i] = c1
                offspring_F[i] = problem(c1)
                used += 1
                
                if i < pop_size - 1:
                    offspring_X[i+1] = c2
                    offspring_F[i+1] = problem(c2)
                    used += 1
                    
                if used >= budget:
                    break
                    
            if used >= budget:
                break
                
            # Combine and Select
            X_all = np.vstack([X, offspring_X])
            F_all = np.vstack([F, offspring_F])
            
            fronts, rank = non_dominated_sort(F_all)
            
            next_X = []
            next_F = []
            curr_front_idx = 0
            while len(next_X) < pop_size:
                if curr_front_idx >= len(fronts):
                    break
                front = fronts[curr_front_idx]
                if len(next_X) + len(front) <= pop_size:
                    next_X.extend(X_all[front])
                    next_F.extend(F_all[front])
                else:
                    # Crowding distance
                    cd = crowding_distance(F_all, front)
                    sorted_front = np.argsort(-cd)
                    needed = pop_size - len(next_X)
                    selected = [front[i] for i in sorted_front[:needed]]
                    next_X.extend(X_all[selected])
                    next_F.extend(F_all[selected])
                curr_front_idx += 1
                
            if len(next_X) == 0:
                break
                
            X = np.array(next_X)
            F = np.array(next_F)
            
        return X
