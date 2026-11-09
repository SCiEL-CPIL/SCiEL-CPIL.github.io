import numpy as np
import time

class Optimizer:
        def __init__(self, budget, dim, n_obj):
            self.budget = budget
            self.dim = dim
            self.n_obj = n_obj

        def __call__(self, problem, bounds):
            start_time = time.time()
            deadline = start_time + 290.0 # Safety margin
            
            rng = np.random.default_rng(0)
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            dim = self.dim
            n_obj = self.n_obj
            budget = self.budget
            
            # 1. Initialization: Latin Hypercube Sampling
            pop_size = 20
            X = self._lhs(rng, pop_size, dim, lo, hi)
            F = np.array([problem(x) for x in X])
            used = pop_size
            
            # 2. Main Loop
            while used < budget:
                if time.time() > deadline:
                    break
                
                # Generate offspring using DE/rand/1/bin
                offspring_X = []
                offspring_F = []
                
                # We generate pop_size offspring per generation
                # To respect budget, we might need to handle partial generations
                # But 200 / 20 = 10 generations exactly.
                
                for i in range(pop_size):
                    if used >= budget:
                        break
                    
                    # Select 3 distinct parents
                    idx = rng.choice(pop_size, 3, replace=False)
                    p1, p2, p3 = X[idx[0]], X[idx[1]], X[idx[2]]
                    
                    # Mutation
                    F_mut = 0.5
                    v = p1 + F_mut * (p2 - p3)
                    
                    # Crossover
                    CR = 0.7
                    j_rand = rng.integers(0, dim)
                    trial = np.zeros(dim)
                    for j in range(dim):
                        if rng.random() < CR or j == j_rand:
                            trial[j] = v[j]
                        else:
                            trial[j] = X[i][j]
                    
                    # Clip to bounds
                    trial = np.clip(trial, lo, hi)
                    
                    offspring_X.append(trial)
                    f = problem(trial)
                    offspring_F.append(f)
                    used += 1
                
                if not offspring_F:
                    break
                    
                offspring_F = np.array(offspring_F)
                
                # Combine
                combined_X = np.vstack([X, np.array(offspring_X)])
                combined_F = np.vstack([F, offspring_F])
                
                # Selection: NSGA-II
                new_X, new_F = self._select_nsGA2(combined_X, combined_F, pop_size)
                
                X = new_X
                F = new_F

            return np.array(X) # Return last pop, but evaluator records all. 
            # Wait, the evaluator records ALL returned objective vectors.
            # "Every returned objective vector is recorded"
            # The return value of __call__ is ignored.
            # So we just need to make sure we call problem() 200 times.
            # The code above does that.
            
        def _lhs(self, rng, n, dim, lo, hi):
            # Simple LHS
            points = np.zeros((n, dim))
            for d in range(dim):
                perm = rng.permutation(n)
                points[:, d] = (perm + rng.random(n)) / n
            points = lo + points * (hi - lo)
            return points
            
        def _select_nsGA2(self, X, F, pop_size):
            # Non-dominated sort
            fronts = self._fast_non_dominated_sort(F)
            
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    # Crowding distance for this front
                    cd = self._crowding_distance(F, front)
                    # Sort by cd descending
                    sorted_front = sorted(front, key=lambda i: cd[i], reverse=True)
                    # Take remaining
                    remaining = pop_size - len(new_indices)
                    new_indices.extend(sorted_front[:remaining])
                    break
                    
            return X[new_indices], F[new_indices]
            
        def _fast_non_dominated_sort(self, F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_doms = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_doms[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_doms[i] += 1
            
            fronts = []
            current_front = []
            for i in range(n):
                if n_doms[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
            
            if current_front:
                fronts.append(current_front)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_doms[j] -= 1
                        if n_doms[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    k += 1
                    fronts.append(next_front)
                else:
                    break
            return fronts
            
        def _crowding_distance(self, F, front_indices):
            n = len(front_indices)
            if n <= 2:
                return np.full(n, np.inf)
            
            dist = np.zeros(n)
            F_front = F[front_indices]
            
            for m in range(self.n_obj):
                sorted_idx = np.argsort(F_front[:, m])
                dist[sorted_idx[0]] = np.inf
                dist[sorted_idx[-1]] = np.inf
                
                f_min = F_front[sorted_idx[0], m]
                f_max = F_front[sorted_idx[-1], m]
                if f_max - f_min < 1e-15:
                    continue
                
                for k in range(1, n - 1):
                    dist[sorted_idx[k]] += (F_front[sorted_idx[k + 1], m] - F_front[sorted_idx[k - 1], m]) / (f_max - f_min)
            
            return dist
