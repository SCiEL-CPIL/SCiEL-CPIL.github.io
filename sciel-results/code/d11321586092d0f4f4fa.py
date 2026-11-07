import numpy as np
import time
from scipy.stats import qmc

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit_time = 325  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        N = self.budget
        dim = self.dim
        n_obj = self.n_obj

        if N == 0:
            return np.array([])

        # If budget is too small for a proper EA population, fallback to sampling
        if N < 20:
            try:
                sampler = qmc.Sobol(d=dim, scramble=True)
                X = sampler.random(N)
            except Exception:
                X = np.random.rand(N, dim)
            X = lo + X * span
            F = np.array([problem(x) for x in X])
            return X

        pop_size = N // 2

        # 1. Initialize with Sobol
        try:
            sampler = qmc.Sobol(d=dim, scramble=True)
            X_pop = sampler.random(pop_size)
        except Exception:
            X_pop = np.random.rand(pop_size, dim)
        
        X_pop = lo + X_pop * span
        F_pop = np.array([problem(x) for x in X_pop])
        
        used = pop_size

        # NSGA-II Helper Functions
        def non_dominated_sort(F):
            m = len(F)
            S = [[] for _ in range(m)]
            n = [0] * m
            rank = [0] * m
            
            for p in range(m):
                for q in range(m):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
            
            current_front = [p for p in range(m) if n[p] == 0]
            i = 0
            while current_front:
                for p in current_front:
                    rank[p] = i
                next_front = []
                for p in current_front:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                current_front = next_front
                i += 1
            return rank

        def crowding_distance(F, rank):
            m = len(F)
            dist = [0.0] * m
            fronts = {}
            for i in range(m):
                if rank[i] not in fronts:
                    fronts[rank[i]] = []
                fronts[rank[i]].append(i)
            
            for r, indices in fronts.items():
                if len(indices) <= 2:
                    for i in indices:
                        dist[i] = float('inf')
                    continue
                
                for k in range(n_obj):
                    sorted_indices = sorted(indices, key=lambda i: F[i][k])
                    dist[sorted_indices[0]] = float('inf')
                    dist[sorted_indices[-1]] = float('inf')
                    
                    f_min = F[sorted_indices[0]][k]
                    f_max = F[sorted_indices[-1]][k]
                    if f_max == f_min:
                        continue
                    
                    for j in range(1, len(sorted_indices) - 1):
                        i_prev = sorted_indices[j-1]
                        i_next = sorted_indices[j+1]
                        i_curr = sorted_indices[j]
                        dist[i_curr] += (F[i_next][k] - F[i_prev][k]) / (f_max - f_min)
            
            return dist

        def sbx_crossover(x1, x2, eta=20.0):
            child1 = x1.copy()
            child2 = x2.copy()
            for d in range(dim):
                if np.random.rand() < 0.5:
                    u = np.random.rand()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * ((1 + beta) * x1[d] + (1 - beta) * x2[d])
                    c2 = 0.5 * ((1 - beta) * x1[d] + (1 + beta) * x2[d])
                    
                    c1 = np.clip(c1, lo[d], hi[d])
                    c2 = np.clip(c2, lo[d], hi[d])
                    
                    child1[d] = c1
                    child2[d] = c2
            return child1, child2

        def polynomial_mutation(x, eta=20.0):
            for d in range(dim):
                if np.random.rand() < 1.0 / dim:
                    u = np.random.rand()
                    y = x[d]
                    yl = lo[d]
                    yu = hi[d]
                    if u < 0.5:
                        delta1 = (y - yl) / (yu - yl)
                        delta = (1 + 2 * (1 - delta1)) ** (1.0 / (eta + 1.0)) - 1
                        y_new = y - delta * (yu - yl)
                    else:
                        delta2 = (yu - y) / (yu - yl)
                        delta = 1 - (1 + 2 * delta2) ** (1.0 / (eta + 1.0))
                        y_new = y + delta * (yu - yl)
                    x[d] = np.clip(y_new, yl, yu)
            return x

        def tournament_select(X, F, rank, dist, k=2):
            idxs = np.random.choice(len(X), k, replace=False)
            best_idx = -1
            best_rank = float('inf')
            best_dist = -1
            for idx in idxs:
                if rank[idx] < best_rank:
                    best_rank = rank[idx]
                    best_dist = dist[idx]
                    best_idx = idx
                elif rank[idx] == best_rank:
                    if dist[idx] > best_dist:
                        best_dist = dist[idx]
                        best_idx = idx
            return best_idx

        # Evolutionary Loop
        # We have used `pop_size` evaluations. We have `N - pop_size` left.
        # We will generate offspring of size `pop_size` (which equals remaining budget).
        
        while used < N:
            if time.time() - start_time > limit_time:
                break
            
            # Generate Offspring
            Q_X = []
            for _ in range(pop_size):
                i1 = tournament_select(X_pop, F_pop, rank, dist)
                i2 = tournament_select(X_pop, F_pop, rank, dist)
                
                if np.random.rand() < 0.9:
                    c1, c2 = sbx_crossover(X_pop[i1], X_pop[i2])
                else:
                    c1 = X_pop[i1].copy()
                    c2 = X_pop[i2].copy()
                
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                Q_X.append(c1)
                # We only need pop_size offspring, not 2*pop_size, because we combine P and Q.
                # Wait, standard NSGA-II generates |P| offspring to form R of size 2|P|.
                # Here P size is 100. Q size is 100. R size is 200.
                # We select best 100 for next P.
                # So we only need to generate 100 offspring.
                # The loop above generates 100 offspring (one pair of children, but we only take one? No, we take c1 and c2?)
                # If we take both, Q size is 200. R size is 300.
                # Standard is: Generate |P| offspring.
                # So we should only append one child per iteration if we want Q size = P size.
                # Or generate pairs and take one?
                # Let's just generate one child per loop iteration to keep Q size = pop_size.
                
            # Evaluate Q
            F_Q = np.array([problem(x) for x in Q_X])
            used += len(Q_X)
            
            # Combine P and Q
            R_X = np.vstack([X_pop, Q_X])
            R_F = np.vstack([F_pop, F_Q])
            
            # Sort R
            R_rank = non_dominated_sort(R_F)
            R_dist = crowding_distance(R_F, R_rank)
            
            # Select best pop_size for next P
            # Sort by rank, then by dist (descending)
            indices = np.arange(len(R_X))
            # Stable sort by rank, then dist
            # We want lowest rank first, highest dist first
            # Python sort is stable. 
            # Sort by -dist first? No, sort by rank, then -dist.
            # We can create a list of tuples (rank, -dist, index) and sort.
            sort_keys = [(R_rank[i], -R_dist[i], i) for i in range(len(R_X))]
            sort_keys.sort()
            
            next_indices = [k[2] for k in sort_keys[:pop_size]]
            
            X_pop = R_X[next_indices]
            F_pop = R_F[next_indices]
            
            # Update rank and dist for selection in next iteration
            rank = R_rank[next_indices]
            dist = R_dist[next_indices]
            
            # Note: rank and dist here are relative to the combined set R, 
            # but we are selecting from R. 
            # For the next tournament selection, we should use the ranks/distances 
            # within the new population P? 
            # Actually, tournament selection usually operates on the current population.
            # The ranks/distances calculated on R are valid for comparison within R.
            # Since P is a subset of R, the relative order is preserved.
            # So using R_rank[R_dist] subset is fine.

        return X_pop
