import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = 280.0 # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size
        N = 20
        
        # Initial Population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Fast Non-Dominated Sorting
        def fast_non_dominated_sort(F):
            M = F.shape[0]
            S = [[] for _ in range(M)]
            n = [0] * M
            rank = [0] * M
            front = [0]
            for p in range(M):
                for q in range(M):
                    if p == q: continue
                    if F[p,0] <= F[q,0] and F[p,1] <= F[q,1] and (F[p,0] < F[q,0] or F[p,1] < F[q,1]):
                        S[p].append(q)
                    elif F[q,0] <= F[p,0] and F[q,1] <= F[p,1] and (F[q,0] < F[p,0] or F[q,1] < F[p,1]):
                        n[p] += 1
                if n[p] == 0:
                    rank[p] = 0
                    front[0] += 1
            i = 0
            while front[i] > 0:
                Q = []
                for p in range(M):
                    if rank[p] == i:
                        Q.append(p)
                        for q in S[p]:
                            n[q] -= 1
                            if n[q] == 0:
                                rank[q] = i + 1
                                front[i+1] += 1
                i += 1
            # Return list of fronts
            fronts = [[] for _ in range(i)]
            for p in range(M):
                fronts[rank[p]].append(p)
            return fronts

        # Helper: Crowding Distance
        def crowding_distance(F, indices):
            if len(indices) == 0:
                return []
            dist = np.zeros(len(indices))
            if len(indices) == 1:
                dist[0] = np.inf
                return dist
            for m in range(n_obj):
                f_vals = F[indices, m]
                min_val = np.min(f_vals)
                max_val = np.max(f_vals)
                if max_val == min_val:
                    continue
                sort_idx = np.argsort(f_vals)
                dist_sorted = np.zeros(len(indices))
                dist_sorted[0] = np.inf
                dist_sorted[-1] = np.inf
                for k in range(1, len(indices)-1):
                    dist_sorted[k] = (f_vals[sort_idx[k+1]] - f_vals[sort_idx[k-1]]) / (max_val - min_val)
                # Map back to original indices
                for i, idx in enumerate(sort_idx):
                    dist[idx] += dist_sorted[i]
            return dist

        # Helper: SBX Crossover
        def sbx_crossover(x1, x2):
            u = np.random.random(dim)
            c1 = np.zeros(dim)
            c2 = np.zeros(dim)
            for i in range(dim):
                if u[i] <= 0.5:
                    beta = (2 * u[i]) ** (1 / (15 + 1))
                else:
                    beta = (1 / (2 * (1 - u[i]))) ** (1 / (15 + 1))
                c1[i] = 0.5 * ((1 + beta) * x1[i] + (1 - beta) * x2[i])
                c2[i] = 0.5 * ((1 - beta) * x1[i] + (1 + beta) * x2[i])
            return c1, c2

        # Helper: PM Mutation
        def pm_mutation(x):
            y = x.copy()
            for i in range(dim):
                if np.random.random() < 0.5: # Mutation probability 1/dim usually, but let's use 1.0 for simplicity? No, 1/dim is standard.
                    # Actually, standard PM mutates each variable with prob 1/dim.
                    # Let's use prob = 1.0 for simplicity in small dim? No, let's stick to standard.
                    # If we use 1/dim, with dim=4, prob=0.25.
                    pass 
            # Let's implement proper PM
            for i in range(dim):
                if np.random.random() < 1.0 / dim:
                    delta1 = (y[i] - lo[i]) / (hi[i] - lo[i])
                    delta2 = (hi[i] - y[i]) / (hi[i] - lo[i])
                    r = np.random.random()
                    if r < 0.5:
                        xy = (2 * r + (1 - 2 * r) * delta1 ** (20 + 1)) ** (1 / (20 + 1))
                        deltaq = (xy - 1) * (hi[i] - lo[i])
                    else:
                        xy = (2 * (1 - r) + (2 * (r - 0.5)) * delta2 ** (20 + 1)) ** (1 / (20 + 1))
                        deltaq = (1 - xy) * (hi[i] - lo[i])
                    y[i] += deltaq
            return np.clip(y, lo, hi)

        while used < budget:
            if time.time() > deadline:
                break
            
            # Selection: Binary Tournament
            parents = []
            while len(parents) < N:
                p1_idx = np.random.randint(N)
                p2_idx = np.random.randint(N)
                
                # Get ranks
                # We need to calculate ranks for current population X, F
                # To save time, we can calculate ranks once per generation
                pass
            
            # Recalculate ranks for current pop
            fronts = fast_non_dominated_sort(F)
            ranks = np.zeros(N, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = i
            
            # Crowding distances for each front
            crowds = np.zeros(N)
            for front in fronts:
                if len(front) > 0:
                    cd = crowding_distance(F, front)
                    for j, idx in enumerate(front):
                        crowds[idx] = cd[j]

            # Generate Offspring
            offspring_X = []
            while len(offspring_X) < N:
                # Tournament
                p1 = np.random.randint(N)
                p2 = np.random.randint(N)
                
                def better(a, b):
                    if ranks[a] < ranks[b]: return a
                    if ranks[b] < ranks[a]: return b
                    if crowds[a] > crowds[b]: return a
                    return b
                
                p = better(p1, p2)
                
                # Crossover
                if np.random.random() < 0.9: # Crossover prob
                    x1 = X[p]
                    # Pick another parent
                    p3 = np.random.randint(N)
                    p4 = np.random.randint(N)
                    p_q = better(p3, p4)
                    x2 = X[p_q]
                    
                    c1, c2 = sbx_crossover(x1, x2)
                    
                    c1 = pm_mutation(c1)
                    c2 = pm_mutation(c2)
                    
                    offspring_X.append(c1)
                    if len(offspring_X) < N:
                        offspring_X.append(c2)
                else:
                    offspring_X.append(X[p].copy())
            
            # Evaluate Offspring
            F_off = np.array([problem(x) for x in offspring_X])
            used += len(offspring_X)
            
            # Combine
            X_comb = np.vstack([X, np.array(offspring_X)])
            F_comb = np.vstack([F, F_off])
            
            # Sort and Select
            fronts_comb = fast_non_dominated_sort(F_comb)
            
            X_new = []
            F_new = []
            for front in fronts_comb:
                if len(X_new) + len(front) <= N:
                    # Add all
                    for idx in front:
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                else:
                    # Sort by crowding distance
                    cd = crowding_distance(F_comb, front)
                    front_sorted = sorted(range(len(front)), key=lambda k: cd[k], reverse=True)
                    needed = N - len(X_new)
                    for k in front_sorted[:needed]:
                        idx = front[k]
                        X_new.append(X_comb[idx])
                        F_new.append(F_comb[idx])
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
            
        return X
