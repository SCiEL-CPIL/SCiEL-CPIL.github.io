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
        budget = self.budget
        n_obj = self.n_obj
        
        pop_size = 50
        eta_c = 15.0
        eta_m = 20.0
        p_c = 0.9
        p_m = 1.0 / dim

        # Initialize with Latin Hypercube Sampling
        X = np.zeros((pop_size, dim))
        for d in range(dim):
            perm = np.random.permutation(pop_size)
            X[:, d] = (perm + np.random.rand(pop_size)) / pop_size
            X[:, d] = lo[d] + X[:, d] * (hi[d] - lo[d])
        
        F = np.array([problem(x) for x in X])
        evals = pop_size

        def fast_non_dominated_sort(F):
            M = len(F)
            S = [set() for _ in range(M)]
            n = [0 for _ in range(M)]
            front = []
            
            for p in range(M):
                for q in range(M):
                    if p == q: continue
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].add(q)
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n[p] += 1
                if n[p] == 0:
                    front.append(p)
            
            fronts = []
            while front:
                fronts.append(front)
                next_front = []
                for p in front:
                    for q in S[p]:
                        n[q] -= 1
                        if n[q] == 0:
                            next_front.append(q)
                front = next_front
            return fronts

        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return [float('inf')] * len(indices)
            
            m = len(indices)
            cd = [0.0] * m
            k = F.shape[1]
            
            for j in range(k):
                sorted_indices = sorted(range(m), key=lambda i: F[indices[i]][j])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                for i in range(1, m - 1):
                    idx_curr = sorted_indices[i]
                    idx_prev = sorted_indices[i-1]
                    idx_next = sorted_indices[i+1]
                    
                    f_curr = F[indices[idx_curr]][j]
                    f_prev = F[indices[idx_prev]][j]
                    f_next = F[indices[idx_next]][j]
                    
                    denom = f_next - f_prev
                    if denom > 1e-12:
                        cd[idx_curr] += (f_curr - f_prev) / denom
                    else:
                        cd[idx_curr] += float('inf')
            return cd

        def sbx_crossover(x1, x2):
            if np.random.rand() < p_c:
                u = np.random.rand(dim)
                beta = np.ones(dim)
                mask1 = u < 0.5
                mask2 = ~mask1
                beta[mask1] = (2 * u[mask1]) ** (1.0 / (eta_c + 1))
                beta[mask2] = (1.0 / (2 * u[mask2])) ** (1.0 / (eta_c + 1))
                
                c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
                c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
                
                c1 = np.clip(c1, lo, hi)
                c2 = np.clip(c2, lo, hi)
                return c1, c2
            else:
                return x1.copy(), x2.copy()

        def polynomial_mutation(x):
            for d in range(dim):
                if np.random.rand() < p_m:
                    u = np.random.rand()
                    delta = 1.0
                    if u < 0.5:
                        delta = (2 * u + 1) ** (1.0 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u) + 1) ** (1.0 / (eta_m + 1))
                    
                    x[d] += delta * (hi[d] - lo[d])
                    x[d] = np.clip(x[d], lo[d], hi[d])
            return x

        while evals < budget:
            X_off = np.zeros((pop_size, dim))
            F_off = np.zeros((pop_size, n_obj))
            
            for i in range(0, pop_size, 2):
                # Simple random selection for speed, relying on diversity from LHS and mutation
                p1 = np.random.randint(0, pop_size)
                p2 = np.random.randint(0, pop_size)
                
                c1, c2 = sbx_crossover(X[p1], X[p2])
                c1 = polynomial_mutation(c1)
                c2 = polynomial_mutation(c2)
                
                X_off[i] = c1
                X_off[i+1] = c2
                F_off[i] = problem(c1)
                F_off[i+1] = problem(c2)
                evals += 2
                
                if time.time() > deadline:
                    break
            
            if time.time() > deadline:
                break

            X_comb = np.vstack([X, X_off])
            F_comb = np.vstack([F, F_off])
            
            fronts = fast_non_dominated_sort(F_comb)
            
            all_keys = []
            for i_f, front in enumerate(fronts):
                if len(front) > 0:
                    cd = crowding_distance(F_comb, front)
                    for j, idx in enumerate(front):
                        all_keys.append((i_f, -cd[j], idx))
                else:
                    for idx in front:
                        all_keys.append((i_f, 0, idx))
            
            all_keys.sort(key=lambda x: (x[0], x[1]))
            selected_indices = [k[2] for k in all_keys[:pop_size]]
            
            X = X_comb[selected_indices]
            F = F_comb[selected_indices]
            
        return F
