import numpy as np
import scipy.stats.qmc
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        limit = 300.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Population size
        # With 200 budget, N=20 gives 10 generations.
        # N=30 gives 6 generations.
        # Let's try N=20 for more generational turnover.
        N = 20
        
        # 1. Initialization using Sobol sequence
        sobol = scipy.stats.qmc.Sobol(d=dim, scramble=True, seed=0)
        # We need at least N points. Sobol generates powers of 2.
        # N=20 is not a power of 2, so we generate 32 and take first 20? 
        # Or just generate random if Sobol is tricky with non-power-of-2.
        # Actually, Sobol can generate any number, but quality is best for powers of 2.
        # Let's generate 32 and pick 20? Or just use random if it's simpler?
        # Sobol is better. Let's generate 2^5=32 points and use the first N.
        m = 5
        while (2 ** m) < N:
            m += 1
        sample = sobol.random(2 ** m)
        X = lo + (hi - lo) * sample[:N]
        
        F = np.array([problem(x) for x in X])
        used = N

        # Helper functions for NSGA-II
        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]  # Dominates
            n_dom = np.zeros(n, dtype=int)  # Number of dominators
            rank = np.zeros(n, dtype=int)
            
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
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    rank[i] = k
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        def crowding_distance(F, rank, n):
            dist = np.zeros(n)
            for r in range(max(rank) + 1):
                front = [i for i in range(n) if rank[i] == r]
                if len(front) <= 2:
                    for i in front:
                        dist[i] = float('inf')
                    continue
                for m in range(n_obj):
                    front_sorted = sorted(front, key=lambda i: F[i][m])
                    dist[front_sorted[0]] = float('inf')
                    dist[front_sorted[-1]] = float('inf')
                    f_min = F[front_sorted[0]][m]
                    f_max = F[front_sorted[-1]][m]
                    if f_max == f_min:
                        continue
                    for k in range(1, len(front_sorted) - 1):
                        dist[front_sorted[k]] += (F[front_sorted[k+1]][m] - F[front_sorted[k-1]][m]) / (f_max - f_min)
            return dist

        def tournament_select(rank, dist, n, size=2):
            idxs = np.random.randint(0, n, size=size)
            best = idxs[0]
            for i in idxs[1:]:
                if rank[i] < rank[best]:
                    best = i
                elif rank[i] == rank[best] and dist[i] > dist[best]:
                    best = i
            return best

        def sbx_crossover(x1, x2, lo, hi, eta=15.0):
            if np.random.rand() > 0.9:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            
            for i in range(len(x1)):
                if np.random.rand() < 0.5:
                    x1_i, x2_i = x1[i], x2[i]
                    if x1_i == x2_i:
                        continue
                    if x1_i > x2_i:
                        x1_i, x2_i = x2_i, x1_i
                    
                    u = np.random.rand()
                    if u <= 0.5:
                        alpha = 1 + 2 * (x1_i - lo[i]) / (x2_i - x1_i)
                        beta = (2 * alpha) ** (1 / (eta + 1))
                        c1[i] = 0.5 * ((x1_i + x2_i) - beta * (x2_i - x1_i))
                    elif u >= 0.5:
                        alpha = 1 + 2 * (hi[i] - x2_i) / (x2_i - x1_i)
                        beta = (2 * alpha) ** (1 / (eta + 1))
                        c2[i] = 0.5 * ((x1_i + x2_i) + beta * (x2_i - x1_i))
                    else:
                        c1[i] = x2_i
                        c2[i] = x1_i
            return c1, c2

        def polynomial_mutation(x, lo, hi, eta=20.0):
            x_mut = x.copy()
            for i in range(len(x)):
                if np.random.rand() < 1.0 / self.dim:
                    u = np.random.rand()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                    
                    x_mut[i] += delta * (hi[i] - lo[i])
                    x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
            return x_mut

        # Main Loop
        while used < budget:
            if time.time() - start_time > limit:
                break
                
            # Create offspring
            offspring_X = []
            for _ in range(N):
                p1 = tournament_select(rank, dist, N)
                p2 = tournament_select(rank, dist, N)
                
                c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
            
            # We generated 2*N offspring, but we only have budget for N more?
            # No, the loop structure is:
            # 1. Generate N offspring (using N evaluations)
            # 2. Combine with parents (2N)
            # 3. Select N
            # This consumes N evaluations per iteration.
            
            # Wait, the above loop generates 2*N offspring per iteration?
            # No, standard NSGA-II generates N offspring per generation.
            # So I should generate N offspring, not 2*N.
            
            # Let's fix the offspring generation to produce exactly N children.
            offspring_X = []
            for _ in range(N // 2):
                p1 = tournament_select(rank, dist, N)
                p2 = tournament_select(rank, dist, N)
                
                c1, c2 = sbx_crossover(X[p1], X[p2], lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                offspring_X.append(c1)
                offspring_X.append(c2)
                
            # If N is odd, we might need to handle the last one, but N=20 is even.
            
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += N
            
            # Combine
            X_all = np.vstack((X, offspring_X))
            F_all = np.vstack((F, offspring_F))
            
            # Non-dominated sort and crowding
            fronts, rank_all = non_dominated_sort(F_all)
            dist_all = crowding_distance(F_all, rank_all, len(X_all))
            
            # Select next N
            X_new = []
            F_new = []
            for front in fronts:
                if len(X_new) + len(front) <= N:
                    for idx in front:
                        X_new.append(X_all[idx])
                        F_new.append(F_all[idx])
                else:
                    # Sort front by crowding distance
                    front_sorted = sorted(front, key=lambda i: dist_all[i], reverse=True)
                    for idx in front_sorted:
                        if len(X_new) < N:
                            X_new.append(X_all[idx])
                            F_new.append(F_all[idx])
                        else:
                            break
                if len(X_new) >= N:
                    break
            
            X = np.array(X_new)
            F = np.array(F_new)
            # Update rank and dist for next iteration
            fronts, rank = non_dominated_sort(F)
            dist = crowding_distance(F, rank, N)

        return X
