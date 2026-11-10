import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        N = 20
        if N > self.budget:
            N = self.budget
            
        # Initial population
        X = lo + (hi - lo) * np.random.random((N, dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        # NSGA-II Parameters
        pc = 0.9  # Crossover probability
        pm = 1.0 / dim  # Mutation probability
        eta_c = 15.0  # Crossover distribution index
        eta_m = 20.0  # Mutation distribution index
        
        def non_dominated_sort(F):
            """Fast non-dominated sorting. Returns list of fronts, each a list of indices."""
            n = len(F)
            S = [[] for _ in range(n)]  # Dominated sets
            n_d = [0] * n  # Number of dominators
            fronts = [[]]
            
            for i in range(n):
                for j in range(n):
                    if i == j: continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # j dominates i?
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_d[i] += 1
                if n_d[i] == 0:
                    fronts[0].append(i)
                    
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_d[j] -= 1
                        if n_d[j] == 0:
                            next_front.append(j)
                k += 1
                fronts.append(next_front)
            return fronts[:-1] # Remove last empty front if any

        def crowding_distance(F, front_indices):
            """Calculate crowding distance for individuals in a front."""
            n = len(front_indices)
            if n <= 2:
                return [np.inf] * n
            
            cd = [0.0] * n
            for m in range(n_obj):
                f_m = F[front_indices][:, m]
                sorted_idx = np.argsort(f_m)
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                f_range = f_m[sorted_idx[-1]] - f_m[sorted_idx[0]]
                if f_range == 0: continue
                for i in range(1, n - 1):
                    cd[sorted_idx[i]] += (f_m[sorted_idx[i+1]] - f_m[sorted_idx[i-1]]) / f_range
            return cd

        def sbx_crossover(x1, x2, lo, hi):
            if np.random.random() > pc:
                return x1.copy(), x2.copy()
            
            c1 = x1.copy()
            c2 = x2.copy()
            for i in range(dim):
                if np.random.random() < 0.5:
                    u = np.random.random()
                    if u <= 0.5:
                        alpha = (2 * u) ** (1 / (eta_c + 1))
                    else:
                        alpha = (2 * (1 - u)) ** (1 / (eta_c + 1))
                    c1[i] = 0.5 * ((1 + alpha) * x1[i] + (1 - alpha) * x2[i])
                    c2[i] = 0.5 * ((1 - alpha) * x1[i] + (1 + alpha) * x2[i])
            return np.clip(c1, lo, hi), np.clip(c2, lo, hi)

        def polynomial_mutation(x, lo, hi):
            for i in range(dim):
                if np.random.random() < pm:
                    u = np.random.random()
                    if u < 0.5:
                        delta = (2 * u) ** (1 / (eta_m + 1)) - 1
                    else:
                        delta = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                    x[i] += delta * (hi[i] - lo[i])
            return np.clip(x, lo, hi)

        def tournament_select(pop_X, pop_F, k=2):
            indices = np.random.randint(0, len(pop_X), size=k)
            best = indices[0]
            for idx in indices[1:]:
                # Simple dominance check for tournament
                if np.all(pop_F[idx] <= pop_F[best]) and np.any(pop_F[idx] < pop_F[best]):
                    best = idx
            return pop_X[best].copy()

        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Create offspring
            offspring_X = []
            offspring_F = []
            
            while len(offspring_X) < N and evals < self.budget:
                p1 = tournament_select(X, F)
                p2 = tournament_select(X, F)
                
                c1, c2 = sbx_crossover(p1, p2, lo, hi)
                c1 = polynomial_mutation(c1, lo, hi)
                c2 = polynomial_mutation(c2, lo, hi)
                
                f1 = problem(c1)
                f2 = problem(c2)
                evals += 2
                
                offspring_X.append(c1)
                offspring_F.append(f1)
                
                if evals < self.budget and len(offspring_X) < N:
                    offspring_X.append(c2)
                    offspring_F.append(f2)
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parents and offspring
            R_X = np.vstack([X, offspring_X])
            R_F = np.vstack([F, offspring_F])
            
            # Non-dominated sort
            fronts = non_dominated_sort(R_F)
            
            # Select next generation
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                else:
                    # Need to select from this front using crowding distance
                    needed = N - len(new_X)
                    cd = crowding_distance(R_F, front)
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(range(len(front)), key=lambda i: cd[i], reverse=True)
                    for i in range(needed):
                        idx = front[sorted_front[i]]
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
