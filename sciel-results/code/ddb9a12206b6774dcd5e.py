import numpy as np
from scipy.spatial.distance import cdist

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.rng = np.random.default_rng(42)

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # NSGA-II Parameters
        # Budget is 200. 
        # Population size N. 
        # Generations G.
        # Total evals = N (init) + G * 2*N (population + offspring).
        # 200 = N + G * 2*N => N(1 + 2G) <= 200.
        # Let's pick N=20, G=4 => 20 * (1 + 8) = 180. 
        # We have 20 extra evaluations. We can use them for a final local search or just stop.
        # Or pick N=16, G=5 => 16 * (1 + 10) = 176.
        # Let's stick with N=20, G=4.
        N = 20
        G = (self.budget // N - 1) // 2
        if G < 1:
            G = 1
            N = self.budget // 3
            
        # Initialize Population
        X = lo + (hi - lo) * self.rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        self.budget_used = N
        
        def is_dominated(a, b):
            return np.all(a <= b) and np.any(a < b)
        
        def fast_non_dominated_sort(F):
            M = len(F)
            S = [[] for _ in range(M)]
            n = [0] * M
            fronts = [[]]
            
            for i in range(M):
                for j in range(M):
                    if i == j:
                        continue
                    if is_dominated(F[i], F[j]):
                        S[i].append(j)
                    elif is_dominated(F[j], F[i]):
                        n[i] += 1
                        
                if n[i] == 0:
                    fronts[0].append(i)
                    
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n[j] -= 1
                        if n[j] == 0:
                            next_front.append(j)
                k += 1
                if next_front:
                    fronts.append(next_front)
                else:
                    break
                    
            return fronts[:k+1]
            
        def crowding_distance(F, front_indices):
            if len(front_indices) <= 1:
                return np.full(len(front_indices), np.inf)
                
            M = len(front_indices)
            cd = np.zeros(M)
            f = F[front_indices]
            
            for k in range(self.n_obj):
                sorted_idx = np.argsort(f[:, k])
                cd[sorted_idx[0]] = np.inf
                cd[sorted_idx[-1]] = np.inf
                
                f_min = f[sorted_idx[0], k]
                f_max = f[sorted_idx[-1], k]
                if f_max == f_min:
                    continue
                
                for i in range(1, M - 1):
                    cd[sorted_idx[i]] += (f[sorted_idx[i+1], k] - f[sorted_idx[i-1], k]) / (f_max - f_min)
                    
            return cd

        # Main Loop
        for g in range(G):
            if self.budget_used >= self.budget:
                break
                
            # Create Offspring
            offspring_X = np.zeros((N, self.dim))
            offspring_F = np.zeros((N, self.n_obj))
            
            # Tournament Selection
            def tournament_select(X, F, fronts, n_t=2):
                indices = np.random.choice(len(X), size=n_t, replace=False)
                # Find the best index based on fronts
                best_idx = -1
                best_front = 1000000
                for idx in indices:
                    # Find which front idx belongs to
                    for fi, front in enumerate(fronts):
                        if idx in front:
                            if fi < best_front:
                                best_front = fi
                                best_idx = idx
                            break
                return best_idx
                
            for i in range(N):
                # Select Parents
                p1_idx = tournament_select(X, F, fronts)
                p2_idx = tournament_select(X, F, fronts)
                
                # Simulated Binary Crossover (SBX)
                if self.rng.random() < 0.9:
                    # Simple crossover: uniform blend
                    alpha = self.rng.random(self.dim)
                    child1 = alpha * X[p1_idx] + (1 - alpha) * X[p2_idx]
                    child2 = (1 - alpha) * X[p1_idx] + alpha * X[p2_idx]
                else:
                    child1 = X[p1_idx].copy()
                    child2 = X[p2_idx].copy()
                    
                # Polynomial Mutation
                # Simple Gaussian mutation
                sigma = 0.1 * (hi - lo)
                child1 += self.rng.normal(0, sigma, self.dim)
                child2 += self.rng.normal(0, sigma, self.dim)
                
                # Clip to bounds
                child1 = np.clip(child1, lo, hi)
                child2 = np.clip(child2, lo, hi)
                
                offspring_X[2*i] = child1
                offspring_F[2*i] = problem(child1)
                self.budget_used += 1
                
                if self.budget_used >= self.budget:
                    break
                    
                offspring_X[2*i+1] = child2
                offspring_F[2*i+1] = problem(child2)
                self.budget_used += 1
                
                if self.budget_used >= self.budget:
                    break
                    
            if self.budget_used >= self.budget:
                break
                
            # Combine Parent and Offspring
            R_X = np.vstack([X, offspring_X])
            R_F = np.vstack([F, offspring_F])
            
            # Non-dominated Sort
            fronts = fast_non_dominated_sort(R_F)
            
            # Selection
            new_X = []
            new_F = []
            
            for front in fronts:
                if len(new_X) + len(front) <= N:
                    # Add all
                    for idx in front:
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                else:
                    # Need to select some from this front based on crowding distance
                    cd = crowding_distance(R_F, front)
                    sorted_front = np.argsort(cd)[::-1]
                    needed = N - len(new_X)
                    for j in range(needed):
                        idx = front[sorted_front[j]]
                        new_X.append(R_X[idx])
                        new_F.append(R_F[idx])
                    break
                    
            X = np.array(new_X)
            F = np.array(new_F)
            
        return X
