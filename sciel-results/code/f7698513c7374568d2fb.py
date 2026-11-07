import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 300s limit, aim for 250s
        deadline = start_time + 250
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Population size
        # For 200 budget, N=20 allows 10 generations (1 init + 9 offspring generations? No, 20*10=200)
        # Let's use N=20.
        N = 20
        
        # Check if budget is sufficient for NSGA-II
        if self.budget < 2 * N:
            # Just do random sampling
            X = lo + (hi - lo) * np.random.random((self.budget, self.dim))
            F = np.array([problem(x) for x in X])
            return X

        # Initialize
        X = lo + (hi - lo) * np.random.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        used = N

        # Parameters
        pc = 0.9  # Crossover probability
        eta_c = 15.0 # Crossover distribution index
        pm = 1.0 / dim # Mutation probability
        eta_m = 20.0 # Mutation distribution index

        while used + N <= self.budget:
            if time.time() > deadline:
                break
                
            offspring_X = np.empty_like(X)
            offspring_F = np.empty_like(F)
            
            for i in range(N):
                if time.time() > deadline:
                    break
                
                # Tournament Selection
                idx1 = np.random.randint(0, N)
                idx2 = np.random.randint(0, N)
                # Simple dominance check for tournament
                if self._dominates(F[idx1], F[idx2]):
                    p1 = X[idx1]
                elif self._dominates(F[idx2], F[idx1]):
                    p1 = X[idx2]
                else:
                    p1 = X[np.random.choice([idx1, idx2])]
                    
                idx3 = np.random.randint(0, N)
                idx4 = np.random.randint(0, N)
                if self._dominates(F[idx3], F[idx4]):
                    p2 = X[idx3]
                elif self._dominates(F[idx4], F[idx3]):
                    p2 = X[idx4]
                else:
                    p2 = X[np.random.choice([idx3, idx4])]

                # Crossover (SBX)
                if np.random.random() < pc:
                    child = self._sbx(p1, p2, lo, hi, eta_c)
                else:
                    child = p1.copy()
                    
                # Mutation (Polynomial)
                child = self._poly_mutate(child, lo, hi, eta_m, pm)
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                used += 1
                
                if used > self.budget:
                    break

            if used > self.budget:
                break

            # Combine
            n_off = used - N
            X_comb = np.vstack([X, offspring_X[:n_off]])
            F_comb = np.vstack([F, offspring_F[:n_off]])
            
            # Select next generation
            X, F = self._nsga2_select(X_comb, F_comb, N)

        return X

    def _dominates(self, a, b):
        # a dominates b if a <= b in all objectives and a < b in at least one
        return np.all(a <= b) and np.any(a < b)

    def _sbx(self, p1, p2, lo, hi, eta):
        child = p1.copy()
        if np.random.random() > 0.5:
            p1, p2 = p2, p1
            
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    x1, x2 = p1[i], p2[i]
                    if x1 > x2:
                        x1, x2 = x2, x1
                    beta = 1.0
                    alpha = 2.0 - (x1 - lo[i]) / (x2 - x1 + 1e-14) if x2 > x1 else 1.0
                    # Correct SBX formula
                    rand = np.random.random()
                    if rand <= 1.0 / (eta + 1.0):
                        beta = (2.0 * rand) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * (x1 + x2 - beta * (x2 - x1))
                    c2 = 0.5 * (x1 + x2 + beta * (x2 - x1))
                    
                    # Clip to bounds
                    if c1 < lo[i]:
                        c1 = lo[i]
                    if c1 > hi[i]:
                        c1 = hi[i]
                    if c2 < lo[i]:
                        c2 = lo[i]
                    if c2 > hi[i]:
                        c2 = hi[i]
                    
                    child[i] = c1 if np.random.random() < 0.5 else c2
        return child

    def _poly_mutate(self, x, lo, hi, eta, pm):
        for i in range(self.dim):
            if np.random.random() < pm:
                delta1 = (x[i] - lo[i]) / (hi[i] - lo[i] + 1e-14)
                delta2 = (hi[i] - x[i]) / (hi[i] - lo[i] + 1e-14)
                rand = np.random.random()
                if rand < 0.5:
                    xy = 1.0 - (1.0 - delta1) ** (eta + 1.0)
                else:
                    xy = 1.0 - (1.0 - delta2) ** (eta + 1.0)
                
                mut_pow = 1.0 / (eta + 1.0)
                xq = x[i] + (hi[i] - lo[i]) * (xy - 0.5)
                
                if xq < lo[i]:
                    xq = lo[i]
                if xq > hi[i]:
                    xq = hi[i]
                x[i] = xq
        return x

    def _nsga2_select(self, X, F, N):
        # Non-dominated sorting
        n = len(X)
        ranks = np.zeros(n, dtype=int)
        fronts = []
        remaining = set(range(n))
        
        while remaining:
            front = []
            for i in remaining:
                is_non_dominated = True
                for j in remaining:
                    if i == j:
                        continue
                    if self._dominates(F[j], F[i]):
                        is_non_dominated = False
                        break
                if is_non_dominated:
                    front.append(i)
            
            for i in front:
                ranks[i] = len(fronts)
                remaining.remove(i)
            
            fronts.append(front)
            
            # Remove dominated points from remaining? 
            # Actually, standard NSGA-II just sorts. 
            # But for efficiency, we can remove points dominated by the current front from 'remaining' 
            # because they will be in later fronts.
            # However, with N=20, n=40, this loop is small.
            
        # Crowding distance
        cd = np.zeros(n)
        for f in fronts:
            if len(f) <= 2:
                for i in f:
                    cd[i] = float('inf')
                continue
            
            # Sort by each objective
            for k in range(self.n_obj):
                sorted_indices = sorted(f, key=lambda i: F[i][k])
                cd[sorted_indices[0]] = float('inf')
                cd[sorted_indices[-1]] = float('inf')
                
                min_val = F[sorted_indices[0]][k]
                max_val = F[sorted_indices[-1]][k]
                range_val = max_val - min_val
                if range_val == 0:
                    continue
                    
                for j in range(1, len(sorted_indices) - 1):
                    idx = sorted_indices[j]
                    prev_idx = sorted_indices[j-1]
                    next_idx = sorted_indices[j+1]
                    cd[idx] += (F[next_idx][k] - F[prev_idx][k]) / range_val
        
        # Select N individuals
        selected = []
        for f in fronts:
            if len(selected) + len(f) <= N:
                selected.extend(f)
            else:
                # Sort front by crowding distance
                f_sorted = sorted(f, key=lambda i: cd[i], reverse=True)
                selected.extend(f_sorted[:N - len(selected)])
                break
        
        return X[selected], F[selected]
