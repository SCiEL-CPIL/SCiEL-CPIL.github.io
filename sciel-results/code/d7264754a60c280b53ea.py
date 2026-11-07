import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        """Check if a dominates b (minimization)."""
        return np.all(a <= b) and np.any(a < b)

    def _fast_nondominated_sort(self, F):
        """Fast non-dominated sorting. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_count = np.zeros(n, dtype=int)  # n[i] = number of points dominating i
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    n_count[i] += 1
        
        front = [i for i in range(n) if n_count[i] == 0]
        if not front:
            return [list(range(n))]
            
        fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_count[j] -= 1
                    if n_count[j] == 0:
                        next_front.append(j)
            if not next_front:
                break
            fronts.append(next_front)
            front = next_front
            
        return fronts

    def _crowding_distance(self, F, front_indices):
        """Calculate crowding distance for points in a front."""
        n = len(front_indices)
        if n <= 2:
            return np.full(n, float('inf'))
            
        cd = np.zeros(n)
        m = self.n_obj
        
        for k in range(m):
            # Sort by objective k
            sorted_idx = sorted(range(n), key=lambda i: F[front_indices[i]][k])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            
            min_val = F[front_indices[sorted_idx[0]]][k]
            max_val = F[front_indices[sorted_idx[-1]]][k]
            range_val = max_val - min_val
            
            if range_val < 1e-14:
                continue
                
            for i in range(1, n - 1):
                idx = sorted_idx[i]
                prev_idx = sorted_idx[i - 1]
                next_idx = sorted_idx[i + 1]
                cd[idx] += (F[front_indices[next_idx]][k] - F[front_indices[prev_idx]][k]) / range_val
                
        return cd

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        """Simulated Binary Crossover."""
        child = p1.copy()
        if np.random.random() > 0.5:
            p1, p2 = p2, p1
            
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) > 1e-14:
                    x1, x2 = p1[i], p2[i]
                    if x1 > x2:
                        x1, x2 = x2, x1
                    
                    rand = np.random.random()
                    if rand <= 1.0 / (eta + 1.0):
                        beta = (2.0 * rand) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - rand))) ** (1.0 / (eta + 1.0))
                    
                    c1 = 0.5 * (x1 + x2 - beta * (x2 - x1))
                    c2 = 0.5 * (x1 + x2 + beta * (x2 - x1))
                    
                    child[i] = c1 if np.random.random() < 0.5 else c2
                    
        return np.clip(child, lo, hi)

    def _poly_mutation(self, x, lo, hi, eta, pm):
        """Polynomial Mutation."""
        x = x.copy()
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
                x[i] = np.clip(xq, lo[i], hi[i])
        return x

    def _tournament_select(self, F, ranks, cd, k=2):
        """Tournament selection based on rank and crowding distance."""
        n = len(F)
        candidates = np.random.randint(0, n, k)
        best = candidates[0]
        for c in candidates[1:]:
            if ranks[c] < ranks[best]:
                best = c
            elif ranks[c] == ranks[best] and cd[c] > cd[best]:
                best = c
        return best

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 330s kill
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size: balance between exploration and exploitation
        # For budget=200, N=20 gives ~10 generations
        N = min(20, max(10, budget // 10))
        
        if budget < 2 * N:
            # Not enough budget for NSGA-II, just do random sampling
            X = lo + (hi - lo) * np.random.random((budget, dim))
            F = np.array([problem(x) for x in X])
            return X
        
        # Initialize population with Latin Hypercube Sampling for better coverage
        X = np.zeros((N, dim))
        for d in range(dim):
            # Create intervals: [i/N, (i+1)/N) for i in range(N)
            intervals = np.arange(N) / N
            # Sample one point from each interval
            samples = intervals + np.random.random(N) / N
            # Permute to decorrelate dimensions
            perm = np.random.permutation(N)
            X[:, d] = samples[perm]
        
        F = np.array([problem(x) for x in X])
        used = N
        
        # NSGA-II parameters
        pc = 0.9
        eta_c = 15.0
        eta_m = 20.0
        pm = 1.0 / dim
        
        # Adaptive mutation: start with larger steps, reduce as we converge
        base_pm = pm
        base_eta_m = eta_m
        
        while used + N <= budget:
            if time.time() > deadline:
                break
            
            # Calculate ranks and crowding distance
            fronts = self._fast_nondominated_sort(F)
            ranks = np.zeros(N, dtype=int)
            cd = np.zeros(N)
            
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
                if len(front) > 2:
                    cd[front] = self._crowding_distance(F, front)
                else:
                    cd[front] = float('inf')
            
            # Generate offspring
            offspring_X = np.empty_like(X)
            offspring_F = np.empty_like(F)
            n_off = 0
            
            # Adaptive parameters based on progress
            progress = used / budget
            current_pm = base_pm * (1.0 + 0.5 * (1.0 - progress))
            current_eta_m = base_eta_m * (1.0 + 0.5 * progress)
            
            for i in range(N):
                if time.time() > deadline:
                    break
                if used + n_off >= budget:
                    break
                    
                # Select parents
                p1_idx = self._tournament_select(F, ranks, cd)
                p2_idx = self._tournament_select(F, ranks, cd)
                
                # Crossover
                if np.random.random() < pc:
                    child = self._sbx_crossover(X[p1_idx], X[p2_idx], lo, hi, eta_c)
                else:
                    child = X[p1_idx].copy()
                
                # Mutation
                child = self._poly_mutation(child, lo, hi, current_eta_m, current_pm)
                
                # Evaluate
                f = problem(child)
                offspring_X[i] = child
                offspring_F[i] = np.asarray(f, dtype=float)
                n_off += 1
                used += 1
            
            if n_off == 0:
                break
                
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X[:n_off]])
            combined_F = np.vstack([F, offspring_F[:n_off]])
            
            # Non-dominated sorting on combined set
            combined_fronts = self._fast_nondominated_sort(combined_F)
            combined_ranks = np.zeros(len(combined_F), dtype=int)
            combined_cd = np.zeros(len(combined_F))
            
            for rank, front in enumerate(combined_fronts):
                for idx in front:
                    combined_ranks[idx] = rank
                if len(front) > 2:
                    combined_cd[front] = self._crowding_distance(combined_F, front)
                else:
                    combined_cd[front] = float('inf')
            
            # Select next generation
            next_indices = []
            for front in combined_fronts:
                if len(next_indices) + len(front) <= N:
                    next_indices.extend(front)
                else:
                    # Need to truncate this front
                    remaining = N - len(next_indices)
                    front_cd = combined_cd[front]
                    # Sort by crowding distance (descending)
                    sorted_front = sorted(front, key=lambda i: front_cd[i], reverse=True)
                    next_indices.extend(sorted_front[:remaining])
                    break
            
            # Update population
            X = combined_X[next_indices]
            F = combined_F[next_indices]
        
        return X
