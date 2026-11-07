import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 250  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Parameters
        # Smaller population allows more generations, improving front refinement
        N = 10 
        if N > budget:
            N = budget
            
        # Initialize with Latin Hypercube Sampling for better spread
        # Simple stratified sampling
        X = np.zeros((N, dim))
        for d in range(dim):
            # Create N intervals
            intervals = np.linspace(lo[d], hi[d], N + 1)
            # Pick random point in each interval
            for i in range(N):
                X[i, d] = np.random.uniform(intervals[i], intervals[i+1])
        
        F = np.array([problem(x) for x in X])
        used = N
        
        # Helper: Non-dominated sort
        def non_dominated_sort(F):
            fronts = []
            remaining = list(range(len(F)))
            while remaining:
                non_dom = []
                for i in remaining:
                    dominated = False
                    for j in remaining:
                        if i == j: continue
                        if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            dominated = True
                            break
                    if not dominated:
                        non_dom.append(i)
                fronts.append(non_dom)
                remaining = [i for i in remaining if i not in non_dom]
            return fronts

        # Helper: Crowding distance
        def crowding_distance(F, indices):
            if len(indices) == 0:
                return {}
            dist = {i: 0.0 for i in indices}
            for m in range(n_obj):
                f_vals = [F[i][m] for i in indices]
                min_f = min(f_vals)
                max_f = max(f_vals)
                if max_f == min_f:
                    for i in indices:
                        dist[i] = float('inf')
                    continue
                sorted_indices = sorted(indices, key=lambda i: F[i][m])
                dist[sorted_indices[0]] = float('inf')
                dist[sorted_indices[-1]] = float('inf')
                for k in range(1, len(sorted_indices) - 1):
                    prev = sorted_indices[k-1]
                    next_ = sorted_indices[k+1]
                    dist[sorted_indices[k]] += (F[next_][m] - F[prev][m]) / (max_f - min_f)
            return dist

        # Main Loop
        while used < budget and (time.time() < deadline):
            # Compute ranks and crowding
            fronts = non_dominated_sort(F)
            ranks = np.zeros(N, dtype=int)
            for rank, front in enumerate(fronts):
                for i in front:
                    ranks[i] = rank
            cd = crowding_distance(F, list(range(N)))
            
            offspring_X = []
            offspring_F = []
            
            # Generate N offspring
            for _ in range(N):
                # Tournament selection
                p1_idx = np.random.randint(N)
                p2_idx = np.random.randint(N)
                if ranks[p1_idx] < ranks[p2_idx]:
                    p1 = p1_idx
                elif ranks[p2_idx] < ranks[p1_idx]:
                    p1 = p2_idx
                else:
                    p1 = p1_idx if cd[p1_idx] > cd[p2_idx] else p2_idx
                
                p3_idx = np.random.randint(N)
                p4_idx = np.random.randint(N)
                if ranks[p3_idx] < ranks[p4_idx]:
                    p2 = p3_idx
                elif ranks[p4_idx] < ranks[p3_idx]:
                    p2 = p4_idx
                else:
                    p2 = p3_idx if cd[p3_idx] > cd[p4_idx] else p4_idx
                
                # Crossover (Blend)
                x1 = X[p1]
                x2 = X[p2]
                alpha = np.random.random()
                child = (1 - alpha) * x1 + alpha * x2
                
                # Mutation
                # Adaptive sigma: start wide, end narrow
                progress = used / budget
                # Use a power law decay for sigma to allow exploration early, exploitation later
                sigma = 0.3 * (1 - progress) ** 1.5 
                # If progress is high, sigma is small
                child = child + sigma * span * np.random.randn(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fc)
                
                if used >= budget or time.time() > deadline:
                    break

            if used >= budget or time.time() > deadline:
                break

            # Combine and Select
            X_new = np.vstack([X, np.array(offspring_X)])
            F_new = np.array(list(F) + offspring_F)
            
            fronts_new = non_dominated_sort(F_new)
            next_X = []
            next_F = []
            
            for front in fronts_new:
                if len(next_X) + len(front) <= N:
                    for i in front:
                        next_X.append(X_new[i])
                        next_F.append(F_new[i])
                else:
                    need = N - len(next_X)
                    cd_front = crowding_distance(F_new, front)
                    sorted_front = sorted(front, key=lambda i: cd_front[i], reverse=True)
                    for i in sorted_front[:need]:
                        next_X.append(X_new[i])
                        next_F.append(F_new[i])
                    break
                        
            X = np.array(next_X)
            F = np.array(next_F)

        return X
