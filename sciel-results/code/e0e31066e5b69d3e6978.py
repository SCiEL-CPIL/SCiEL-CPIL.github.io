import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Population size
        N = 20
        
        # Initialize
        rng = np.random.default_rng(42)
        X = lo + (hi - lo) * rng.random((N, dim))
        F = np.array([problem(x) for x in X])
        
        evals_used = N
        
        start_time = time.time()
        deadline = start_time + 290 # 40s margin
        
        def fast_non_dominated_sort(F):
            """Returns list of lists of indices, sorted by non-dominated front."""
            n = len(F)
            S = [[] for _ in range(n)]  # S[i] contains indices dominated by i
            n_dom = np.zeros(n, dtype=int)
            rank = np.zeros(n, dtype=int)
            
            fronts = []
            current_front = []
            
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    # i dominates j?
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                    # j dominates i?
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        n_dom[i] += 1
                        
                if n_dom[i] == 0:
                    rank[i] = 0
                    current_front.append(i)
                    
            if current_front:
                fronts.append(current_front)
            
            k = 0
            while fronts[k]:
                next_front = []
                for i in fronts[k]:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                if next_front:
                    k += 1
                    fronts.append(next_front)
            
            return fronts

        def crowding_distance(F, indices):
            """Calculate crowding distance for a list of indices."""
            n = len(indices)
            if n <= 2:
                return {i: float('inf') for i in indices}
            
            dist = np.zeros(n)
            m = self.n_obj
            
            for k in range(m):
                # Sort indices by objective k
                # We need to map the sorted values back to the dist array
                vals = np.array([F[i][k] for i in indices])
                sort_order = np.argsort(vals)
                sorted_vals = vals[sort_order]
                
                dist_sorted = np.zeros(n)
                dist_sorted[0] = float('inf')
                dist_sorted[-1] = float('inf')
                
                if sorted_vals[-1] - sorted_vals[0] > 1e-12:
                    for i in range(1, n - 1):
                        dist_sorted[i] = (sorted_vals[i+1] - sorted_vals[i-1]) / (sorted_vals[-1] - sorted_vals[0])
                
                for j in range(n):
                    dist[sort_order[j]] = dist_sorted[j]
                    
            return {indices[i]: dist[i] for i in range(n)}

        while evals_used < budget:
            # Check time
            if time.time() > deadline:
                break
                
            # Generate offspring
            Q = np.zeros((N, dim))
            for i in range(N):
                # DE Mutation
                r1, r2, r3 = rng.choice(N, size=3, replace=False)
                v = X[r1] + 0.5 * (X[r2] - X[r3])
                # Clip to bounds
                v = np.clip(v, lo, hi)
                
                # Crossover (Uniform)
                r4 = rng.choice(N)
                mask = rng.random(dim) < 0.5
                child = np.where(mask, v, X[r4])
                Q[i] = child
            
            # Evaluate offspring
            F_Q = np.array([problem(x) for x in Q])
            evals_used += N
            
            # Combine
            R = np.vstack([X, Q])
            F_R = np.vstack([F, F_Q])
            
            # Non-dominated sort
            fronts = fast_non_dominated_sort(F_R)
            
            # Select next population
            P_next = []
            for front in fronts:
                if len(P_next) + len(front) <= N:
                    P_next.extend(front)
                else:
                    # Need to select from this front
                    need = N - len(P_next)
                    # Calculate crowding distance for this front
                    cd_map = crowding_distance(F_R, front)
                    # Sort by crowding distance descending
                    sorted_front = sorted(front, key=lambda i: cd_map[i], reverse=True)
                    P_next.extend(sorted_front[:need])
                    break
            
            X = R[P_next]
            F = F_R[P_next]

        return X
