import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # 300s limit, keep 10s margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        pop_size = 20
        # Initialize population
        X = lo + range_ * np.random.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        evals = pop_size
        
        # Helper: Non-dominated sorting
        def fast_non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_count = np.zeros(n, dtype=int)
            fronts = [[]]
            
            for p in range(n):
                for q in range(n):
                    if p == q: continue
                    # p dominates q
                    if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                        S[p].append(q)
                    # q dominates p
                    elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                        n_count[p] += 1
                        
                if n_count[p] == 0:
                    fronts[0].append(p)
            
            i = 0
            while len(fronts[i]) > 0:
                Q = []
                for p in fronts[i]:
                    for q in S[p]:
                        n_count[q] -= 1
                        if n_count[q] == 0:
                            Q.append(q)
                i += 1
                if len(Q) > 0:
                    fronts.append(Q)
                else:
                    break
            return fronts

        # Helper: Hypervolume contribution (approximate for selection)
        # For 3 objectives, we can compute the contribution of each point to the total HV.
        # Since we only use this for selection within a front, we can use a simpler metric:
        # Distance to other points in the front (crowding) or actual HV contribution.
        # Let's use a simple HV contribution approximation:
        # Volume of the box defined by the point and the reference point, 
        # minus volumes covered by other points? Too complex.
        # Let's stick to Crowding Distance for now, it's standard and fast.
        
        def crowding_distance(F, indices):
            if len(indices) <= 2:
                return np.ones(len(indices)) * np.inf
            dist = np.zeros(len(indices))
            F_sub = F[indices]
            for m in range(self.n_obj):
                min_val = np.min(F_sub[:, m])
                max_val = np.max(F_sub[:, m])
                if max_val == min_val:
                    continue
                idx = np.argsort(F_sub[:, m])
                dist[idx[0]] = np.inf
                dist[idx[-1]] = np.inf
                for i in range(1, len(indices) - 1):
                    dist[idx[i]] += (F_sub[idx[i+1], m] - F_sub[idx[i-1], m]) / (max_val - min_val)
            return dist

        # Main loop
        while evals < self.budget:
            if time.time() > deadline:
                break
                
            # Generate offspring
            offspring_X = np.zeros_like(X)
            offspring_F = np.zeros_like(F)
            
            for i in range(pop_size):
                if time.time() > deadline:
                    break
                
                # Selection: Tournament of 2 based on rank and crowding
                p1_idx, p2_idx = self._tournament_select(F, pop_size)
                p1 = X[p1_idx]
                p2 = X[p2_idx]
                
                # SBX Crossover
                child = self._sbx_crossover(p1, p2, lo, hi)
                
                # Polynomial Mutation
                child = self._polynomial_mutate(child, lo, hi)
                
                offspring_X[i] = child
                offspring_F[i] = problem(child)
                evals += 1
                
            # Combine
            total_X = np.vstack([X, offspring_X])
            total_F = np.vstack([F, offspring_F])
            
            # Selection
            fronts = fast_non_dominated_sort(total_F)
            new_indices = []
            for front in fronts:
                if len(new_indices) + len(front) <= pop_size:
                    new_indices.extend(front)
                else:
                    # Need to select from this front
                    remaining = pop_size - len(new_indices)
                    cd = crowding_distance(total_F, front)
                    # Sort by crowding distance descending
                    sorted_front = np.array(front)[np.argsort(cd)[::-1]]
                    new_indices.extend(sorted_front[:remaining].tolist())
                    break
                    
            X = total_X[new_indices]
            F = total_F[new_indices]
            
        return X

    def _tournament_select(self, F, pop_size):
        # Simple rank-based tournament
        # Calculate ranks
        n = len(F)
        S = [[] for _ in range(n)]
        n_count = np.zeros(n, dtype=int)
        ranks = np.zeros(n, dtype=int)
        
        for p in range(n):
            for q in range(n):
                if p == q: continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    n_count[p] += 1
        # Assign ranks
        current_front = []
        for p in range(n):
            if n_count[p] == 0:
                current_front.append(p)
                ranks[p] = 1
        i = 1
        while current_front:
            next_front = []
            for p in current_front:
                for q in S[p]:
                    n_count[q] -= 1
                    if n_count[q] == 0:
                        next_front.append(q)
                        ranks[q] = i + 1
            current_front = next_front
            i += 1
            
        # Tournament
        idx1 = np.random.randint(pop_size)
        idx2 = np.random.randint(pop_size)
        if ranks[idx1] < ranks[idx2]:
            return idx1, idx2
        elif ranks[idx2] < ranks[idx1]:
            return idx2, idx1
        else:
            # Same rank, pick random
            return idx1, idx2

    def _sbx_crossover(self, p1, p2, lo, hi):
        child = p1.copy()
        if np.random.random() < 0.5:
            for i in range(self.dim):
                if np.random.random() < 0.5:
                    child[i] = p2[i]
        # Simple BLX-alpha or just SBX? Let's use simple arithmetic crossover with probability
        # Actually, SBX is better.
        # Simplified SBX
        eta_c = 20
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if p1[i] == p2[i]:
                    continue
                if p1[i] < p2[i]:
                    y1, y2 = p1[i], p2[i]
                else:
                    y1, y2 = p2[i], p1[i]
                u = np.random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta_c + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta_c + 1))
                c1 = 0.5 * ((1 + beta) * y1 + (1 - beta) * y2)
                c2 = 0.5 * ((1 - beta) * y1 + (1 + beta) * y2)
                # Clip
                c1 = np.clip(c1, lo[i], hi[i])
                c2 = np.clip(c2, lo[i], hi[i])
                # We only return one child here, so we pick one
                if np.random.random() < 0.5:
                    child[i] = c1
                else:
                    child[i] = c2
        return child

    def _polynomial_mutate(self, x, lo, hi):
        eta_m = 20
        for i in range(self.dim):
            if np.random.random() < 0.5:
                u = np.random.random()
                if u < 0.5:
                    delta1 = (2 * u) ** (1 / (eta_m + 1)) - 1
                else:
                    delta1 = 1 - (2 * (1 - u)) ** (1 / (eta_m + 1))
                x[i] += delta1 * (hi[i] - lo[i])
                x[i] = np.clip(x[i], lo[i], hi[i])
        return x
