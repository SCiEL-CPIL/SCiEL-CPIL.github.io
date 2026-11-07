import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Choose population size
        # 200 budget. N=20 is reasonable.
        N = 20
        if N > self.budget:
            N = self.budget
            
        # Initialize
        rng = np.random
        X = lo + (hi - lo) * rng.random((N, self.dim))
        F = np.array([problem(x) for x in X])
        evals = N
        
        start_time = time.time()
        deadline = start_time + 290
        
        while True:
            # Check budget and time
            if evals >= self.budget:
                break
            if time.time() > deadline:
                break
                
            # 1. Sort and Crowding
            ranks, fronts = self._non_dominated_sort(F)
            cd = self._crowding_distance(F, fronts)
            
            # 2. Create Mating Pool (size 2*N)
            # But we might not have enough budget for a full generation.
            # Max offspring we can create in this generation:
            max_offspring = min(2 * N, self.budget - evals)
            # Offspring are created in pairs.
            pairs_needed = max_offspring // 2
            if pairs_needed == 0:
                # If we have 1 eval left, we can't do SBX (needs 2 parents -> 2 children).
                # We could do a simple mutation of one parent.
                # But for simplicity, if we can't make a pair, break.
                # Or, we can make 1 child by mutating a random parent.
                # Let's just break if we can't make a pair, as SBX is the core.
                break

            parent_indices = []
            for _ in range(pairs_needed * 2):
                i = rng.integers(N)
                j = rng.integers(N)
                if self._better(ranks[i], cd[i], ranks[j], cd[j]):
                    parent_indices.append(i)
                else:
                    parent_indices.append(j)
            
            # 3. Generate Offspring
            new_X = []
            new_F = []
            
            for k in range(0, len(parent_indices), 2):
                if evals >= self.budget:
                    break
                p1 = X[parent_indices[k]]
                p2 = X[parent_indices[k+1]]
                
                c1, c2 = self._sbx(p1, p2, lo, hi)
                c1 = self._poly_mutation(c1, lo, hi)
                c2 = self._poly_mutation(c2, lo, hi)
                
                f1 = problem(c1)
                evals += 1
                if evals >= self.budget:
                    # Add only c1
                    new_X.append(c1)
                    new_F.append(f1)
                    break
                f2 = problem(c2)
                evals += 1
                
                new_X.append(c1)
                new_F.append(c2) # Wait, c2 is X, f2 is F.
                new_X.append(c2)
                new_F.append(f2)

            if not new_X:
                break

            new_X = np.array(new_X)
            new_F = np.array(new_F)
            
            # 4. Merge and Select
            total_N = N + len(new_X)
            total_X = np.vstack([X, new_X])
            total_F = np.vstack([F, new_F])
            
            total_ranks, total_fronts = self._non_dominated_sort(total_F)
            total_cd = self._crowding_distance(total_F, total_fronts)
            
            # Select best N
            selected_idx = []
            for front in total_fronts:
                if len(selected_idx) + len(front) <= N:
                    selected_idx.extend(front)
                else:
                    # Sort front by crowding distance descending
                    front_cd = [total_cd[i] for i in front]
                    sorted_front = sorted(front, key=lambda i: total_cd[i], reverse=True)
                    needed = N - len(selected_idx)
                    selected_idx.extend(sorted_front[:needed])
                    break
            
            X = total_X[selected_idx]
            F = total_F[selected_idx]
            
            # Ensure X and F are consistent arrays
            X = np.array(X)
            F = np.array(F)

        return X

    def _better(self, r1, cd1, r2, cd2):
        if r1 < r2:
            return True
        if r2 < r1:
            return False
        return cd1 > cd2

    def _non_dominated_sort(self, F):
        N = len(F)
        ranks = np.zeros(N, dtype=int)
        fronts = []
        remaining = set(range(N))
        
        while remaining:
            front = []
            for i in remaining:
                dominated = False
                for j in remaining:
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    front.append(i)
            
            if not front:
                # Should not happen if we remove dominated ones correctly, 
                # but in case of cycles or errors, add all remaining
                front = list(remaining)
                
            for i in front:
                ranks[i] = len(fronts)
            fronts.append(front)
            remaining -= set(front)
            
        return ranks, fronts

    def _crowding_distance(self, F, fronts):
        N = len(F)
        cd = np.zeros(N)
        
        for front in fronts:
            if len(front) <= 2:
                for i in front:
                    cd[i] = float('inf')
                continue
            
            n_obj = self.n_obj
            for m in range(n_obj):
                # Sort front by objective m
                sorted_front = sorted(front, key=lambda i: F[i, m])
                cd[sorted_front[0]] = float('inf')
                cd[sorted_front[-1]] = float('inf')
                
                f_min = F[sorted_front[0], m]
                f_max = F[sorted_front[-1], m]
                range_m = f_max - f_min
                if range_m == 0:
                    continue
                    
                for k in range(1, len(sorted_front) - 1):
                    i_prev = sorted_front[k-1]
                    i_next = sorted_front[k+1]
                    cd[sorted_front[k]] += (F[i_next, m] - F[i_prev, m]) / range_m
                    
        return cd

    def _sbx(self, p1, p2, lo, hi):
        # Simulated Binary Crossover
        c1 = np.zeros_like(p1)
        c2 = np.zeros_like(p1)
        
        for i in range(self.dim):
            if np.random.random() < 0.5:
                if abs(p1[i] - p2[i]) < 1e-14:
                    c1[i] = p1[i]
                    c2[i] = p2[i]
                else:
                    u = np.random.random()
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (20 + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (20 + 1))
                    
                    c1[i] = 0.5 * ((1 + beta) * p1[i] + (1 - beta) * p2[i])
                    c2[i] = 0.5 * ((1 - beta) * p1[i] + (1 + beta) * p2[i])
                    
                    # Clip
                    c1[i] = np.clip(c1[i], lo[i], hi[i])
                    c2[i] = np.clip(c2[i], lo[i], hi[i])
            else:
                c1[i] = p1[i]
                c2[i] = p2[i]
                
        return c1, c2

    def _poly_mutation(self, x, lo, hi):
        # Polynomial Mutation
        x_mut = x.copy()
        prob_mut = 1.0 / self.dim
        eta_m = 20
        
        for i in range(self.dim):
            if np.random.random() < prob_mut:
                if np.random.random() < 0.5:
                    y = x[i] - lo[i]
                    y = y / (hi[i] - lo[i]) if hi[i] > lo[i] else 0
                    delta1 = (1 + y) ** (eta_m + 1)
                    u = np.random.random()
                    delta = (2 * u + (1 - 2 * u) * delta1) ** (1.0 / (eta_m + 1)) - 1
                    y_new = y * (1 + delta)
                    x_mut[i] = lo[i] + y_new * (hi[i] - lo[i])
                else:
                    y = hi[i] - x[i]
                    y = y / (hi[i] - lo[i]) if hi[i] > lo[i] else 0
                    delta2 = (1 + y) ** (eta_m + 1)
                    u = np.random.random()
                    delta = (2 * u + (1 - 2 * u) * delta2) ** (1.0 / (eta_m + 1)) - 1
                    y_new = y * (1 + delta)
                    x_mut[i] = hi[i] - y_new * (hi[i] - lo[i])
                    
                x_mut[i] = np.clip(x_mut[i], lo[i], hi[i])
                
        return x_mut
