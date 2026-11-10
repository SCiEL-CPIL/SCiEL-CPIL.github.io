import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _dominates(self, a, b):
        return np.all(a <= b) and np.any(a < b)

    def _non_dominated_sort(self, F):
        # Standard fast non-dominated sort
        n = len(F)
        S = [[] for _ in range(n)]
        nd = [0] * n
        rank = [0] * n
        fronts = []
        
        for i in range(n):
            for j in range(n):
                if i == j: continue
                if self._dominates(F[i], F[j]):
                    S[i].append(j)
                elif self._dominates(F[j], F[i]):
                    nd[i] += 1
            if nd[i] == 0:
                rank[i] = 0
                fronts.append([i])
        
        front_idx = 0
        while front_idx < len(fronts):
            next_front = []
            for i in fronts[front_idx]:
                for j in S[i]:
                    nd[j] -= 1
                    if nd[j] == 0:
                        rank[j] = front_idx + 1
                        next_front.append(j)
            front_idx += 1
            if next_front:
                fronts.append(next_front)
        return fronts

    def _crowding_distance(self, F, indices):
        # Calculate crowding distance for points in indices
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        cd = np.zeros(n)
        for m in range(self.n_obj):
            # Sort indices by objective m
            sorted_idx = sorted(indices, key=lambda i: F[i][m])
            # Assign inf to best and worst
            cd[sorted_idx[0] - indices[0]] = np.inf # Note: indices are global, cd is local to indices list? 
            # Let's make cd a dict or map back.
            # Better: return a list of (global_idx, cd)
            
        # Let's implement it to return a dict {global_idx: distance}
        cd_dict = {i: 0.0 for i in indices}
        if n <= 2:
            for i in indices:
                cd_dict[i] = np.inf
            return cd_dict

        for m in range(self.n_obj):
            sorted_idx = sorted(indices, key=lambda i: F[i][m])
            cd_dict[sorted_idx[0]] = np.inf
            cd_dict[sorted_idx[-1]] = np.inf
            f_min = F[sorted_idx[0]][m]
            f_max = F[sorted_idx[-1]][m]
            f_range = f_max - f_min
            if f_range == 0:
                continue
            for k in range(1, n - 1):
                i_curr = sorted_idx[k]
                i_prev = sorted_idx[k-1]
                i_next = sorted_idx[k+1]
                cd_dict[i_curr] += (F[i_next][m] - F[i_prev][m]) / f_range
                
        return cd_dict

    def __call__(self, problem, bounds):
        deadline = time.time() + 280.0
        dim, n_obj = self.dim, self.n_obj
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        rng = np.random
        
        pop_size = 20
        X = np.zeros((pop_size, dim))
        F = np.zeros((pop_size, n_obj))
        evals = 0
        
        # 1. Initialization: LHS
        lhs = np.zeros((pop_size, dim))
        for d in range(dim):
            bins = np.linspace(0, 1, pop_size + 1)
            pts = rng.uniform(bins[:-1], bins[1:])
            rng.shuffle(pts)
            lhs[:, d] = pts
        X = lo + lhs * span
        for j in range(pop_size):
            F[j] = problem(X[j])
            evals += 1

        # 2. NSGA-II Loop
        while evals < self.budget - 50: # Reserve 50 for polish
            if time.time() > deadline:
                break
            
            # Sort
            fronts = self._non_dominated_sort(F)
            
            # Select parents from first front using crowding
            if len(fronts[0]) > 0:
                cd = self._crowding_distance(F, fronts[0])
                # Probability proportional to crowding distance? Or just pick random from top?
                # Let's pick 2 parents randomly from the first front, weighted by crowding.
                # For simplicity, random selection from first front.
                p1_idx = fronts[0][rng.randint(len(fronts[0]))]
                p2_idx = fronts[0][rng.randint(len(fronts[0]))]
            else:
                p1_idx = rng.randint(pop_size)
                p2_idx = rng.randint(pop_size)
            
            # DE Variation
            # x_child = x_p1 + F * (x_p2 - x_p3)
            p3_idx = rng.randint(pop_size)
            F_factor = 0.5 + 0.5 * rng.random()
            CR = 0.5
            
            child = X[p1_idx].copy()
            for d in range(dim):
                if rng.random() < CR:
                    child[d] = X[p1_idx][d] + F_factor * (X[p2_idx][d] - X[p3_idx][d])
                else:
                    child[d] = X[p1_idx][d]
            
            # Clip
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Replacement
            # Check if child is dominated
            dominated = False
            for j in range(pop_size):
                if self._dominates(F[j], fc):
                    dominated = True
                    break
            
            if not dominated:
                # Find point to replace: worst in last front
                # Re-sort to get latest fronts
                fronts = self._non_dominated_sort(F)
                last_front = fronts[-1]
                if len(last_front) > 1:
                    cd = self._crowding_distance(F, last_front)
                    # Pick the one with min crowding
                    worst_idx = min(last_front, key=lambda i: cd[i])
                else:
                    worst_idx = last_front[0]
                
                X[worst_idx] = child
                F[worst_idx] = fc

        # 3. Local Polish
        # Get non-dominated points
        fronts = self._non_dominated_sort(F)
        nd_indices = fronts[0]
        
        polish_budget = self.budget - evals
        if polish_budget > 0 and len(nd_indices) > 0:
            # Distribute budget among non-dominated points
            n_nd = len(nd_indices)
            evals_per_point = max(1, polish_budget // n_nd)
            
            for idx in nd_indices:
                if evals >= self.budget:
                    break
                if time.time() > deadline:
                    break
                    
                best_child = X[idx].copy()
                best_child_F = F[idx].copy()
                
                for _ in range(evals_per_point):
                    if evals >= self.budget:
                        break
                    # Small Gaussian step
                    sigma = 0.05 * span
                    child = X[idx] + sigma * rng.standard_normal(dim)
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    evals += 1
                    
                    # Accept if it dominates the current best child or the parent
                    # To keep it simple, accept if it is non-dominated in the global set?
                    # No, local polish should just try to improve this specific point.
                    # If child dominates best_child_F, update.
                    if self._dominates(fc, best_child_F):
                        best_child = child
                        best_child_F = fc
                    # Also, if child is non-dominated globally, it's a good candidate.
                    # But we want to improve the front.
                
                # Update the population with the polished point
                X[idx] = best_child
                F[idx] = best_child_F

        return X
