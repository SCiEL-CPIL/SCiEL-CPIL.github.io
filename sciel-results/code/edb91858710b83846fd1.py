import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.N = 12  # Population size
        self.F = 0.8  # DE scaling factor
        self.CR = 0.9  # DE crossover rate

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Initialize population
        X = lo + (hi - lo) * np.random.rand(self.N, self.dim)
        F = np.array([problem(x) for x in X])
        evals = self.N
        
        # Generate reference points
        ref_points = self._generate_reference_points(self.N, self.n_obj)
        
        while evals < self.budget:
            # Generate offspring using DE
            offspring_X = np.zeros_like(X)
            for i in range(self.N):
                # Select 3 distinct random indices
                r1, r2, r3 = np.random.choice(self.N, 3, replace=False)
                # Ensure they are distinct and not i? 
                # Standard DE/rand/1/bin usually picks from whole population.
                # If r1, r2, r3 can be i, it's fine.
                
                v = X[r1] + self.F * (X[r2] - X[r3])
                
                # Crossover
                mask = np.random.rand(self.dim) < self.CR
                # Ensure at least one gene is from v
                if not np.any(mask):
                    mask[np.random.randint(self.dim)] = True
                    
                u = np.where(mask, v, X[i])
                offspring_X[i] = np.clip(u, lo, hi)
            
            # Evaluate offspring
            F_off = np.array([problem(x) for x in offspring_X])
            evals += self.N
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, F_off])
            
            # NSGA-III Selection
            X, F = self._nsga3_select(X_comb, F_comb, ref_points)
            
        return X

    def _generate_reference_points(self, N, n_obj):
        """
        Generate N uniform reference points on the simplex for n_obj objectives.
        Uses a greedy farthest-point sampling from a random set to ensure spread.
        """
        # Generate many random points on the simplex
        # Simplex: sum(x) = 1, x >= 0
        # Method: Generate n_obj-1 uniform randoms, sort, take diffs
        K = 1000
        pts = []
        for _ in range(K):
            # For n_obj=3, we need 2 randoms
            if n_obj == 3:
                r1, r2 = np.random.rand(2)
                if r1 > r2: r1, r2 = r2, r1
                x0 = r1
                x1 = r2 - r1
                x2 = 1.0 - r2
                pts.append([x0, x1, x2])
            else:
                # General case
                r = np.random.rand(n_obj - 1)
                r.sort()
                r = np.concatenate([[0.0], r, [1.0]])
                pts.append(np.diff(r))
        
        pts = np.array(pts)
        
        # Farthest point sampling
        # Start with a random point
        ref_pts = []
        idx = np.random.randint(K)
        ref_pts.append(pts[idx])
        
        for _ in range(N - 1):
            # Calculate min distance from each candidate to existing ref_pts
            dists = np.min([np.linalg.norm(pts - rp, axis=1) for rp in ref_pts], axis=0)
            # Pick the one with max min distance
            next_idx = np.argmax(dists)
            ref_pts.append(pts[next_idx])
            
        return np.array(ref_pts)

    def _nsga3_select(self, X, F, ref_points):
        """
        NSGA-III selection mechanism.
        """
        N_total = len(F)
        N_target = self.N
        
        # 1. Non-dominated sorting
        fronts = self._fast_non_dominated_sort(F)
        
        # 2. Select from fronts
        selected_indices = []
        i = 0
        while i < len(fronts) and len(selected_indices) + len(fronts[i]) <= N_target:
            selected_indices.extend(fronts[i])
            i += 1
            
        if len(selected_indices) == N_target:
            return X[selected_indices], F[selected_indices]
        
        # If we filled up, we are done.
        # If not, we need to select from the next front (fronts[i])
        # to fill the remaining slots.
        
        if i >= len(fronts):
            # Should not happen if N_target <= N_total
            return X[selected_indices], F[selected_indices]
            
        remaining = N_target - len(selected_indices)
        last_front = fronts[i]
        
        # 3. Association with reference points
        # Normalize the last front
        F_last = F[last_front]
        f_min = np.min(F_last, axis=0)
        f_max = np.max(F_last, axis=0)
        f_range = f_max - f_min
        f_range[f_range == 0] = 1.0
        F_norm = (F_last - f_min) / f_range
        
        # Associate each point in last_front with nearest reference point
        # ref_points are on simplex, so they are already normalized (sum=1)
        # But our F_norm sums to n_obj? No, min-max normalization doesn't preserve sum.
        # NSGA-III assumes objectives are normalized such that the ideal point is 0 and nadir is 1?
        # Or it uses the reference points in the normalized space.
        # Standard NSGA-III:
        # 1. Normalize objectives using min/max of the population (or ideal/nadir).
        # 2. Reference points are on the simplex (sum=1).
        # 3. To compare, we need to project the normalized points onto the simplex?
        # Actually, the standard approach is:
        # Let $z^*$ be the ideal point (min of each obj).
        # Let $z^n$ be the nadir point (max of each obj).
        # Normalized $f_i' = (f_i - z^*_i) / (z^n_i - z^*_i)$.
        # This maps to [0,1].
        # Reference points $r$ are on the simplex $\sum r_i = 1$.
        # The distance is calculated in this normalized space.
        
        # Let's recalculate normalization properly for the last front + selected points?
        # NSGA-III usually normalizes using the min/max of the *entire* combined population or just the last front?
        # Using the last front is common for the association step.
        
        # Calculate distances
        # F_norm shape: (L, n_obj) where L is len(last_front)
        # ref_points shape: (N_ref, n_obj)
        
        # We need to find the nearest reference point for each point in F_norm.
        # dist[j, k] = || F_norm[j] - ref_points[k] ||
        
        dists = np.linalg.norm(F_norm[:, np.newaxis, :] - ref_points[np.newaxis, :, :], axis=2)
        nearest_ref_idx = np.argmin(dists, axis=1)
        
        # Count how many points are associated with each reference point
        ref_counts = np.zeros(len(ref_points), dtype=int)
        for idx in nearest_ref_idx:
            ref_counts[idx] += 1
            
        # Selection from last front
        # We need to select 'remaining' points.
        # Strategy: 
        # 1. While remaining > 0:
        #    - Find reference point with min count.
        #    - Among points associated with it, pick the one with min distance to that ref point.
        #    - Add to selected.
        #    - Decrement count.
        
        # To do this efficiently, we can just loop 'remaining' times.
        # Since L is small (<= N_total), this is fine.
        
        selected_in_last = []
        available_points = list(range(len(last_front)))
        
        # We need to track which points are still available
        # And which ref counts are current
        
        # Copy ref_counts
        curr_ref_counts = ref_counts.copy()
        
        for _ in range(remaining):
            if not available_points:
                break
                
            # Find ref point with min count
            min_count = np.min(curr_ref_counts)
            # Get all ref points with this min count
            candidate_refs = np.where(curr_ref_counts == min_count)[0]
            
            # For each candidate ref, find the best point associated with it
            best_point_global = -1
            best_dist = np.inf
            best_ref = -1
            
            for r in candidate_refs:
                # Find points in available_points associated with r
                pts_for_r = [p for p in available_points if nearest_ref_idx[p] == r]
                if not pts_for_r:
                    continue
                
                # Find point with min distance to ref r
                for p in pts_for_r:
                    d = dists[p, r]
                    if d < best_dist:
                        best_dist = d
                        best_point_global = p
                        best_ref = r
            
            if best_point_global == -1:
                # Fallback: pick any available point
                best_point_global = available_points[0]
                best_ref = nearest_ref_idx[best_point_global]
                
            # Select best_point_global
            selected_in_last.append(last_front[best_point_global])
            available_points.remove(best_point_global)
            curr_ref_counts[best_ref] -= 1
            
        selected_indices.extend(selected_in_last)
        
        return X[selected_indices], F[selected_indices]

    def _fast_non_dominated_sort(self, F):
        """
        Returns a list of lists, where each inner list contains indices of a front.
        """
        N = len(F)
        domination_count = np.zeros(N, dtype=int)
        dominated_set = [[] for _ in range(N)]
        
        for i in range(N):
            for j in range(i + 1, N):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        fronts = []
        current_front = [i for i in range(N) if domination_count[i] == 0]
        
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
            
        return fronts
