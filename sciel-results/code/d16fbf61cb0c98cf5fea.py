"""Multi-objective optimizer for RE21 using hypervolume-guided search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Hypervolume-guided multi-objective optimizer with adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_2d(self, points, ref):
        """Compute exact hypervolume for 2D points against reference point."""
        valid = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
        if len(valid) == 0:
            return 0.0
        valid = valid[np.argsort(valid[:, 0])]
        hv = 0.0
        prev_y = ref[1]
        for i in range(len(valid)):
            x = valid[i, 0]
            y = valid[i, 1]
            if y < prev_y:
                hv += (ref[0] - x) * (prev_y - y)
                prev_y = y
        return hv

    def _hv_contrib_2d(self, points, ref):
        """Compute hypervolume contribution of each point in 2D."""
        n = len(points)
        contrib = np.zeros(n)
        valid_mask = (points[:, 0] < ref[0]) & (points[:, 1] < ref[1])
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return contrib
        # Sort valid points by x ascending
        order = valid_idx[np.argsort(points[valid_idx, 0])]
        # For each point, its contribution is the area it uniquely covers
        # In 2D, sort by x asc, y desc (non-dominated front)
        # Contribution of point i = (x_{i+1} - x_i) * (y_i - y_{i+1}) where i+1 is next in sorted order
        # Actually for a sorted list by x ascending, the contribution of point i
        # is the rectangle from (x_i, y_i) to (x_{i+1}, min(y_i, y_{i+1}))
        # But we need the exact HV contribution: the area that would be lost if point i were removed.
        # For 2D with points sorted by x ascending:
        #   contrib[i] = (x[i+1] - x[i]) * (y[i] - y[i+1]) if y[i] > y[i+1]
        #   For the last point: (ref[0] - x[last]) * (y[last] - ref[1])
        #   For the first point: (x[1] - x[0]) * (y[0] - y[1]) ... 
        # More carefully: sort by x ascending. The non-dominated subset has y decreasing.
        # Let's just compute: for each point, remove it and compute HV difference.
        # That's O(n^2 log n) which is fine for n=80.
        full_hv = self._hv_2d(points, ref)
        for i in valid_idx:
            pts_without_i = np.delete(points, i, axis=0)
            hv_without_i = self._hv_2d(pts_without_i, ref)
            contrib[i] = full_hv - hv_without_i
        return contrib

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295.0  # 300s limit, keep margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3000.0, 0.06])
        
        # Population size
        pop_size = 40
        if self.budget < pop_size * 2:
            pop_size = max(10, self.budget // 4)

        # Stratified initialization: divide [0,1]^dim into cells, pick one point per cell
        # For 4D with pop_size=40, use 2x2x2x5 or similar stratification
        # Use a simple stratified approach: for each dimension, divide into k_i equal parts
        # and pick one random point in each cell
        # 40 = 2 * 2 * 2 * 5 or 2 * 2 * 5 * 2 or 4 * 5 * 2 * 1 etc.
        # Let's use 2x2x5x2 = 40
        n_per_dim = [2, 2, 5, 2]
        while len(n_per_dim) < self.dim:
            n_per_dim.append(1)
        n_per_dim = n_per_dim[:self.dim]
        
        X = np.zeros((pop_size, self.dim))
        idx = 0
        for d1 in range(n_per_dim[0]):
            for d2 in range(n_per_dim[1]):
                for d3 in range(n_per_dim[2]):
                    for d4 in range(n_per_dim[3]):
                        if idx >= pop_size:
                            break
                        cell_lo = np.zeros(self.dim)
                        cell_hi = np.zeros(self.dim)
                        for d in range(self.dim):
                            n_d = n_per_dim[d]
                            cell_lo[d] = lo[d] + d * (hi[d] - lo[d]) / n_d
                            cell_hi[d] = lo[d] + (d + 1) * (hi[d] - lo[d]) / n_d
                        X[idx] = cell_lo + (cell_hi - cell_lo) * np.random.rand(self.dim)
                        idx += 1
                    if idx >= pop_size:
                        break
                if idx >= pop_size:
                    break
            if idx >= pop_size:
                break
        
        # Fill remaining if any
        if idx < pop_size:
            X[idx:] = lo + (hi - lo) * np.random.rand(pop_size - idx, self.dim)

        F = np.array([problem(x) for x in X])
        used = pop_size

        # Parameters
        eta_c = 20.0
        eta_m = 20.0
        p_c = 0.9
        p_m_base = 1.0 / self.dim

        # Track best HV so far
        best_hv = self._hv_2d(F, ref)

        while used + pop_size <= self.budget:
            # Time check
            if time.time() > deadline - 3.0:
                break

            # Compute HV contributions for parent selection
            hv_contrib = self._hv_contrib_2d(F, ref)
            
            # Select parents using HV-contribution-weighted tournament
            parents_idx = np.zeros(pop_size, dtype=int)
            for i in range(pop_size):
                idx1 = np.random.randint(pop_size)
                idx2 = np.random.randint(pop_size)
                if hv_contrib[idx1] >= hv_contrib[idx2]:
                    parents_idx[i] = idx1
                else:
                    parents_idx[i] = idx2
            
            # Generate offspring
            offspring = np.zeros((pop_size, self.dim))
            for i in range(0, pop_size, 2):
                if np.random.rand() < p_c:
                    p1 = X[parents_idx[i]]
                    p2 = X[parents_idx[i+1]]
                    c1, c2 = self._sbx_crossover(p1, p2, lo, hi, eta_c)
                    offspring[i] = c1
                    offspring[i+1] = c2
                else:
                    offspring[i] = X[parents_idx[i]].copy()
                    offspring[i+1] = X[parents_idx[i+1]].copy()
            
            # Adaptive mutation: scale based on population diversity
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            # Normalize std to get a diversity measure
            ranges = hi - lo
            diversity = np.mean(X_std / ranges)
            # Scale mutation probability: more diverse -> less mutation needed
            p_m = p_m_base * (1.0 + 0.5 * (1.0 - diversity * 4))
            p_m = np.clip(p_m, 0.05, 0.5)

            for i in range(pop_size):
                offspring[i] = self._polynomial_mutation(offspring[i], lo, hi, eta_m, p_m)
            
            # Evaluate offspring
            F_offspring = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # Combine parents and offspring
            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, F_offspring])
            
            # Select best pop_size using HV contribution
            hv_contrib_comb = self._hv_contrib_2d(F_combined, ref)
            
            # Keep top pop_size by HV contribution
            top_indices = np.argsort(-hv_contrib_comb)[:pop_size]
            
            X = X_combined[top_indices]
            F = F_combined[top_indices]
            
            # Update best HV
            current_hv = self._hv_2d(F, ref)
            if current_hv > best_hv:
                best_hv = current_hv

        return X

    def _non_dominated_sort(self, F):
        """
        Compute non-dominated ranks and crowding distances.
        Returns:
            ranks: array of ranks (0 is best)
            crowding: array of crowding distances
        """
        n = len(F)
        ranks = np.zeros(n, dtype=int)
        crowding = np.zeros(n)
        
        # Non-dominated sorting
        S = [[] for _ in range(n)]
        n_dominates = np.zeros(n, dtype=int)
        front = []
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[i] += 1
        
        for i in range(n):
            if n_dominates[i] == 0:
                ranks[i] = 0
                front.append(i)
        
        k = 0
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dominates[j] -= 1
                    if n_dominates[j] == 0:
                        ranks[j] = k + 1
                        next_front.append(j)
            k += 1
            front = next_front
        
        # Calculate crowding distance for each front
        max_rank = max(ranks) if n > 0 else 0
        fronts = [[] for _ in range(max_rank + 1)]
        for i in range(n):
            fronts[ranks[i]].append(i)
        
        for front_indices in fronts:
            if len(front_indices) <= 2:
                for i in front_indices:
                    crowding[i] = np.inf
                continue
            
            m = self.n_obj
            for k_obj in range(m):
                sorted_idx = sorted(front_indices, key=lambda x: F[x, k_obj])
                crowding[sorted_idx[0]] = np.inf
                crowding[sorted_idx[-1]] = np.inf
                f_min = F[sorted_idx[0], k_obj]
                f_max = F[sorted_idx[-1], k_obj]
                if f_max - f_min == 0:
                    continue
                for i in range(1, len(sorted_idx) - 1):
                    idx = sorted_idx[i]
                    idx_prev = sorted_idx[i-1]
                    idx_next = sorted_idx[i+1]
                    crowding[idx] += (F[idx_next, k_obj] - F[idx_prev, k_obj]) / (f_max - f_min)
        
        return ranks, crowding

    def _sbx_crossover(self, p1, p2, lo, hi, eta):
        c1 = p1.copy()
        c2 = p2.copy()
        for i in range(self.dim):
            if np.random.rand() < 0.5:
                if abs(p2[i] - p1[i]) < 1e-12:
                    beta = 1.0
                else:
                    r = np.random.rand()
                    if r <= 0.5:
                        alpha = 2.0 * r
                        beta = (alpha) ** (1.0 / (eta + 1.0))
                    else:
                        alpha = 2.0 * (1.0 - r)
                        beta = (alpha) ** (1.0 / (eta + 1.0))
                
                c1[i] = 0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i])
                c2[i] = 0.5 * ((1.0 - beta) * p1[i] + (1.0 + beta) * p2[i])
        
        c1 = np.clip(c1, lo, hi)
        c2 = np.clip(c2, lo, hi)
        return c1, c2

    def _polynomial_mutation(self, x, lo, hi, eta, p_m):
        x_mut = x.copy()
        for i in range(self.dim):
            if np.random.rand() < p_m:
                range_i = hi[i] - lo[i]
                if range_i == 0:
                    continue
                x_norm = (x[i] - lo[i]) / range_i
                
                y_low = 0.0
                y_high = 1.0
                
                r = np.random.rand()
                if r < 0.5:
                    xy = 2.0 * r
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * (y_high - y_low)
                    y_new = x_norm - delta
                else:
                    xy = 2.0 * (1.0 - r)
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * (y_high - y_low)
                    y_new = x_norm + delta
                
                y_new = np.clip(y_new, 0.0, 1.0)
                x_mut[i] = lo[i] + y_new * range_i
        
        return x_mut
# EVOLVE-BLOCK-END
