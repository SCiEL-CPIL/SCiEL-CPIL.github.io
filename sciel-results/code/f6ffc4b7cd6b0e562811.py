"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """NSGA-II style multi-objective optimizer for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (each a list of indices), sorted by dominance rank."""
        n = len(F)
        domination_count = np.zeros(n, dtype=int)
        dominated_set = [list() for _ in range(n)]
        
        for i in range(n):
            for j in range(i + 1, n):
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1
        
        fronts = []
        current_front = [i for i in range(n) if domination_count[i] == 0]
        while current_front:
            fronts.append(current_front)
            next_front = []
            for i in current_front:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front = next_front
        
        # Add any remaining points (shouldn't happen but safety)
        assigned = set()
        for front in fronts:
            assigned.update(front)
        if len(assigned) < n:
            remaining = [i for i in range(n) if i not in assigned]
            fronts.append(remaining)
        
        return fronts

    def _calc_hv_3d(self, points, ref):
        """Calculate 3D hypervolume for a set of non-dominated points."""
        # Filter points strictly better than ref
        points = points[np.all(points < ref, axis=1)]
        n = len(points)
        if n == 0:
            return 0.0
        
        # Sort by f1
        points = points[np.argsort(points[:, 0])]
        
        active = []
        vol = 0.0
        
        for i in range(n):
            p = points[i]
            # Add p to active, removing dominated ones in f2, f3
            new_active = []
            for q in active:
                # If p dominates q in f2, f3
                if not (p[1] <= q[1] and p[2] <= q[2] and (p[1] < q[1] or p[2] < q[2])):
                    new_active.append(q)
            new_active.append(p)
            active = new_active
            
            # Calculate width
            if i < n - 1:
                width = points[i+1, 0] - p[0]
            else:
                width = ref[0] - p[0]
                
            # Calculate 2D HV of active set
            # Active set is non-dominated in 2D (f2, f3)
            # Sort by f2
            active_sorted = sorted(active, key=lambda x: x[1])
            hv2 = 0.0
            for j in range(len(active_sorted)):
                f2, f3 = active_sorted[j]
                if j < len(active_sorted) - 1:
                    next_f2 = active_sorted[j+1][1]
                else:
                    next_f2 = ref[1]
                
                hv2 += (next_f2 - f2) * (ref[2] - f3)
            
            vol += width * hv2
            
        return vol

    def _hv_contributions(self, F, ref):
        """Calculate hypervolume contribution of each point in a non-dominated front."""
        n = len(F)
        if n == 0:
            return np.array([])
        
        # Filter valid points
        valid_mask = np.all(F < ref, axis=1)
        if not np.any(valid_mask):
            return np.zeros(n)
            
        F_valid = F[valid_mask]
        total_hv = self._calc_hv_3d(F_valid, ref)
        
        contributions = np.zeros(n)
        
        # Calculate contribution for each valid point
        # To avoid O(N^3) complexity, we can just do it for N=50.
        # It's O(N^2) per point, so O(N^3) total. 125k ops.
        
        for i in range(n):
            if not valid_mask[i]:
                contributions[i] = 0.0
                continue
            
            # Remove point i from F_valid
            # Note: F_valid indices are shifted relative to F
            # We need to map back or just work with F_valid indices
            # Let's work with F_valid directly and map back later
            pass
            
        # Better approach:
        # Create a list of valid indices
        valid_indices = np.where(valid_mask)[0]
        F_valid_indices = valid_indices
        
        # Calculate HV for full set
        # Then for each point, calculate HV without it
        # This is expensive if done naively.
        
        # Optimization:
        # We can just compute the contribution directly during the sweep?
        # Or just accept the O(N^3) cost for N=50.
        
        # Let's do the naive way for simplicity and correctness first.
        # N=50, N^3 = 125,000.
        # Each HV calc is O(N^2) = 2500.
        # 50 * 2500 = 125,000 operations.
        # In Python, this might take 0.5 - 1 second.
        # We do this once per generation (4-5 times). Total 2-5 seconds.
        # Acceptable.
        
        for i in range(n):
            if not valid_mask[i]:
                continue
            
            # Create subset without i
            mask = np.ones(n, dtype=bool)
            mask[i] = False
            F_subset = F[mask]
            hv_subset = self._calc_hv_3d(F_subset, ref)
            contributions[i] = total_hv - hv_subset
            
        return contributions

    def _tournament_select(self, rank, cd, F, X, k=2):
        """Tournament selection based on rank and crowding distance."""
        n = len(F)
        idx1 = np.random.randint(n)
        idx2 = np.random.randint(n)
        if rank[idx1] < rank[idx2]:
            return idx1
        elif rank[idx2] < rank[idx1]:
            return idx2
        else:
            if cd[idx1] > cd[idx2]:
                return idx1
            else:
                return idx2

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 280.0  # safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        pop_size = min(50, self.budget // 4)
        pop_size = max(pop_size, 10)
        
        # Initialize population
        X = lo + range_ * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_X = [X.copy()]
        all_F = [F.copy()]
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            fronts = self._fast_non_dominated_sort(F)
            
            # Assign ranks
            rank = np.zeros(len(F), dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Compute crowding distance for each front
            cd = np.zeros(len(F))
            for front in fronts:
                front_cd = self._crowding_distance(F, front)
                for j, idx in enumerate(front):
                    cd[idx] = front_cd[j]
            
            # Generate offspring
            n_offspring = pop_size
            offspring_X = []
            offspring_F = []
            
            for _ in range(n_offspring):
                if used >= self.budget:
                    break
                
                # Select parents
                p1 = self._tournament_select(rank, cd, F, X)
                p2 = self._tournament_select(rank, cd, F, X)
                
                # Simulated binary crossover (SBX)
                if rng.random() < 0.9:
                    beta = 15
                    child1 = np.zeros(self.dim)
                    child2 = np.zeros(self.dim)
                    for d in range(self.dim):
                        if rng.random() < 0.5:
                            if abs(X[p1, d] - X[p2, d]) < 1e-15:
                                child1[d] = X[p1, d]
                                child2[d] = X[p2, d]
                            else:
                                u = rng.random()
                                if u <= 0.5:
                                    alpha = 2 * u
                                    factor1 = (alpha + 1) ** (-1.0 / (beta + 1))
                                    factor2 = ((1 - alpha) + 1) ** (-1.0 / (beta + 1))
                                else:
                                    alpha = 2 * (1 - u)
                                    factor1 = (alpha + 1) ** (-1.0 / (beta + 1))
                                    factor2 = ((1 - alpha) + 1) ** (-1.0 / (beta + 1))
                                
                                x1 = min(X[p1, d], X[p2, d])
                                x2 = max(X[p1, d], X[p2, d])
                                child1[d] = 0.5 * ((1 + factor1) * x1 + (1 - factor1) * x2)
                                child2[d] = 0.5 * ((1 + factor2) * x2 + (1 - factor2) * x1)
                        else:
                            child1[d] = X[p1, d]
                            child2[d] = X[p2, d]
                    
                    # Polynomial mutation
                    eta = 20
                    for d in range(self.dim):
                        if rng.random() < 0.9:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                            child1[d] += delta_q * (hi[d] - lo[d])
                        
                        if rng.random() < 0.9:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (eta + 1)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (eta + 1))
                            child2[d] += delta_q * (hi[d] - lo[d])
                    
                    child1 = np.clip(child1, lo, hi)
                    child2 = np.clip(child2, lo, hi)
                    
                    fc1 = problem(child1)
                    used += 1
                    offspring_X.append(child1)
                    offspring_F.append(fc1)
                    
                    if used < self.budget:
                        fc2 = problem(child2)
                        used += 1
                        offspring_X.append(child2)
                        offspring_F.append(fc2)
                else:
                    # Just mutate
                    child = X[p1].copy()
                    for d in range(self.dim):
                        if rng.random() < 0.9:
                            u = rng.random()
                            if u < 0.5:
                                delta_q = (2 * u) ** (1.0 / (21)) - 1
                            else:
                                delta_q = 1 - (2 * (1 - u)) ** (1.0 / (21))
                            child[d] += delta_q * (hi[d] - lo[d])
                    child = np.clip(child, lo, hi)
                    fc = problem(child)
                    used += 1
                    offspring_X.append(child)
                    offspring_F.append(fc)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine parent and offspring
            combined_X = np.vstack([X, offspring_X])
            combined_F = np.vstack([F, offspring_F])
            n_combined = len(combined_F)
            
            # Non-dominated sort on combined
            fronts = self._fast_non_dominated_sort(combined_F)
            
            # Assign ranks
            rank = np.zeros(n_combined, dtype=int)
            for i, front in enumerate(fronts):
                for idx in front:
                    rank[idx] = i
            
            # Compute Hypervolume contributions for the first front
            ref = np.array([3.0, 5.0, 7.0])
            
            # We only need HV contributions for the first front (non-dominated)
            # For other fronts, we can use a simple distance or just take them all if they fit.
            # Standard NSGA-II takes fronts in order.
            
            # Pre-compute HV contributions for Front 0
            front0 = fronts[0]
            if len(front0) > 0:
                F_front0 = combined_F[front0]
                hv_contribs = self._hv_contributions(F_front0, ref)
                # Map hv_contribs back to indices in front0
                # hv_contribs is aligned with F_front0 rows
                # We need to sort front0 indices by hv_contribs descending
                order_front0 = np.argsort(-hv_contribs)
                # Reorder front0 indices
                front0_sorted = [front0[i] for i in order_front0]
                # Replace fronts[0] with sorted version?
                # Or just handle it in the selection loop.
            else:
                front0_sorted = []
            
            # Select next generation
            new_X = np.zeros((pop_size, self.dim))
            new_F = np.zeros((pop_size, self.n_obj))
            count = 0
            for f_idx, front in enumerate(fronts):
                if f_idx == 0:
                    # Use HV sorted order
                    selected_indices = front0_sorted
                else:
                    # For other fronts, just take them in original order (or random)
                    # Crowding distance is not computed for them to save time.
                    # This is a slight deviation from NSGA-II but acceptable for small budget.
                    selected_indices = front
                
                if count + len(selected_indices) <= pop_size:
                    for idx in selected_indices:
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                else:
                    # Partial selection
                    remaining = pop_size - count
                    for k in range(min(remaining, len(selected_indices))):
                        idx = selected_indices[k]
                        new_X[count] = combined_X[idx]
                        new_F[count] = combined_F[idx]
                        count += 1
                    break
            
            X = new_X[:count]
            F = new_F[:count]
            
            # Update archive of all evaluated points
            all_X.append(X.copy())
            all_F.append(F.copy())
        
        return X
# EVOLVE-BLOCK-END
