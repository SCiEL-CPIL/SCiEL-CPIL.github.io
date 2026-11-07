"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """NSGA-II with Hypervolume-guided selection and adaptive mutation."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # Reference point for hypervolume calculation (from task description)
        ref = np.array([3000.0, 0.06])

        # Population size
        pop_size = 40
        if self.budget < pop_size * 2:
             pop_size = max(10, self.budget // 4)

        # Initialize population
        X = lo + (hi - lo) * np.random.rand(pop_size, self.dim)
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for final hypervolume calculation
        # But we only need to return the final population or all points?
        # Task says: "Every returned objective vector is recorded".
        # The evaluator records the return value of __call__? No, "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded" -> This implies the evaluator intercepts the `problem` calls?
        # No, "problem(x) ... returns 2 objective values."
        # "Every returned objective vector is recorded" likely refers to the output of the `problem` function calls made by the optimizer.
        # The evaluator wraps `problem` or counts calls.
        # "The return value of __call__ is ignored."
        # So I just need to make sure I call `problem` correctly and don't exceed budget.
        # The score is based on "all recorded vectors". This means ALL points evaluated by `problem` are part of the set for hypervolume?
        # "Scoring ... exact hypervolume of all recorded vectors".
        # This is crucial. It means the union of ALL evaluated points counts, not just the final population.
        # So I should keep track of all evaluated points to ensure I don't miss any, but I don't need to return them.
        # However, the evaluator records them. I just need to call problem.
        
        # NSGA-II parameters
        eta_c = 20.0  # Distribution index for crossover
        eta_m = 20.0  # Distribution index for mutation
        p_c = 0.9     # Crossover probability

        # Adaptive mutation scale
        mutation_scale = 0.5
        min_mutation_scale = 0.05
        
        while used + pop_size <= self.budget:
            # Check time guard
            if time.time() > deadline:
                break

            # 1. Compute ranks and crowding for current X, F
            # We will use a modified selection: 
            # First, standard NSGA-II non-dominated sort to get fronts.
            # Then, for the first front (Pareto front), we will select parents based on Hypervolume Contribution.
            
            ranks, crowding = self._non_dominated_sort(F)
            
            # Identify the first front (rank 0)
            front0_indices = np.where(ranks == 0)[0]
            
            # Calculate HV contributions for points in the first front
            # To be efficient, we only calculate HV contribution for the non-dominated points.
            # For dominated points, their contribution is 0.
            
            # We need to select 2 * pop_size parents? No, NSGA-II selects pop_size parents to produce pop_size offspring.
            # Standard NSGA-II: Tournament selection on the combined population.
            # Here, we want to bias selection towards high HV contribution points.
            
            # Let's compute HV contribution for all points in the current population?
            # That's expensive O(N^2) or O(N log N). N=40, so it's fine.
            
            hv_contribs = np.zeros(pop_size)
            # Only non-dominated points have non-zero HV contribution
            for idx in front0_indices:
                hv_contribs[idx] = self._hv_contribution_2d(F[idx], F, ref)
            
            # Normalize contributions to probabilities
            total_contrib = np.sum(hv_contribs)
            if total_contrib < 1e-10:
                probs = np.ones(pop_size) / pop_size
            else:
                probs = hv_contribs / total_contrib
                # Ensure no zero probabilities for numerical stability? 
                # If a point has 0 contribution, it shouldn't be selected.
                # But if all are 0, we fall back to uniform.
                # If some are 0, they just have 0 probability.
                # However, we need to ensure that we can actually select parents.
                # If a point has 0 probability, it will never be selected. That's fine.
                
            # Select parents using weighted random selection (equivalent to tournament on HV contribution)
            # We need pop_size parents.
            # Standard NSGA-II uses binary tournament. 
            # Here we use direct weighted sampling.
            
            parents_idx = np.zeros(pop_size, dtype=int)
            # To ensure diversity, we might use a mix of HV-guided and random?
            # Let's stick to HV-guided for the first front, and random for others?
            # Actually, if we only select from front 0, we might lose diversity in later fronts.
            # But HV contribution is 0 for dominated points.
            # So dominated points will never be selected as parents.
            # This is aggressive but focuses search on the Pareto front.
            # To maintain diversity, we can use a small epsilon for dominated points?
            # Or, we can use standard NSGA-II tournament but with a twist:
            # In tournament, if two points are from the same front, compare HV contribution instead of crowding?
            # Or, simpler: use the HV contribution as the fitness for a roulette wheel selection.
            
            # Let's use roulette wheel selection for parents.
            # But we need to handle the case where probs sum to 1.
            
            # One issue: if we only select from front 0, we might stagnate.
            # Let's add a small probability to all points to maintain exploration.
            # p_min = 0.1 / pop_size
            # p_max = 1 - p_min * pop_size
            # This is getting complicated.
            
            # Alternative: Use NSGA-II standard selection, but replace crowding distance with HV contribution for the first front.
            # For other fronts, use crowding distance.
            
            # Let's implement a custom tournament:
            # 1. Pick 2 random individuals.
            # 2. If ranks are different, pick lower rank.
            # 3. If ranks are same:
            #    a. If rank is 0, pick higher HV contribution.
            #    b. If rank > 0, pick higher crowding distance.
            
            parents_idx = np.zeros(pop_size, dtype=int)
            for i in range(pop_size):
                idx1 = np.random.randint(pop_size)
                idx2 = np.random.randint(pop_size)
                
                if ranks[idx1] < ranks[idx2]:
                    parents_idx[i] = idx1
                elif ranks[idx2] < ranks[idx1]:
                    parents_idx[i] = idx2
                else:
                    if ranks[idx1] == 0:
                        # Compare HV contribution
                        if hv_contribs[idx1] >= hv_contribs[idx2]:
                            parents_idx[i] = idx1
                        else:
                            parents_idx[i] = idx2
                    else:
                        # Compare crowding
                        if crowding[idx1] >= crowding[idx2]:
                            parents_idx[i] = idx1
                        else:
                            parents_idx[i] = idx2
            
            # 3. Generate offspring
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
            
            # 4. Mutation
            # Adaptive mutation: decrease scale as we get closer to deadline or after many iterations?
            # Let's decrease mutation scale slowly.
            for i in range(pop_size):
                offspring[i] = self._polynomial_mutation(offspring[i], lo, hi, eta_m, mutation_scale)
            
            # 5. Evaluate offspring
            F_offspring = np.array([problem(x) for x in offspring])
            used += pop_size
            
            # 6. Combine parents and offspring
            X_combined = np.vstack([X, offspring])
            F_combined = np.vstack([F, F_offspring])
            
            # 7. Non-dominated sort and selection
            # For selection of the next generation, we again use HV contribution for the first front.
            ranks_comb, crowding_comb = self._non_dominated_sort(F_combined)
            
            # Calculate HV contributions for the combined population's first front
            front0_comb_indices = np.where(ranks_comb == 0)[0]
            hv_contribs_comb = np.zeros(len(X_combined))
            for idx in front0_comb_indices:
                hv_contribs_comb[idx] = self._hv_contribution_2d(F_combined[idx], F_combined, ref)
            
            # Select best pop_size individuals
            # We need to sort by rank, then by (HV contribution if rank 0 else Crowding) descending.
            
            # Create a sort key
            # Rank is primary.
            # Secondary key: if rank 0, use -HV_contrib (since we want to sort descending, negating makes it ascending for argsort)
            # if rank > 0, use -Crowding
            # But we can't easily mix these in a single lexsort without a unified metric.
            
            # Alternative: 
            # 1. Get all indices sorted by rank.
            # 2. For each rank, sort by the appropriate metric.
            # 3. Concatenate.
            
            sorted_indices = []
            max_rank = np.max(ranks_comb)
            for r in range(max_rank + 1):
                idx_in_rank = np.where(ranks_comb == r)[0]
                if len(idx_in_rank) == 0:
                    continue
                
                if r == 0:
                    # Sort by HV contribution descending
                    # If HV contribution is 0, use crowding as tie-breaker?
                    # Let's just use HV contribution.
                    # To handle ties or zeros, we can add a small random noise or use crowding as secondary.
                    # Let's use (-hv_contribs_comb[idx_in_rank], -crowding_comb[idx_in_rank])
                    sort_keys = np.lexsort((-crowding_comb[idx_in_rank], -hv_contribs_comb[idx_in_rank]))
                    sorted_in_rank = idx_in_rank[sort_keys]
                else:
                    # Sort by Crowding descending
                    sort_keys = np.lexsort((-crowding_comb[idx_in_rank],))
                    sorted_in_rank = idx_in_rank[sort_keys]
                
                sorted_indices.append(sorted_in_rank)
            
            sorted_indices = np.concatenate(sorted_indices)
            
            # Select first pop_size
            X = X_combined[sorted_indices[:pop_size]]
            F = F_combined[sorted_indices[:pop_size]]
            
            # Decrease mutation scale
            mutation_scale *= 0.95
            if mutation_scale < min_mutation_scale:
                mutation_scale = min_mutation_scale

        return X

    def _hv_2d(self, points, ref):
        """Compute exact hypervolume for 2D points against reference point."""
        # Filter points that are strictly better than reference in all objectives
        valid = points[(points[:, 0] < ref[0]) & (points[:, 1] < ref[1])]
        if len(valid) == 0:
            return 0.0
        
        # Sort by first objective (ascending)
        valid = valid[np.argsort(valid[:, 0])]
        
        hv = 0.0
        prev_y = ref[1]
        for i in range(len(valid)):
            x, y = valid[i]
            if y < prev_y:
                width = ref[0] - x
                height = prev_y - y
                hv += width * height
                prev_y = y
        return hv

    def _hv_contribution_2d(self, point, front, ref):
        """
        Compute hypervolume contribution of a single point to the given front.
        front: array of shape (N, 2)
        point: array of shape (2,)
        """
        # Check if point is dominated by any point in front. If so, contribution is 0.
        # Actually, if point is dominated, it's not in the non-dominated set, so its contribution to the HV of the set 
        # formed by adding it is 0.
        # But we are calling this for points in the front (or combined pop).
        # If the point is dominated by others in `front`, its contribution is 0.
        
        # Efficient check:
        # A point p has non-zero contribution if and only if it is not dominated by any other point in the set.
        # Since we are computing this for points in the first front (rank 0) of the population, they are non-dominated.
        # However, in `_non_dominated_sort`, we identify rank 0.
        # So we can assume the point is non-dominated.
        
        # To calculate contribution:
        # HV(front ∪ {p}) - HV(front)
        # But `front` here is the entire population? Or just the non-dominated set?
        # In my call `self._hv_contribution_2d(F[idx], F, ref)`, `F` is the whole population.
        # If `F` contains dominated points, they don't affect HV.
        # So HV(F) == HV(NonDominated(F)).
        # And HV(F ∪ {p}) == HV(NonDominated(F ∪ {p})).
        
        # We can optimize by only considering the non-dominated points of `front`.
        # But `front` is passed as the whole population array.
        # Let's extract non-dominated points from `front` first?
        # That's expensive if done for every point.
        
        # Alternative:
        # The contribution of a non-dominated point p to a set S is the area of the region 
        # dominated by p but not dominated by any point in S.
        
        # In 2D, if points are sorted by x (obj1), the contribution of p_i is:
        # (x_{i-1} - x_i) * (y_{i+1} - y_i) ... roughly.
        
        # Let's just compute HV with and without p, but restrict to non-dominated points of `front`.
        
        # Get non-dominated points from `front`
        # We can use a quick non-dominated filter.
        n = len(front)
        nd_mask = np.ones(n, dtype=bool)
        for i in range(n):
            if not nd_mask[i]:
                continue
            for j in range(n):
                if i == j:
                    continue
                if np.all(front[j] <= front[i]) and np.any(front[j] < front[i]):
                    nd_mask[i] = False
                    break
        
        nd_front = front[nd_mask]
        
        # Check if p is dominated by nd_front. If yes, return 0.
        # (It shouldn't be, since p is from rank 0, but just in case)
        for f in nd_front:
            if np.all(f <= point) and np.any(f < point):
                return 0.0
        
        # Calculate HV of nd_front
        hv_without = self._hv_2d(nd_front, ref)
        
        # Calculate HV of nd_front ∪ {p}
        # We can just add p and call _hv_2d again? 
        # _hv_2d handles sorting and filtering.
        # But we need to be careful if p is identical to a point in nd_front.
        # If p is in nd_front, then HV_with == HV_without, contribution 0?
        # No, if p is already in the set, removing it and adding it back changes nothing.
        # But we are calculating contribution of p *to* the set.
        # If p is already in the set, its marginal contribution is 0?
        # Wait, the definition of contribution is usually HV(S ∪ {p}) - HV(S).
        # If p ∈ S, then S ∪ {p} = S, so contribution is 0.
        # This is correct. A point already in the front doesn't add new HV.
        # But we are using this for selection. We want to select points that *would* add HV if they were in a smaller set?
        # No, we are selecting parents from the current population.
        # If a point is in the Pareto front, it has already "claimed" its HV.
        # The contribution metric is often used to identify "redundant" points or to guide search towards gaps.
        # In the context of parent selection, selecting points with high HV contribution might mean selecting points that are "extreme" or in sparse regions.
        
        # Let's proceed with the calculation.
        # Create array with p added
        new_front = np.vstack([nd_front, [point]])
        hv_with = self._hv_2d(new_front, ref)
        
        return max(0.0, hv_with - hv_without)

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
        S = [[] for _ in range(n)] # S[i] = set of individuals dominated by i
        n_dominates = np.zeros(n, dtype=int) # n_dominates[i] = number of individuals dominating i
        front = []
        
        for i in range(n):
            for j in range(i + 1, n):
                # Check if i dominates j
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dominates[j] += 1
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    S[j].append(i)
                    n_dominates[i] += 1
        
        # Find first front
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

    def _polynomial_mutation(self, x, lo, hi, eta, scale_factor=1.0):
        x_mut = x.copy()
        p_m = 1.0 / self.dim
        for i in range(self.dim):
            if np.random.rand() < p_m:
                range_i = hi[i] - lo[i]
                if range_i == 0:
                    continue
                x_norm = (x[i] - lo[i]) / range_i
                
                y_low = 0.0
                y_high = 1.0
                delta_2 = y_high - y_low
                
                r = np.random.rand()
                if r < 0.5:
                    xy = 2.0 * r
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * delta_2 * scale_factor
                    y_new = x_norm - delta
                else:
                    xy = 2.0 * (1.0 - r)
                    c = (xy) ** (1.0 / (eta + 1.0))
                    delta = c * delta_2 * scale_factor
                    y_new = x_norm + delta
                
                y_new = np.clip(y_new, 0.0, 1.0)
                x_mut[i] = lo[i] + y_new * range_i
        
        return x_mut
# EVOLVE-BLOCK-END
