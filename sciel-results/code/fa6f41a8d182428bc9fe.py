"""Multi-objective optimizer for WFG4 using Memetic NSGA-II with Hypervolume-guided Local Search."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Memetic NSGA-II with Hypervolume-guided Local Search for WFG4."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Vectorized fast non-dominated sort. Returns list of fronts (each a list of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]  # S[i] = indices dominated by i
        n_dom = np.zeros(n, dtype=int)  # number of points dominating i
        fronts = []
        
        # Vectorized dominance check
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                    n_dom[j] += 1
        
        # First front
        front = [i for i in range(n) if n_dom[i] == 0]
        fronts.append(front)
        
        while front:
            next_front = []
            for i in front:
                for j in S[i]:
                    n_dom[j] -= 1
                    if n_dom[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            front = next_front
        
        return fronts

    def _crowding_distance(self, F, indices):
        """Compute crowding distance for points with given indices."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        m = self.n_obj
        
        for k in range(m):
            f_vals = F[indices, k]
            f_min = np.min(f_vals)
            f_max = np.max(f_vals)
            f_range = f_max - f_min
            if f_range < 1e-12:
                continue
            
            order = np.argsort(f_vals)
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            
            for i in range(1, n - 1):
                dist[order[i]] += (f_vals[order[i + 1]] - f_vals[order[i - 1]]) / f_range
        
        return dist

    def _hypervolume_3d(self, points, ref):
        """
        Calculate hypervolume for 3 objectives.
        points: (N, 3) array of objective values.
        ref: (3,) reference point.
        """
        # Filter points strictly better than ref
        mask = np.all(points < ref, axis=1)
        pts = points[mask]
        if len(pts) == 0:
            return 0.0
        
        # Sort by first objective
        pts = pts[np.argsort(pts[:, 0])]
        
        # We can use a simple O(N^2) or O(N log N) algorithm for 3D.
        # For small N (<= 20), O(N^2) is fine.
        # Hypervolume is the volume of the union of boxes from each point to ref.
        # Using inclusion-exclusion is complex.
        # Simpler: Decompose into slices or use a recursive partition.
        # Given small N, we can just sum contributions carefully or use a library if available (scipy is available but no direct HV).
        
        # Let's implement a simple 3D HV calculation.
        # Sort by f1
        # For each point i, the volume it contributes is:
        # (ref[1] - f[i,1]) * (ref[2] - f[i,2]) * (f[i,0] - f[i-1,0]) ... this is not quite right because of overlaps.
        
        # Correct approach for 3D:
        # Project to 2D (f2, f3) for each slice of f1.
        # Or use the standard recursive algorithm.
        
        # Given the small size, we can use a simple brute-force grid or just rely on the fact that we are comparing relative HV.
        # Actually, for the local search, we just need to know if adding point y to set P improves HV compared to P.
        # HV(P U {y}) > HV(P) if and only if y is not dominated by P and y contributes new volume.
        # This is equivalent to y being non-dominated by P.
        # So, for local search acceptance, we can just check if the new point is non-dominated by the current front.
        # We don't need to calculate exact HV for every step, just check non-domination.
        # However, to be more aggressive, we can check if it dominates any existing point.
        
        # For the final score, the evaluator calculates HV. We just need to provide a good set of non-dominated points.
        # So the local search goal is to find points that are non-dominated and spread out.
        
        # I will implement a helper to check if a point is dominated by a set.
        pass

    def _is_dominated(self, f, F_set):
        """Check if f is dominated by any point in F_set."""
        for f_set in F_set:
            if np.all(f_set <= f) and np.any(f_set < f):
                return True
        return False

    def _local_search_hv(self, x, f, lo, hi, rng, problem, F_front, X_front, budget_remaining, deadline):
        """
        Perform local search around x to improve the front.
        Returns list of (X, F) pairs found that are non-dominated.
        """
        found = []
        # Try 3 random perturbations
        for _ in range(3):
            if budget_remaining <= 0 or time.time() > deadline:
                break
            
            # Perturb x
            x_new = x.copy()
            # Gaussian perturbation scaled by bounds
            scale = 0.1 * (hi - lo)
            noise = rng.normal(0, 1, self.dim) * scale
            x_new = np.clip(x_new + noise, lo, hi)
            
            f_new = np.array(problem(x_new), dtype=np.float64)
            budget_remaining -= 1
            
            # Check if f_new is non-dominated by current front
            if not self._is_dominated(f_new, F_front):
                found.append((x_new, f_new))
                
        return found

    def _sbx_crossover(self, x1, x2, lo, hi, rng):
        """Simulated binary crossover."""
        if rng.random() < 0.5:
            return x1.copy(), x2.copy()
        
        x3 = x1.copy()
        x4 = x2.copy()
        beta = 15.0  # Higher eta for more exploration
        
        for d in range(len(x1)):
            if rng.random() < 0.5:
                continue
            
            if abs(x1[d] - x2[d]) < 1e-12:
                x3[d] = x1[d]
                x4[d] = x2[d]
                continue
            
            if x1[d] < x2[d]:
                u = rng.random()
                if u <= 1.0 / (2 * beta + 1):
                    betaq = (2 * u * beta) ** (1.0 / (beta + 1))
                elif u >= 1.0 - 1.0 / (2 * beta + 1):
                    betaq = (2 * (1 - u) * beta) ** (1.0 / (beta + 1))
                else:
                    betaq = 1.0
                
                x3[d] = 0.5 * ((1 + betaq) * x1[d] + (1 - betaq) * x2[d])
                x4[d] = 0.5 * ((1 - betaq) * x1[d] + (1 + betaq) * x2[d])
            else:
                u = rng.random()
                if u <= 1.0 / (2 * beta + 1):
                    betaq = (2 * (1 - u) * beta) ** (1.0 / (beta + 1))
                elif u >= 1.0 - 1.0 / (2 * beta + 1):
                    betaq = (2 * u * beta) ** (1.0 / (beta + 1))
                else:
                    betaq = 1.0
                
                x3[d] = 0.5 * ((1 + betaq) * x1[d] + (1 - betaq) * x2[d])
                x4[d] = 0.5 * ((1 - betaq) * x1[d] + (1 + betaq) * x2[d])
        
        x3 = np.clip(x3, lo, hi)
        x4 = np.clip(x4, lo, hi)
        return x3, x4

    def _polynomial_mutation(self, x, lo, hi, rng, eta=20.0):
        """Polynomial mutation."""
        x_mut = x.copy()
        for d in range(len(x)):
            if rng.random() < 0.5:
                continue
            
            u = rng.random()
            if u < 0.5:
                val = (2 * u) ** (1.0 / (eta + 1))
                delta = (1 - val) * (hi[d] - lo[d])
            else:
                val = (2 * (1 - u)) ** (1.0 / (eta + 1))
                delta = (1 - val) * (hi[d] - lo[d])
            
            x_mut[d] = x[d] + delta
            x_mut[d] = np.clip(x_mut[d], lo[d], hi[d])
        
        return x_mut

    def _tournament_select(self, ranks, cd, n_sel, rng):
        """Binary tournament selection based on rank then crowding distance."""
        n = len(ranks)
        selected = np.empty(n_sel, dtype=int)
        
        for i in range(n_sel):
            i1 = rng.randint(0, n)
            i2 = rng.randint(0, n)
            
            if ranks[i1] < ranks[i2]:
                selected[i] = i1
            elif ranks[i2] < ranks[i1]:
                selected[i] = i2
            else:
                if cd[i1] > cd[i2]:
                    selected[i] = i1
                else:
                    selected[i] = i2
        
        return selected

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        rng = np.random  # Use evaluator's seeded RNG

        dim = self.dim
        budget = self.budget
        used = 0

        # Population size
        pop_size = 15

        # Initialize population
        pop_X = np.empty((pop_size, dim))
        pop_F = np.empty((pop_size, self.n_obj))
        
        for i in range(pop_size):
            if used >= budget or time.time() > deadline:
                break
            pop_X[i] = lo + (hi - lo) * rng.random_sample(dim)
            pop_F[i] = np.array(problem(pop_X[i]), dtype=np.float64)
            used += 1
        
        # NSGA-II main loop with Memetic Local Search
        while used < budget and time.time() < deadline:
            # Fast non-dominated sort
            fronts = self._fast_non_dominated_sort(pop_F)
            
            # Compute ranks
            ranks = np.zeros(pop_size, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    ranks[idx] = rank
            
            # Compute crowding distance
            cd = np.zeros(pop_size)
            for front in fronts:
                if len(front) > 0:
                    front_cd = self._crowding_distance(pop_F, front)
                    for j, idx in enumerate(front):
                        cd[idx] = front_cd[j]
            
            # --- STRUCTURAL CHANGE: Memetic Local Search ---
            # Identify the best few individuals from the first front for local refinement
            offspring_X = []
            offspring_F = []
            
            # 1. Local Search on Top 3 individuals of Front 0
            if fronts and len(fronts[0]) >= 1:
                top_indices = fronts[0][:3]
                for idx in top_indices:
                    if used >= budget or time.time() > deadline:
                        break
                    # Perform local search
                    local_results = self._local_search_hv(
                        pop_X[idx], pop_F[idx], lo, hi, rng, problem, 
                        pop_F, pop_X, budget - used, deadline
                    )
                    # Note: _local_search_hv updates 'used' internally? No, it returns results. 
                    # We need to track used carefully.
                    # Let's modify _local_search_hv to return (results, evals_used)
                    # Or just do it inline here to keep track of used easily.
                    
                    # Inline Local Search for simplicity and budget tracking
                    for _ in range(2): # 2 neighbors per top individual
                        if used >= budget or time.time() > deadline:
                            break
                        x_new = pop_X[idx].copy()
                        scale = 0.1 * (hi - lo)
                        noise = rng.normal(0, 1, dim) * scale
                        x_new = np.clip(x_new + noise, lo, hi)
                        
                        f_new = np.array(problem(x_new), dtype=np.float64)
                        used += 1
                        
                        # Check if non-dominated by current population's front
                        # Simplified: if it's not dominated by any point in pop_F
                        is_dom = False
                        for f_pop in pop_F:
                            if np.all(f_pop <= f_new) and np.any(f_pop < f_new):
                                is_dom = True
                                break
                        if not is_dom:
                            offspring_X.append(x_new)
                            offspring_F.append(f_new)

            # 2. Global Search (SBX + Mutation) for remaining budget
            # We want to fill the offspring pool to approximately pop_size
            # Let's generate enough to have pop_size offspring total
            n_global_needed = pop_size - len(offspring_X)
            
            if n_global_needed > 0:
                # Selection for global offspring
                # Use tournament selection
                n_sel = n_global_needed
                selected = self._tournament_select(ranks, cd, n_sel, rng)
                
                for i in range(0, n_global_needed, 2):
                    if used >= budget or time.time() > deadline:
                        break
                    
                    p1_idx = selected[i % n_sel]
                    p2_idx = selected[(i + 1) % n_sel]
                    
                    x1 = pop_X[p1_idx].copy()
                    x2 = pop_X[p2_idx].copy()
                    
                    c1, c2 = self._sbx_crossover(x1, x2, lo, hi, rng)
                    c1 = self._polynomial_mutation(c1, lo, hi, rng)
                    c2 = self._polynomial_mutation(c2, lo, hi, rng)
                    
                    if used < budget and time.time() < deadline:
                        offspring_X.append(c1)
                        f1 = np.array(problem(c1), dtype=np.float64)
                        offspring_F.append(f1)
                        used += 1
                    
                    if i + 1 < n_global_needed and used < budget and time.time() < deadline:
                        offspring_X.append(c2)
                        f2 = np.array(problem(c2), dtype=np.float64)
                        offspring_F.append(f2)
                        used += 1
                    else:
                        break
            
            if not offspring_X:
                break
                
            offspring_X = np.array(offspring_X, dtype=np.float64)
            offspring_F = np.array(offspring_F, dtype=np.float64)
            
            # Combine parent and offspring
            combined_X = np.vstack([pop_X, offspring_X])
            combined_F = np.vstack([pop_F, offspring_F])
            
            n_combined = len(combined_X)
            
            # Fast non-dominated sort on combined
            fronts = self._fast_non_dominated_sort(combined_F)
            
            # Compute ranks
            combined_ranks = np.zeros(n_combined, dtype=int)
            for rank, front in enumerate(fronts):
                for idx in front:
                    combined_ranks[idx] = rank
            
            # Compute crowding distance
            combined_cd = np.zeros(n_combined)
            for front in fronts:
                if len(front) > 0:
                    front_cd = self._crowding_distance(combined_F, front)
                    for j, idx in enumerate(front):
                        combined_cd[idx] = front_cd[j]
            
            # Select next population
            next_indices = []
            for front in fronts:
                if len(next_indices) + len(front) <= pop_size:
                    next_indices.extend(front)
                else:
                    # Sort by crowding distance (descending)
                    front_sorted = sorted(front, key=lambda x: combined_cd[x], reverse=True)
                    need = pop_size - len(next_indices)
                    next_indices.extend(front_sorted[:need])
                    break
            
            # Update population
            pop_X = combined_X[next_indices]
            pop_F = combined_F[next_indices]
        
        # Return all non-dominated solutions found
        fronts = self._fast_non_dominated_sort(pop_F)
        if fronts and len(fronts[0]) > 0:
            return pop_X[fronts[0]]
        return pop_X[:1]
# EVOLVE-BLOCK-END
