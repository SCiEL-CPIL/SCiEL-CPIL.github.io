"""Optimized program for WFG9 using MOEA/D with local search refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Multi-Objective Optimizer combining MOEA/D with local refinement."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        range_ = hi - lo
        
        # --- Phase 1: MOEA/D ---
        # Determine number of subproblems based on budget
        # We want enough evaluations for some generations.
        # Let's aim for ~10-15 generations.
        # If budget=200, and pop_size=20, we have 10 generations.
        # If pop_size=30, we have ~6-7 generations.
        # Let's use a grid of H=5 for 3 objectives -> (5+1)*(5+2)/2 = 21 points.
        H = 5
        ref_points = self._generate_grid_points(H, self.n_obj)
        n_sub = len(ref_points)
        
        # Build neighborhood
        dist = np.zeros((n_sub, n_sub))
        for i in range(n_sub):
            for j in range(n_sub):
                dist[i, j] = np.linalg.norm(ref_points[i] - ref_points[j])
        sorted_indices = np.argsort(dist, axis=1)
        T = min(10, n_sub) # Neighborhood size
        neighbors = [sorted_indices[i, :T] for i in range(n_sub)]
        
        # Initialize population
        X = np.zeros((n_sub, self.dim))
        F = np.zeros((n_sub, self.n_obj))
        
        # Use a local RNG for reproducibility within the call if needed, 
        # but the task says np.random.seed is set externally. 
        # We will use np.random directly to respect the seed.
        
        for i in range(n_sub):
            X[i] = lo + np.random.rand(self.dim) * range_
            F[i] = problem(X[i])
        
        evals = n_sub
        
        # Ideal point (minimums)
        z = np.min(F, axis=0)
        
        # DE parameters
        F_DE = 0.5
        CR = 0.5
        
        # Main MOEA/D loop
        # Reserve budget for local search
        # Let's use 70% of budget for MOEA/D and 30% for local search
        # Or simply run until evals > budget - reserved
        reserved_for_local = 40
        moea_budget_limit = self.budget - reserved_for_local
        
        while evals < moea_budget_limit:
            if time.time() > deadline - 5:
                break
            
            # Select subproblem to evolve
            # Random selection is standard
            idx = np.random.randint(n_sub)
            
            # Select parents from neighborhood
            # Pick 2 distinct parents from neighbors[idx]
            p1_idx, p2_idx = np.random.choice(neighbors[idx], 2, replace=False)
            
            # Ensure p1 and p2 are distinct from each other and preferably from idx?
            # Standard DE/rand/1 uses 3 parents. Let's pick a 3rd.
            candidates = [j for j in neighbors[idx] if j != p1_idx and j != p2_idx]
            if len(candidates) > 0:
                p3_idx = np.random.choice(candidates)
            else:
                # Fallback if neighborhood is too small or all picked
                p3_idx = p1_idx
            
            p1 = X[p1_idx]
            p2 = X[p2_idx]
            p3 = X[p3_idx]
            current = X[idx]
            
            # DE/rand/1 mutation
            v = p1 + F_DE * (p2 - p3)
            
            # Binomial Crossover
            child = np.zeros_like(v)
            j_rand = np.random.randint(self.dim)
            for i in range(self.dim):
                if np.random.rand() < CR or i == j_rand:
                    child[i] = v[i]
                else:
                    child[i] = current[i]
            
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            evals += 1
            
            # Update Ideal Point
            z = np.minimum(z, fc)
            
            # Update using Tchebycheff scalarization
            w = ref_points[idx]
            
            curr_fit = self._tchebycheff(F[idx], w, z)
            child_fit = self._tchebycheff(fc, w, z)
            
            # Accept if child is better or equal (with slight probability for diversity?)
            # Standard MOEA/D accepts if better.
            if child_fit <= curr_fit:
                X[idx] = child
                F[idx] = fc
            
            # Optional: Update neighbors? 
            # For small budgets, static neighborhoods are often sufficient and faster.
            # Dynamic neighborhood update can help but adds overhead.
            # Given budget 200, static is safer for speed.
        
        # --- Phase 2: Local Search Refinement ---
        # Identify candidates for local search.
        # We want to refine points that are on the Pareto front or close to it.
        # Compute non-dominated set of the current population.
        
        non_dominated_idx = self._find_non_dominated_indices(F)
        
        # If we have many non-dominated points, pick a diverse subset.
        # If few, pick the best ones.
        # Let's take up to 5 points for local search to stay within budget.
        # Prioritize points with high crowding distance or extreme values.
        
        if len(non_dominated_idx) > 0:
            # Calculate crowding distance for the non-dominated points to pick diverse ones
            F_nd = F[non_dominated_idx]
            cd = self._crowding_distance(F_nd)
            
            # Sort by crowding distance descending
            sorted_nd_idx = np.argsort(-cd)
            
            # Take top 3-5 diverse points
            n_local = min(5, len(non_dominated_idx))
            local_candidates = non_dominated_idx[sorted_nd_idx[:n_local]]
            
            # Also, always include the point with the minimum sum of objectives (best compromise)
            sums = F.sum(axis=1)
            best_sum_idx = np.argmin(sums)
            if best_sum_idx not in local_candidates:
                # Replace the worst candidate with the best sum if necessary
                if len(local_candidates) > 0:
                    local_candidates[-1] = best_sum_idx
                else:
                    local_candidates = [best_sum_idx]
            
            # Perform local search on these candidates
            local_step = 0.05 * np.max(range_) # Initial step size relative to range
            
            for idx in local_candidates:
                if evals >= self.budget or time.time() > deadline - 2:
                    break
                
                best_x = X[idx].copy()
                best_f = F[idx].copy()
                best_sum = best_f.sum()
                
                # Simple coordinate descent with adaptive step
                # Try to improve the sum of objectives, which often correlates with hypervolume contribution
                # if the point is on the front.
                
                for _iter in range(3): # Limit iterations per point to save time
                    if evals >= self.budget or time.time() > deadline - 2:
                        break
                    
                    improved = False
                    for coord in range(self.dim):
                        if evals >= self.budget or time.time() > deadline - 2:
                            break
                        
                        # Try positive and negative directions
                        for direction in [-1, 1]:
                            trial_x = best_x.copy()
                            trial_x[coord] = np.clip(trial_x[coord] + direction * local_step, lo[coord], hi[coord])
                            
                            # Check if it's the same point (due to clipping)
                            if np.array_equal(trial_x, best_x):
                                continue
                            
                            trial_f = problem(trial_x)
                            evals += 1
                            
                            # Accept if it improves the sum (greedy local search)
                            # In multi-objective, this might move off the front, but 
                            # since we started from a non-dominated point, small steps 
                            # often stay near the front or find better trade-offs.
                            if trial_f.sum() < best_sum:
                                best_x = trial_x
                                best_f = trial_f
                                best_sum = best_f.sum()
                                improved = True
                                # Reduce step size if improved
                                local_step *= 0.5
                                break # Move to next coordinate after improvement
                            
                            # If not improved, we could try reducing step, but for speed, 
                            # we just move to next direction/coord in this simple version.
                    
                    if not improved:
                        # If no improvement in a full pass, reduce step and try again?
                        # Or break? Let's reduce step slightly and try one more pass if budget allows.
                        local_step *= 0.5
                        if local_step < 1e-4:
                            break
                    else:
                        # If improved, keep current step or reduce? 
                        # We reduced it upon improvement. Let's keep it for next coord.
                        pass
            
            # Update the population with the refined point
            X[idx] = best_x
            F[idx] = best_f
        
        return X

    def _generate_grid_points(self, H, n_obj):
        """Generate reference points on a simplex grid."""
        points = []
        if n_obj == 3:
            for i in range(H + 1):
                for j in range(H + 1 - i):
                    k = H - i - j
                    points.append([i / H, j / H, k / H])
        elif n_obj == 2:
            for i in range(H + 1):
                points.append([i / H, 1 - i / H])
        else:
            # Fallback for other dimensions (not expected for WFG9)
            for i in range(H + 1):
                points.append([1.0] * n_obj)
        return np.array(points)

    def _tchebycheff(self, f, w, z):
        """Calculate Tchebycheff fitness."""
        return np.max(w * np.abs(f - z))

    def _find_non_dominated_indices(self, F):
        """Find indices of non-dominated points."""
        n = len(F)
        is_dominated = np.zeros(n, dtype=bool)
        
        for i in range(n):
            if is_dominated[i]:
                continue
            for j in range(n):
                if i == j or is_dominated[j]:
                    continue
                if self._dominates(F[i], F[j]):
                    is_dominated[j] = True
        
        return np.where(~is_dominated)[0]

    def _dominates(self, a, b):
        """Check if a dominates b."""
        return np.all(a <= b) and np.any(a < b)

    def _crowding_distance(self, F):
        """Calculate crowding distance for a set of non-dominated points."""
        n = len(F)
        cd = np.zeros(n)
        
        if n <= 2:
            cd[:] = float('inf')
            return cd
        
        for m in range(self.n_obj):
            sorted_idx = np.argsort(F[:, m])
            cd[sorted_idx[0]] = float('inf')
            cd[sorted_idx[-1]] = float('inf')
            
            f_min = F[sorted_idx[0], m]
            f_max = F[sorted_idx[-1], m]
            
            if f_max == f_min:
                continue
            
            for i in range(1, n - 1):
                cd[sorted_idx[i]] += (F[sorted_idx[i + 1], m] - F[sorted_idx[i - 1], m]) / (f_max - f_min)
        
        return cd
# EVOLVE-BLOCK-END
