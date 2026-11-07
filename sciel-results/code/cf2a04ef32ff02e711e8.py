import numpy as np
import time
from scipy.integrate import simpson  # Not needed for WFG7, but scipy is available. We'll use pure numpy/scipy.optimize.
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _sobol_sequence(self, n, dim, seed=0):
        """Generate a Sobol sequence using a simple implementation or fallback to Halton if scipy.stats.qmc not desired/available in older versions, but scipy is allowed.
        However, to avoid dependency issues with specific scipy versions, I'll implement a simple Halton sequence which is also space-filling and deterministic.
        """
        # Halton sequence is simpler to implement from scratch without external dependencies beyond numpy.
        def halton(index, base):
            f = 1.0
            r = 0.0
            while index > 0:
                f /= base
                r += f * (index % base)
                index //= base
            return r

        points = np.zeros((n, dim))
        for i in range(1, n + 1):
            for d in range(dim):
                base = d + 2  # 2, 3, 5, ...
                points[i - 1, d] = halton(i, base)
        return points

    def _non_dominated_sort(self, F):
        """
        Return list of fronts (indices).
        F: (n, m) array of objective values.
        """
        n = F.shape[0]
        if n == 0:
            return []
        
        domination_count = np.zeros(n, dtype=int)
        dominated_indices = [[] for _ in range(n)]
        fronts = []
        
        # Precompute domination relationships
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                # Check if i dominates j: F[i] <= F[j] and F[i] != F[j]
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    domination_count[j] += 1
                    dominated_indices[i].append(j)
        
        # First front
        front = [i for i in range(n) if domination_count[i] == 0]
        if front:
            fronts.append(front)
        
        current_front = front
        while current_front:
            next_front = []
            for i in current_front:
                for j in dominated_indices[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            current_front = next_front
            
        return fronts

    def _crowding_distance(self, F, indices):
        """
        Calculate crowding distance for points with given indices.
        Returns array of distances corresponding to indices.
        """
        if not indices:
            return np.array([])
        
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        distances = np.zeros(n)
        F_subset = F[indices]
        
        for m in range(self.n_obj):
            # Sort indices by objective m
            sorted_idx = np.argsort(F_subset[:, m])
            distances[sorted_idx[0]] = np.inf
            distances[sorted_idx[-1]] = np.inf
            
            f_min = F_subset[sorted_idx[0], m]
            f_max = F_subset[sorted_idx[-1], m]
            
            if f_max == f_min:
                continue
                
            for i in range(1, n - 1):
                prev_val = F_subset[sorted_idx[i - 1], m]
                next_val = F_subset[sorted_idx[i + 1], m]
                distances[sorted_idx[i]] += (next_val - prev_val) / (f_max - f_min)
                
        return distances

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320.0  # Safety margin
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([3.0, 5.0, 7.0])
        
        # 1. Generate initial population using Halton sequence (space-filling)
        # Use first 50% of budget for initial exploration
        n_initial = self.budget // 2
        if n_initial < 10:
            n_initial = 10
        if n_initial > self.budget - 10: # Keep some for refinement
            n_initial = self.budget - 10
            
        # Halton sequence points in [0,1]^d
        halton_pts = self._sobol_sequence(n_initial, self.dim)
        
        # Map to bounds
        x_initial = lo + halton_pts * (hi - lo)
        
        archive_X = []
        archive_F = []
        used = 0
        
        # Evaluate initial points
        for x in x_initial:
            if used >= self.budget:
                break
            f = np.array(problem(x))
            archive_X.append(x)
            archive_F.append(f)
            used += 1
            
        if not archive_F:
            return []
            
        archive_F = np.array(archive_F)
        
        # 2. Identify non-dominated points
        fronts = self._non_dominated_sort(archive_F)
        if not fronts:
            # If all are dominated (shouldn't happen with Halton usually, but just in case)
            # Pick the best one based on sum of distances to ref
            dists = np.sum(archive_F - ref, axis=1)
            best_idx = np.argmin(dists)
            return [archive_X[best_idx]]
            
        # The first front contains the non-dominated points
        nd_indices = fronts[0]
        
        # 3. Local Refinement Phase
        # Use remaining budget to refine the most promising non-dominated points.
        # We will select a subset of non-dominated points to refine.
        # To ensure spread, we pick points based on crowding distance.
        
        remaining_budget = self.budget - used
        
        if remaining_budget <= 0:
            return archive_X
            
        # Calculate crowding distance for the first front
        if len(nd_indices) > 1:
            cd = self._crowding_distance(archive_F, nd_indices)
            # Sort by crowding distance descending (most isolated first)
            # But we also want to ensure we refine points that are "good" (close to origin ideally, but non-dominated implies trade-off)
            # Let's just refine the top K points with highest crowding distance to maintain spread.
            order = np.argsort(cd)[::-1]
            # We have remaining_budget evaluations.
            # Each Nelder-Mead run might take ~10-20 evaluations.
            # Let's limit the number of refinement runs to avoid timeout.
            # Assume ~15 evals per run.
            max_runs = remaining_budget // 15
            if max_runs < 1:
                max_runs = 1
                
            # Pick top max_runs indices from the non-dominated set
            selected_indices = [nd_indices[i] for i in order[:max_runs]]
        else:
            selected_indices = nd_indices
            
        # Perform local optimization on selected points
        for idx in selected_indices:
            if used >= self.budget:
                break
            if time.time() > deadline:
                break
                
            x_start = archive_X[idx]
            
            # Define a scalarization function to guide the local search.
            # We want to minimize the distance to the "utopia" point (min of objectives found so far) or just use a weighted sum.
            # A common approach for multi-objective local search is to minimize the Tchebycheff metric or a weighted sum.
            # Let's use a weighted sum with weights derived from the current position relative to the utopia point.
            
            utopia = np.min(archive_F, axis=0)
            
            # Weights: inverse distance to utopia? Or just uniform?
            # Let's try a simple approach: minimize the sum of normalized objectives.
            # Or better, use the scalarization: min_x max_i (f_i(x) - z_i^*) / (z_i^ref - z_i^*)
            # This is the Chebyshev scalarization.
            
            def scalarize(x):
                x_clipped = np.clip(x, lo, hi)
                f = problem(x_clipped)
                # Normalize
                norm_f = (f - utopia) / (ref - utopia + 1e-10)
                return np.max(norm_f)
                
            # Nelder-Mead requires a scalar objective.
            # We need to wrap it to handle bounds and counting.
            
            nm_evals = [0]
            def nm_obj(x):
                nm_evals[0] += 1
                if used + nm_evals[0] > self.budget:
                    # Return a large value to stop
                    return 1e10
                return scalarize(x)
                
            # Options for Nelder-Mead
            options = {
                'maxiter': 20, # Limit iterations to control evals
                'maxfev': remaining_budget - used,
                'xatol': 1e-4,
                'fatol': 1e-6
            }
            
            try:
                res = minimize(nm_obj, x_start, method='Nelder-Mead', options=options)
                x_opt = res.x
                f_opt = np.array(problem(x_opt))
                
                # Add to archive
                archive_X.append(x_opt)
                archive_F.append(f_opt)
                used += nm_evals[0]
                
            except Exception:
                pass
                
        return archive_X
