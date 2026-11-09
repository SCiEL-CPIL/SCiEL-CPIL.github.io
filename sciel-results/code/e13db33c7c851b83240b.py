"""Multi-objective optimizer on RE37 with structured search and local refinement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
import math


class Optimizer:
    """Multi-objective optimizer using LHS + adaptive local search with deadline guard."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _fast_non_dominated_sort(self, F):
        """Return list of fronts (list of indices)."""
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]
        nd = np.zeros(n, dtype=int)
        
        for p in range(n):
            for q in range(n):
                if p == q:
                    continue
                if np.all(F[p] <= F[q]) and np.any(F[p] < F[q]):
                    S[p].append(q)
                elif np.all(F[q] <= F[p]) and np.any(F[q] < F[p]):
                    nd[p] += 1
        
        front = [i for i in range(n) if nd[i] == 0]
        if not front:
            front = [0]
        
        fronts = [front]
        i = 0
        while front:
            next_front = []
            for p in front:
                for q in S[p]:
                    nd[q] -= 1
                    if nd[q] == 0:
                        next_front.append(q)
            if next_front:
                fronts.append(next_front)
            front = next_front
            i += 1
        return fronts

    def _crowding_distance(self, F, indices):
        """Calculate crowding distance for a front."""
        n = len(indices)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        F_sub = F[indices]
        
        for m in range(self.n_obj):
            order = np.argsort(F_sub[:, m])
            dist[order[0]] = np.inf
            dist[order[-1]] = np.inf
            
            min_val = F_sub[order[0], m]
            max_val = F_sub[order[-1], m]
            if max_val == min_val:
                continue
                
            for k in range(1, n - 1):
                dist[order[k]] += (F_sub[order[k + 1], m] - F_sub[order[k - 1], m]) / (max_val - min_val)
        return dist

    def _nelder_mead_minimize(self, problem, bounds, x0, max_iter=50, tol=1e-8):
        """
        Minimize a single scalar objective (sum of objectives) using Nelder-Mead.
        Returns the best x found.
        """
        lo = np.array([b[0] for b in bounds])

    def _nelder_mead_minimize(self, problem, bounds, x0, max_iter=50, tol=1e-8):
        """
        Minimize a single scalar objective (sum of objectives) using Nelder-Mead.
        Returns the best x found.
        """
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        def objective(x):
            x_clipped = np.clip(x, lo, hi)
            f = problem(x_clipped)
            return np.sum(f)
        
        # Initialize simplex
        n = self.dim
        simplex = [x0.copy()]
        for i in range(n):
            x_trial = x0.copy()
            x_trial[i] += 0.05 * (hi[i] - lo[i])
            x_trial = np.clip(x_trial, lo, hi)
            simplex.append(x_trial)
        
        f_vals = [objective(x) for x in simplex]
        
        alpha = 1.0
        gamma = 2.0
        rho = 0.5
        sigma = 0.5
        
        for _ in range(max_iter):
            # Sort simplex by objective value
            order = np.argsort(f_vals)
            simplex = [simplex[i] for i in order]
            f_vals = [f_vals[i] for i in order]
            
            # Check convergence
            if np.max(f_vals) - np.min(f_vals) < tol:
                break
            
            # Centroid of all but worst
            centroid = np.mean(simplex[:-1], axis=0)
            
            # Reflection
            x_r = centroid + alpha * (centroid - simplex[-1])
            x_r = np.clip(x_r, lo, hi)
            f_r = objective(x_r)
            
            if f_r < f_vals[0]:
                # Expansion
                x_e = centroid + gamma * (x_r - centroid)
                x_e = np.clip(x_e, lo, hi)
                f_e = objective(x_e)
                if f_e < f_r:
                    simplex[-1] = x_e
                    f_vals[-1] = f_e
                else:
                    simplex[-1] = x_r
                    f_vals[-1] = f_r
            elif f_r < f_vals[-2]:
                simplex[-1] = x_r
                f_vals[-1] = f_r
            else:
                # Contraction
                if f_r < f_vals[-1]:
                    # Outside contraction
                    x_c = centroid + 0.5 * (x_r - centroid)
                    x_c = np.clip(x_c, lo, hi)
                    f_c = objective(x_c)
                    if f_c <= f_r:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_vals[i] = objective(simplex[i])
                else:
                    # Inside contraction
                    x_c = centroid - 0.5 * (centroid - simplex[-1])
                    x_c = np.clip(x_c, lo, hi)
                    f_c = objective(x_c)
                    if f_c <= f_vals[-1]:
                        simplex[-1] = x_c
                        f_vals[-1] = f_c
                    else:
                        # Shrink
                        for i in range(1, len(simplex)):
                            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                            simplex[i] = np.clip(simplex[i], lo, hi)
                            f_vals[i] = objective(simplex[i])
        
        # Return best point
        best_idx = np.argmin(f_vals)
        return simplex[best_idx]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        
        # Phase 1: Latin Hypercube Sampling
        # LHS ensures good coverage of the 4D space.
        n_lhs = 100
        indices = np.arange(n_lhs)
        permuted_indices = []
        for d in range(self.dim):
            perm = np.random.permutation(n_lhs)
            permuted_indices.append(perm)
        
        X_lhs = np.zeros((n_lhs, self.dim))
        for d in range(self.dim):
            X_lhs[:, d] = lo[d] + (permuted_indices[d] + 0.5) * (hi[d] - lo[d]) / n_lhs
            
        F_lhs = np.array([problem(x) for x in X_lhs])
        used = n_lhs
        
        # Check deadline
        if time.time() > deadline:
            return F_lhs
        
        # Phase 2: Local refinement using Nelder-Mead on scalarized objectives
        # We will pick a diverse set of points from the Pareto front and refine them.
        
        # Get non-dominated points
        fronts = self._fast_non_dominated_sort(F_lhs)
        
        if fronts:
            front_indices = fronts[0]
        else:
            front_indices = np.arange(n_lhs)
        
        # Select a subset of points to refine
        # We want diversity, so use crowding distance
        if len(front_indices) > 20:
            dist = self._crowding_distance(F_lhs, front_indices)
            order = np.argsort(dist)[::-1]
            front_indices = [front_indices[i] for i in order[:20]]
        
        # Refine these points
        # We have 100 evaluations left.
        # Nelder-Mead can use many evaluations. Let's limit it.
        # For 4 dimensions, NM uses 5 points per iteration.
        # Let's allow 10 iterations per point -> 50 evals per point.
        # That's too many for 20 points.
        # Let's refine only the top 5 diverse points with more iterations.
        
        if len(front_indices) > 5:
            # Re-select top 5 by crowding distance
            dist = self._crowding_distance(F_lhs, front_indices)
            order = np.argsort(dist)[::-1]
            front_indices = [front_indices[i] for i in order[:5]]
        
        new_X = []
        new_F = []
        
        remaining_budget = self.budget - used
        points_to_refine = len(front_indices)
        if points_to_refine == 0:
            return F_lhs
        
        # Estimate budget per point
        # NM with 4 dims: simplex size 5.
        # Let's say we allow 20 iterations per point.
        # That's ~100 evals per point (rough estimate, depends on contraction/expansion).
        # We have 100 evals total.
        # So we can afford to refine 1-2 points deeply, or 5 points shallowly.
        # Let's do 5 points with ~20 evals each.
        
        max_iter_per_point = max(5, remaining_budget // (points_to_refine * 5))
        
        for idx in front_indices:
            if time.time() > deadline - 10:
                break
            
            x_seed = X_lhs[idx]
            
            # Run Nelder-Mead to minimize sum of objectives
            # This is a heuristic to find a good point.
            # We need to track evaluations inside NM.
            # For simplicity, we'll just call it and hope it stays within budget.
            # To be safe, we'll reduce max_iter if we are running low on time.
            
            # Actually, let's implement a simpler local search:
            # Coordinate descent with random restarts.
            
            # Let's try a few random perturbations with decreasing step size.
            step_sizes = [0.2, 0.1, 0.05, 0.02, 0.01]
            
            for step in step_sizes:
                if used >= self.budget or time.time() > deadline - 5:
                    break
                
                # Generate 3 perturbations
                for _ in range(3):
                    if used >= self.budget or time.time() > deadline - 5:
                        break
                    
                    x_new = x_seed + step * span * np.random.standard_normal(self.dim)
                    x_new = np.clip(x_new, lo, hi)
                    
                    f_new = problem(x_new)
                    new_X.append(x_new)
                    new_F.append(f_new)
                    used += 1
                    
                    # If this is better (in some sense), update seed?
                    # For simplicity, we don't track best, we just record.
        
        # Combine all results
        if new_F:
            F_combined = np.vstack([F_lhs, np.array(new_F)])
        else:
            F_combined = F_lhs
        
        return F_combined
# EVOLVE-BLOCK-END
