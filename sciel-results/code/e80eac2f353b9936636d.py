# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for RE34 using adaptive mutation with non-dominated sorting."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        
        # Vectorized: for each pair (i,j), check if i dominates j
        # F[i] dominates F[j] if F[i] <= F[j] in all obj AND F[i] < F[j] in at least one
        # Shape: (n, n, n_obj)
        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)
        
        leq = np.all(F_i <= F_j, axis=2)  # (n, n)
        lt = np.any(F_i < F_j, axis=2)   # (n, n)
        
        # dominates[i,j] = True if i dominates j
        dominates = leq & lt
        # A point is dominated if any other point dominates it
        # dominated[j] = any(i != j such that dominates[i,j])
        dominated = np.any(dominates, axis=0)
        
        return ~dominated

    def _find_worst_index(self, F, ref_point):
        """Find the index of the worst point based on hypervolume contribution (approx)."""
        if len(F) == 0:
            return 0
        
        # Use a hypervolume-like contribution estimate:
        # For each point, compute the volume of the box it "owns" relative to ref_point
        # A point is worse if it's far from the front (large objectives)
        # Also penalize dominated points more heavily
        
        # First, check which points are dominated
        nd_mask = self._non_dominated_mask(F)
        
        # For non-dominated points, use distance from ideal point
        # For dominated points, give them highest penalty
        F_min = F.min(axis=0)
        F_max = F.max(axis=0)
        F_range = F_max - F_min
        F_range[F_range == 0] = 1.0
        F_norm = (F - F_min) / F_range
        
        # Distance from origin in normalized space (lower is better)
        dist = np.sqrt(np.sum(F_norm ** 2, axis=1))
        
        # Dominated points get a large penalty
        penalty = np.where(~nd_mask, 100.0, 0.0)
        
        quality = dist + penalty
        return np.argmax(quality)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 270  # 270s safety margin before 300s limit
        
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_obj = self.n_obj
        dim = self.dim
        budget = self.budget
        
        # Reference point for RE34
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Initial population: mix of random and boundary points
        pop_size = min(50, budget // 3)
        X = lo + (hi - lo) * rng.random((pop_size, dim))
        
        # Add some boundary points for diversity
        n_boundary = min(15, pop_size // 4)
        for i in range(n_boundary):
            # Randomly choose which dimensions to set to boundary
            n_dims_to_boundary = rng.integers(1, dim + 1)
            dims_to_boundary = rng.choice(dim, size=n_dims_to_boundary, replace=False)
            for d in dims_to_boundary:
                if rng.random() < 0.5:
                    X[i, d] = lo[d]
                else:
                    X[i, d] = hi[d]
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Track all evaluated points for hypervolume
        all_F = F.copy()
        
        # Adaptive parameters
        base_mut_scale = 0.4
        decay_rate = 0.993
        
        # Phase 1: Evolutionary search
        search_budget = int(budget * 0.75)
        polish_budget = budget - search_budget
        
        while used < search_budget:
            # Check time
            if time.time() > deadline - 60:  # Reserve 60s for polishing
                break
            
            # Find non-dominated points
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            
            if len(nd_idx) == 0:
                # Fallback: use best single objective points
                nd_idx = np.array([np.argmin(F[:, 0]), np.argmin(F[:, 1]), np.argmin(F[:, 2])])
                nd_idx = np.unique(nd_idx)
            
            # Adaptive mutation scale
            progress = used / search_budget
            mut_scale = base_mut_scale * (decay_rate ** used)
            mut_scale = max(mut_scale, 0.03)
            
            # Generate children
            n_children = min(20, search_budget - used)
            for _ in range(n_children):
                if time.time() > deadline - 60:
                    break
                if used >= search_budget:
                    break
                
                # Choose parent from non-dominated set with some exploration
                if rng.random() < 0.75:
                    parent_idx = nd_idx[rng.integers(len(nd_idx))]
                else:
                    parent_idx = rng.integers(len(F))
                
                parent = X[parent_idx].copy()
                
                # Gaussian mutation with occasional larger jumps
                if rng.random() < 0.08:
                    jump_scale = mut_scale * 4.0
                else:
                    jump_scale = mut_scale
                
                child = parent + jump_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                
                # Update all_F
                all_F = np.vstack([all_F, fc])
                
                # Decide whether to replace a point in the population
                worst_idx = self._find_worst_index(F, ref_point)
                
                # Check if child is better than worst in at least one objective
                # and not worse in all
                child_better = np.any(fc < F[worst_idx])
                child_not_worse_all = not np.all(fc > F[worst_idx])
                
                if child_better and child_not_worse_all:
                    X[worst_idx] = child
                    F[worst_idx] = fc
        
        # Phase 2: Local refinement using scipy minimize on scalarized objectives
        if used < budget and time.time() < deadline - 30:
            # Get non-dominated points from current population
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            
            if len(nd_idx) > 0:
                # Try to refine each non-dominated point with a local optimizer
                # Use weighted sum scalarization with different weight vectors
                # For each ND point, try a few weight vectors and local optimization
                from scipy.optimize import minimize
                
                # Generate weight vectors on the simplex
                n_weights = min(10, polish_budget // 3)
                # Use random weight vectors
                W = rng.random((n_weights, n_obj))
                W /= W.sum(axis=1, keepdims=True)
                
                # For each non-dominated point, try local refinement
                points_to_refine = nd_idx[:min(5, len(nd_idx))]  # Limit to avoid timeout
                for pt_idx in points_to_refine:
                    if used >= budget or time.time() > deadline - 30:
                        break
                    
                    x0 = X[pt_idx].copy()
                    
                    # Try a few weight vectors
                    for w in W[:3]:
                        if used >= budget or time.time() > deadline - 30:
                            break
                        
                        # Scalarized objective: weighted sum
                        def obj(x):
                            return np.dot(w, problem(x))
                        
                        try:
                            res = minimize(obj, x0, method='L-BFGS-B', 
                                          bounds=list(zip(lo, hi)),
                                          options={'maxiter': 20, 'ftol': 1e-8})
                            if res.success:
                                fx = problem(res.x)
                                used += 1
                                all_F = np.vstack([all_F, fx])
                                
                                # Replace if improved
                                if np.any(fx < F[pt_idx]):
                                    X[pt_idx] = res.x
                                    F[pt_idx] = fx
                        except:
                            pass
        
        return X
# EVOLVE-BLOCK-END
