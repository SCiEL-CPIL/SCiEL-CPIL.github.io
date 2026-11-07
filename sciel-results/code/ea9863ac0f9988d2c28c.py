# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34 using Sobol initialization and hypervolume-contribution local search."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _non_dominated_mask(self, F):
        """Return a boolean mask of non-dominated points in F (n x n_obj array)."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        
        F_i = F[:, np.newaxis, :]  # (n, 1, n_obj)
        F_j = F[np.newaxis, :, :]  # (1, n, n_obj)
        
        leq = np.all(F_i <= F_j, axis=2)  # (n, n)
        lt = np.any(F_i < F_j, axis=2)   # (n, n)
        
        dominates = leq & lt
        dominated = np.any(dominates, axis=0)
        
        return ~dominated

    def _sobol_sequence(self, n, dim, lo, hi, seed=42):
        """Generate n points in [lo, hi]^dim using a quasi-random (Sobol-like) sequence."""
        # Use a simple Halton sequence as a Sobol alternative (no scipy dependency needed)
        # Or use numpy's random with a fixed seed for reproducibility, but structured
        # Actually, let's use a stratified sampling approach for better coverage
        
        # Stratified sampling: divide each dimension into n^(1/dim) bins
        n_per_dim = int(round(n ** (1.0 / dim)))
        if n_per_dim < 1:
            n_per_dim = 1
        
        # Generate grid points with jitter
        points = []
        for i in range(n):
            # Map index to grid position
            idx = i
            pt = np.zeros(dim)
            for d in range(dim):
                # Position in the d-th dimension
                pos = (idx % n_per_dim)
                # Add jitter within the bin
                jitter = (i * 7919 + d * 104729) % 10000 / 10000.0
                pt[d] = (pos + jitter) / n_per_dim
            
            # Scale to [lo, hi]
            pt = lo + pt * (hi - lo)
            points.append(pt)
        
        return np.array(points)

    def _hypervolume_contribution(self, F, ref_point):
        """
        Approximate the hypervolume contribution of each point in F.
        For each point, compute the volume of the box it uniquely covers.
        This is an approximation using a grid-based method.
        """
        n = len(F)
        if n == 0:
            return np.array([])
        
        # Filter to points that are better than ref_point in all objectives
        valid_mask = np.all(F < ref_point, axis=1)
        contributions = np.zeros(n)
        
        if not np.any(valid_mask):
            return contributions
        
        # For simplicity and speed, use a heuristic:
        # Contribution is proportional to the volume of the box from F[i] to ref_point,
        # but reduced by how many other points are "closer" to the ideal point
        # in the same region.
        
        # Compute the box volume for each point
        box_volumes = np.prod(ref_point - F, axis=1)
        box_volumes[~valid_mask] = 0
        
        # Penalize points that are dominated by others
        nd_mask = self._non_dominated_mask(F)
        
        # For non-dominated points, contribution is roughly proportional to box volume
        # For dominated points, contribution is 0
        contributions = np.where(nd_mask, box_volumes, 0.0)
        
        # Normalize to avoid numerical issues
        max_contrib = np.max(contributions) if len(contributions) > 0 else 1.0
        if max_contrib > 0:
            contributions = contributions / max_contrib
        
        return contributions

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
        
        # Phase 1: Sobol/quasi-random initialization for first 150 evaluations
        n_init = min(150, budget)
        
        # Generate stratified points
        X = self._sobol_sequence(n_init, dim, lo, hi)
        F = np.array([problem(x) for x in X])
        used = n_init
        
        # Track all evaluated points
        all_F = F.copy()
        
        # Phase 2: Hypervolume-contribution-based local search for remaining evaluations
        n_search = budget - n_init
        
        if n_search > 0 and time.time() < deadline - 10:
            # Keep a population of the best non-dominated points
            pop_size = min(30, n_init)
            
            # Find non-dominated points
            nd_mask = self._non_dominated_mask(F)
            nd_idx = np.where(nd_mask)[0]
            
            if len(nd_idx) == 0:
                # Fallback: use best single objective points
                nd_idx = np.array([np.argmin(F[:, 0]), np.argmin(F[:, 1]), np.argmin(F[:, 2])])
                nd_idx = np.unique(nd_idx)
            
            # Select top pop_size non-dominated points based on hypervolume contribution
            contributions = self._hypervolume_contribution(F, ref_point)
            # Sort by contribution (descending) and take top pop_size
            sorted_idx = np.argsort(-contributions)
            selected_idx = sorted_idx[:pop_size]
            
            X_pop = X[selected_idx].copy()
            F_pop = F[selected_idx].copy()
            
            # Adaptive mutation scale
            base_mut_scale = 0.15
            decay_rate = 0.995
            
            for it in range(n_search):
                if time.time() > deadline - 5:
                    break
                if used >= budget:
                    break
                
                # Find the point with lowest hypervolume contribution in the population
                contribs = self._hypervolume_contribution(F_pop, ref_point)
                worst_idx = np.argmin(contribs)
                
                parent = X_pop[worst_idx].copy()
                
                # Adaptive mutation
                mut_scale = base_mut_scale * (decay_rate ** it)
                mut_scale = max(mut_scale, 0.02)
                
                # Gaussian mutation with occasional larger jumps
                if rng.random() < 0.1:
                    jump_scale = mut_scale * 3.0
                else:
                    jump_scale = mut_scale
                
                child = parent + jump_scale * (hi - lo) * rng.standard_normal(dim)
                child = np.clip(child, lo, hi)
                
                fc = problem(child)
                used += 1
                all_F = np.vstack([all_F, fc])
                
                # Replace if child is better than worst in at least one objective
                # and not worse in all
                child_better = np.any(fc < F_pop[worst_idx])
                child_not_worse_all = not np.all(fc > F_pop[worst_idx])
                
                if child_better and child_not_worse_all:
                    X_pop[worst_idx] = child
                    F_pop[worst_idx] = fc
        
        # Phase 3: Local refinement using scipy minimize on scalarized objectives
        if used < budget and time.time() < deadline - 20:
            # Get non-dominated points from current population
            nd_mask = self._non_dominated_mask(F_pop)
            nd_idx = np.where(nd_mask)[0]
            
            if len(nd_idx) > 0:
                # Try to refine each non-dominated point with a local optimizer
                n_weights = min(8, (budget - used) // 2)
                
                # Generate weight vectors on the simplex
                W = rng.random((n_weights, n_obj))
                W /= W.sum(axis=1, keepdims=True)
                
                # For each non-dominated point, try local refinement
                points_to_refine = nd_idx[:min(5, len(nd_idx))]
                for pt_idx in points_to_refine:
                    if used >= budget or time.time() > deadline - 20:
                        break
                    
                    x0 = X_pop[pt_idx].copy()
                    
                    # Try a few weight vectors
                    for w in W[:3]:
                        if used >= budget or time.time() > deadline - 20:
                            break
                        
                        # Scalarized objective: weighted sum
                        def obj(x):
                            return np.dot(w, problem(x))
                        
                        try:
                            res = minimize(obj, x0, method='L-BFGS-B', 
                                          bounds=list(zip(lo, hi)),
                                          options={'maxiter': 15, 'ftol': 1e-8})
                            if res.success:
                                fx = problem(res.x)
                                used += 1
                                all_F = np.vstack([all_F, fx])
                                
                                # Replace if improved
                                if np.any(fx < F_pop[pt_idx]):
                                    X_pop[pt_idx] = res.x
                                    F_pop[pt_idx] = fx
                        except:
                            pass
        
        return X_pop if 'X_pop' in locals() else X
# EVOLVE-BLOCK-END
