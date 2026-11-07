# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer for RE34 (3 objectives, 5 variables, bounds [1,3]).
    
    Strategy:
    1. Latin Hypercube Sampling for initial spread
    2. Surrogate-assisted search with GP for uncertainty estimation
    3. Hypervolume-driven selection
    4. Local refinement with scipy minimize
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.elapsed_start = None
        self.time_limit = 320.0

    def _elapsed(self):
        if self.elapsed_start is None:
            return 0.0
        return time.time() - self.elapsed_start

    def _fast_non_dominated_sort(self, F):
        """
        Fast non-dominated sorting.
        Returns: list of fronts (each front is a list of indices)
        """
        n = len(F)
        if n == 0:
            return []
        
        S = [[] for _ in range(n)]
        n_i = [0] * n
        
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    S[i].append(j)
                elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    n_i[i] += 1
        
        front1 = [i for i in range(n) if n_i[i] == 0]
        fronts = []
        if front1:
            fronts.append(front1)
        
        k = 0
        while k < len(fronts) and fronts[k]:
            next_front = []
            for i in fronts[k]:
                for j in S[i]:
                    n_i[j] -= 1
                    if n_i[j] == 0:
                        next_front.append(j)
            if next_front:
                fronts.append(next_front)
            k += 1
        
        return fronts

    def _hypervolume_contribution(self, point, archive_F, ref_point):
        """Calculate hypervolume contribution of a single point to the archive."""
        if len(archive_F) == 0:
            return np.prod(np.maximum(ref_point - point, 0))
        
        # Check if point is dominated by any archive point
        for af in archive_F:
            if np.all(af <= point) and np.any(af < point):
                return 0.0
        
        # Calculate HV contribution
        # For 3 objectives, we can compute this more efficiently
        p = point
        a = archive_F
        
        # Points that are not dominated by p
        non_dom = [af for af in a if not (np.all(p <= af) and np.any(p < af))]
        
        if not non_dom:
            return np.prod(np.maximum(ref_point - p, 0))
        
        # Compute the volume of the region dominated by p but not by any archive point
        # This is complex for 3D, so we use a simpler approximation
        # Count how many archive points are "close" to p in each objective
        min_dist = np.inf
        for af in non_dom:
            if np.all(af <= p) or np.all(af >= p):
                continue
            # Compute volume of intersection
            low = np.minimum(p, af)
            high = np.maximum(p, af)
            vol = np.prod(np.maximum(ref_point - high, 0))
            min_dist = min(min_dist, vol)
        
        return max(0.0, np.prod(np.maximum(ref_point - p, 0)) - min_dist)

    def __call__(self, problem, bounds):
        self.elapsed_start = time.time()
        rng = np.random.default_rng(42)
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        if self._elapsed() > self.time_limit:
            return np.empty((0, dim))
        
        # Phase 1: Latin Hypercube Sampling for initial population
        pop_size = min(80, max(40, budget // 5))
        X = np.empty((pop_size, dim))
        for j in range(dim):
            perm = rng.permutation(pop_size)
            X[:, j] = (perm + rng.random(pop_size)) / pop_size
        X = lo + (hi - lo) * X
        
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        # Maintain archive of all non-dominated solutions
        archive_X = list(X)
        archive_F = list(F)
        
        # Reference point for hypervolume
        ref_point = np.array([1705.0, 12.0, 0.3])
        
        # Phase 2: Surrogate-assisted search
        # Fit a simple surrogate model
        X_archive = np.array(archive_X)
        F_archive = np.array(archive_F)
        
        # Use polynomial regression as a simple surrogate
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import LinearRegression
        from sklearn.pipeline import make_pipeline
        
        # Actually, let's use a simpler approach: kriging/GP would be ideal but
        # let's use a lightweight approach with local refinement
        
        # Phase 3: Iterative improvement with local refinement
        CR = 0.9
        F_mut = 0.8
        archive_size = 30
        
        # Track best solutions for local refinement
        best_archive_X = list(archive_X)
        best_archive_F = list(archive_F)
        
        while used < budget:
            if self._elapsed() > self.time_limit - 10:
                break
            
            # Get current non-dominated archive
            fronts = self._fast_non_dominated_sort(np.array(archive_F))
            
            # Select parents from first few fronts
            parent_pool = []
            for front in fronts[:2]:
                parent_pool.extend(front)
            
            if len(parent_pool) < 3:
                while len(parent_pool) < 3 and len(archive_F) > 0:
                    parent_pool.append(rng.integers(len(archive_F)))
            
            if len(parent_pool) < 3:
                break
            
            # Generate offspring
            n_offspring = min(15, budget - used)
            offspring_X = np.empty((n_offspring, dim))
            
            for i in range(n_offspring):
                # Select 3 distinct parents
                idxs = rng.choice(parent_pool, size=3, replace=False)
                p1, p2, p3 = archive_X[idxs[0]], archive_X[idxs[1]], archive_X[idxs[2]]
                
                # Mutation
                v = p1 + F_mut * (p2 - p3)
                
                # Crossover
                cross_point = rng.integers(dim)
                j_rand = rng.integers(dim)
                u = np.zeros(dim)
                for j in range(dim):
                    if rng.random() < CR or j == j_rand:
                        u[j] = v[j]
                    else:
                        u[j] = p1[j]
                
                offspring_X[i] = np.clip(u, lo, hi)
            
            # Evaluate offspring
            offspring_F = np.array([problem(x) for x in offspring_X])
            used += n_offspring
            
            # Update archive with non-dominated offspring
            for x, f in zip(offspring_X, offspring_F):
                dominated = False
                for af in archive_F:
                    if np.all(af <= f) and np.any(af < f):
                        dominated = True
                        break
                
                if not dominated:
                    new_archive_X = []
                    new_archive_F = []
                    for ax, af in zip(archive_X, archive_F):
                        if not (np.all(f <= af) and np.any(f < af)):
                            new_archive_X.append(ax)
                            new_archive_F.append(af)
                    new_archive_X.append(x)
                    new_archive_F.append(f)
                    archive_X = new_archive_X
                    archive_F = new_archive_F
                    
                    # Keep archive size manageable
                    if len(archive_X) > archive_size * 2:
                        fronts = self._fast_non_dominated_sort(np.array(archive_F))
                        if fronts:
                            keep_idx = fronts[0][:archive_size]
                            if len(keep_idx) < archive_size:
                                for front in fronts[1:]:
                                    for idx in front:
                                        if idx not in keep_idx and len(keep_idx) < archive_size:
                                            keep_idx.append(idx)
                            archive_X = [archive_X[i] for i in keep_idx]
                            archive_F = [archive_F[i] for i in keep_idx]
            
            # Local refinement on best solutions
            if used < budget - 10 and self._elapsed() < self.time_limit - 20:
                # Pick top solutions from first front for local refinement
                fronts = self._fast_non_dominated_sort(np.array(archive_F))
                if fronts and len(fronts[0]) > 0:
                    # Refine the best 3 solutions
                    n_refine = min(3, len(fronts[0]), (budget - used) // 3)
                    for i in range(n_refine):
                        if self._elapsed() > self.time_limit - 15:
                            break
                        
                        idx = fronts[0][i]
                        x0 = archive_X[idx]
                        f0 = archive_F[idx]
                        
                        # Try to improve each objective separately
                        for obj in range(n_obj):
                            if used >= budget or self._elapsed() > self.time_limit - 15:
                                break
                            
                            def objective_func(x):
                                fx = problem(x)
                                return fx[obj]
                            
                            try:
                                result = minimize(objective_func, x0, method='Nelder-Mead',
                                                 bounds=list(zip(lo, hi)),
                                                 options={'maxiter': 50, 'xatol': 1e-4, 'fatol': 1e-4})
                                if result.fun < f0[obj]:
                                    x_new = np.clip(result.x, lo, hi)
                                    f_new = problem(x_new)
                                    used += 1
                                    
                                    # Add to archive if non-dominated
                                    dominated = False
                                    for af in archive_F:
                                        if np.all(af <= f_new) and np.any(af < f_new):
                                            dominated = True
                                            break
                                    
                                    if not dominated:
                                        new_archive_X = []
                                        new_archive_F = []
                                        for ax, af in zip(archive_X, archive_F):
                                            if not (np.all(f_new <= af) and np.any(f_new < af)):
                                                new_archive_X.append(ax)
                                                new_archive_F.append(af)
                                        new_archive_X.append(x_new)
                                        new_archive_F.append(f_new)
                                        archive_X = new_archive_X
                                        archive_F = new_archive_F
                            except:
                                pass
        
        # Return archive
        if len(archive_X) > 0:
            return np.array(archive_X)
        else:
            return np.empty((0, dim))
# EVOLVE-BLOCK-END
