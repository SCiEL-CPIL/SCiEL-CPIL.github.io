"""Multi-objective optimizer for RE34 with hypervolume-aware evolutionary search."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


class Optimizer:
    """Multi-objective optimizer for RE34 (5 variables, 3 objectives, bounds [1,3])."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _hv_3d(self, pts, ref):
        """Compute 3D hypervolume using inclusion-exclusion via 2D slices."""
        if len(pts) == 0:
            return 0.0
        m = np.all(pts < ref, axis=1)
        p = pts[m]
        if len(p) == 0:
            return 0.0
        # Sort by first objective
        p = p[np.argsort(p[:, 0])]
        n = len(p)
        hv = 0.0
        for i in range(n):
            next_f0 = p[i + 1, 0] if i + 1 < n else ref[0]
            width = next_f0 - p[i, 0]
            # 2D hypervolume of remaining points projected onto dims 1,2
            sub = p[i:, 1:]
            ref2 = ref[1:]
            # Filter points better than ref in both dims
            mask2 = np.all(sub < ref2, axis=1)
            sub = sub[mask2]
            if len(sub) == 0:
                continue
            sub = sub[np.argsort(sub[:, 0])]
            area = 0.0
            for j in range(len(sub)):
                next_x = sub[j + 1, 0] if j + 1 < len(sub) else ref2[0]
                w2 = next_x - sub[j, 0]
                h2 = ref2[1] - sub[j, 1]
                if w2 > 0 and h2 > 0:
                    area += w2 * h2
            hv += width * area
        return hv

    def _hv_contributions(self, pts, ref):
        """Compute per-point hypervolume contributions for 3D points."""
        n = len(pts)
        if n == 0:
            return np.array([])
        mask = np.all(pts < ref, axis=1)
        if not np.any(mask):
            return np.zeros(n)
        pts_f = pts[mask]
        idx = np.where(mask)[0]
        m = len(pts_f)
        contrib = np.zeros(n)
        full_hv = self._hv_3d(pts_f, ref)
        for k in range(m):
            mask_remove = np.ones(m, dtype=bool)
            mask_remove[k] = False
            hv_without = self._hv_3d(pts_f[mask_remove], ref)
            contrib[idx[k]] = full_hv - hv_without
        return contrib

    def _non_dominated_mask(self, F):
        """Return boolean mask of non-dominated points."""
        n = len(F)
        if n == 0:
            return np.array([], dtype=bool)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            for j in range(n):
                if i != j and np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                    dominated[i] = True
                    break
        return ~dominated

    def _local_refine(self, x0, problem, bounds, max_evals=5):
        """Local refinement using L-BFGS-B on weighted sum."""
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random
        
        # Try a few random weight vectors
        best_x = x0.copy()
        best_f = np.array(problem(x0))
        evals = 0
        
        for _ in range(3):
            if evals >= max_evals:
                break
            w = rng.dirichlet(np.ones(3))
            
            def objective(x):
                nonlocal evals
                fx = np.array(problem(x))
                evals += 1
                return np.dot(w, fx)
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B',
                              bounds=list(bounds),
                              options={'maxiter': 10, 'ftol': 1e-10})
                if res.success and res.fun < np.dot(w, best_f):
                    best_x = res.x.copy()
                    best_f = np.array(problem(res.x))
                    evals += 1
            except:
                pass
        
        return best_x, best_f, evals

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280  # Safety margin before 300s limit

        rng = np.random
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        ref = np.array([1705.0, 12.0, 0.3])

        all_X = []
        all_F = []
        used = 0

        # Phase 1: Diverse random initialization
        pop_size = 30
        X = lo + (hi - lo) * rng.random((pop_size, self.dim))
        F = np.array([problem(x) for x in X])
        used += pop_size
        all_X.extend(X.tolist())
        all_F.extend(F.tolist())

        # Phase 2: Weighted-sum grid search for broad coverage
        # Use a set of weight vectors to explore different parts of the front
        n_weights = 15
        # Create weight vectors on the simplex
        weights = []
        for i in range(n_weights):
            for j in range(n_weights - i - 1):
                w1 = (i + 0.5) / n_weights
                w2 = (j + 0.5) / (n_weights - i - 1)
                w3 = 1 - w1 - w2
                if w3 > 0:
                    weights.append([w1, w2, w3])
        
        weights = np.array(weights)
        
        for w in weights:
            if used >= self.budget or time.time() > deadline:
                break
            
            def weighted_obj(x, w=w):
                fx = np.array(problem(x))
                return np.dot(w, fx)
            
            # Start from a random point
            x0 = lo + (hi - lo) * rng.random(self.dim)
            
            try:
                res = minimize(weighted_obj, x0, method='L-BFGS-B',
                              bounds=list(bounds),
                              options={'maxiter': 20, 'ftol': 1e-8})
                if res.success:
                    x_opt = res.x
                    f_opt = np.array(problem(x_opt))
                    used += 1
                    all_X.append(x_opt)
                    all_F.append(f_opt)
                    
                    # Add to population if non-dominated or improves HV
                    if used < self.budget:
                        # Check if this point is good
                        F_new = np.vstack([F, f_opt.reshape(1, -1)])
                        nd_mask = self._non_dominated_mask(F_new)
                        if nd_mask[-1]:  # New point is non-dominated
                            X = np.vstack([X, x_opt.reshape(1, -1)])
                            F = F_new
                        else:
                            # Replace worst point
                            dominated = np.where(~nd_mask[:-1])[0]
                            if len(dominated) > 0:
                                worst = dominated[np.argmax(np.sum(F[dominated], axis=1))]
                                X[worst] = x_opt
                                F[worst] = f_opt
            except:
                pass

        # Phase 3: Evolutionary refinement
        # Ensure we have a population
        if len(X) == 0:
            X = lo + (hi - lo) * rng.random((10, self.dim))
            F = np.array([problem(x) for x in X])
            used += 10
            all_X.extend(X.tolist())
            all_F.extend(F.tolist())
        
        X = np.array(X)
        F = np.array(F)
        
        while used < self.budget:
            if time.time() > deadline:
                break
            
            # Compute non-dominated front
            nd_mask = self._non_dominated_mask(F)
            nd_indices = np.where(nd_mask)[0]
            
            if len(nd_indices) == 0:
                parent_idx = rng.randint(len(F))
                victim_idx = int(np.argmax(np.sum(F, axis=1)))
            else:
                # Select parent from non-dominated front
                parent_idx = nd_indices[rng.randint(len(nd_indices))]
                
                # Select victim
                dom_mask = ~nd_mask
                dom_indices = np.where(dom_mask)[0]
                if len(dom_indices) > 0:
                    victim_idx = dom_indices[int(np.argmax(np.sum(F[dom_indices], axis=1)))]
                else:
                    # All non-dominated: remove point with lowest HV contribution
                    contribs = self._hv_contributions(F, ref)
                    nd_contribs = contribs[nd_indices]
                    min_pos = np.argmin(nd_contribs)
                    victim_idx = nd_indices[min_pos]
            
            parent = X[parent_idx].copy()
            
            # Adaptive mutation
            progress = used / self.budget
            step = (0.3 - 0.27 * progress) * (hi - lo)
            
            # Gaussian mutation
            child = parent + step * rng.standard_normal(self.dim)
            child = np.clip(child, lo, hi)
            
            fc = problem(child)
            used += 1
            
            all_X.append(child)
            all_F.append(fc)
            
            # Replace victim in population
            X[victim_idx] = child
            F[victim_idx] = fc
            
            # Periodically do local refinement on best points
            if used % 20 == 0 and used < self.budget - 10:
                # Refine the best non-dominated point
                if len(nd_indices) > 0:
                    best_idx = nd_indices[0]
                    x_refined, f_refined, evals_used = self._local_refine(
                        X[best_idx], problem, bounds, max_evals=5
                    )
                    used += evals_used
                    if evals_used > 0:
                        all_X.append(x_refined)
                        all_F.append(f_refined)
                        # Check if improvement
                        if np.all(f_refined <= F[best_idx]) and np.any(f_refined < F[best_idx]):
                            X[best_idx] = x_refined
                            F[best_idx] = f_refined

        return all_X

# EVOLVE-BLOCK-END
