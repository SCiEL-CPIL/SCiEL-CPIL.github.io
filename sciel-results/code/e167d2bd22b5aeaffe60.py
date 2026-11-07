import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def _latin_hypercube(self, N, dim, lo, hi):
        """Generate Latin Hypercube sample points"""
        points = np.zeros((N, dim))
        for d in range(dim):
            intervals = np.linspace(0, 1, N + 1)
            for i in range(N):
                points[i, d] = lo[d] + (hi[d] - lo[d]) * (intervals[i] + np.random.random() * (intervals[i+1] - intervals[i]))
        for d in range(dim):
            perm = np.random.permutation(N)
            points[:, d] = points[perm, d]
        return points

    def _crowding_distance(self, F):
        """Compute crowding distance for each point in objective space"""
        n = len(F)
        if n <= 2:
            return np.full(n, np.inf)
        
        dist = np.zeros(n)
        for m in range(F.shape[1]):
            idx = np.argsort(F[:, m])
            dist[idx[0]] = np.inf
            dist[idx[-1]] = np.inf
            for i in range(1, n-1):
                dist[idx[i]] += (F[idx[i+1], m] - F[idx[i-1], m]) / (F[idx[-1], m] - F[idx[0], m] + 1e-30)
        return dist

    def _find_non_dominated(self, F):
        """Return indices of non-dominated points"""
        n = len(F)
        dominated = np.zeros(n, dtype=bool)
        for i in range(n):
            if dominated[i]:
                continue
            for j in range(n):
                if i == j or dominated[j]:
                    continue
                if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                    dominated[j] = True
        return np.where(~dominated)[0]

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo
        dim = self.dim
        budget = self.budget
        n_obj = self.n_obj
        
        # Archive for best non-dominated points found
        archive_X = []
        archive_F = []
        used = 0

        def evaluate_and_track(x):
            """Evaluate x and track it. Returns (f, dominated_by_archive)"""
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.array(problem(x), dtype=np.float64)
            used += 1
            return f

        def update_archive(x, f):
            """Add point to archive if non-dominated, remove dominated points"""
            if len(archive_F) == 0:
                archive_X.append(x)
                archive_F.append(f)
                return
            F_arr = np.array(archive_F)
            # Is x dominated by any archive point?
            mask_le = np.all(F_arr <= f, axis=1)
            mask_lt = np.any(F_arr < f, axis=1)
            if not np.any(mask_le & mask_lt):
                # Not dominated; remove archive points dominated by x
                mask_le2 = np.all(f <= F_arr, axis=1)
                mask_lt2 = np.any(f < F_arr, axis=1)
                to_remove = np.where(mask_le2 & mask_lt2)[0]
                for idx in sorted(to_remove, reverse=True):
                    archive_X.pop(idx)
                    archive_F.pop(idx)
                archive_X.append(x)
                archive_F.append(f)

        # Phase 1: Latin Hypercube initialization
        N_init = 30
        X_init = self._latin_hypercube(N_init, dim, lo, hi)
        F_init = np.array([evaluate_and_track(x) for x in X_init])
        
        # Build initial archive from non-dominated points
        non_dom = self._find_non_dominated(F_init)
        for idx in non_dom:
            archive_X.append(X_init[idx])
            archive_F.append(F_init[idx])

        # Phase 2: Evolutionary search with adaptive mutation
        # Reserve more budget for local polish
        n_reserve = min(60, budget // 3)
        
        while used < budget - n_reserve:
            if time.time() > deadline:
                break
            
            # Select parent from archive (or random if empty)
            if len(archive_X) > 0:
                parent = np.array(archive_X[np.random.randint(len(archive_X))])
            else:
                parent = lo + span * np.random.random(dim)
            
            # Adaptive mutation strategies
            progress = used / budget
            base_step = 0.15 * (1.0 - 0.7 * progress)
            
            r = np.random.random()
            if r < 0.25 and len(archive_X) >= 2:
                # Crossover with another archive point
                other_idx = np.random.randint(len(archive_X))
                other = np.array(archive_X[other_idx])
                alpha = np.random.random()
                child = alpha * parent + (1 - alpha) * other
                child += 0.05 * span * np.random.standard_normal(dim)
            elif r < 0.65:
                # Gaussian mutation with adaptive step
                child = parent + base_step * span * np.random.standard_normal(dim)
            else:
                # Larger jump for exploration
                child = parent + 0.3 * span * np.random.standard_normal(dim)
            
            child = np.clip(child, lo, hi)
            f_child = evaluate_and_track(child)
            
            # Update archive: check dominance
            update_archive(child, f_child)

        # Phase 3: Tchebycheff decomposition polish for better Pareto front coverage
        if used < budget and len(archive_F) > 0:
            F_arr = np.array(archive_F)
            # Tchebycheff weight vectors: diverse spread across the front
            # Using weights that cover the full [0,1] range for both objectives
            tcheb_weights = []
            n_w = 10
            for i in range(n_w):
                w1 = i / (n_w - 1)
                w2 = 1.0 - w1
                tcheb_weights.append(np.array([w1, w2]))
            # Add extreme points for boundary exploration
            tcheb_weights.append(np.array([1.0, 0.0]))
            tcheb_weights.append(np.array([0.0, 1.0]))
            
            # Compute reference point from current archive for Tchebycheff
            F_ref = np.max(F_arr, axis=0)
            
            # Sort archive points by hypervolume contribution (approx: worst-case)
            worst_case = np.max(F_arr, axis=1)
            top_idx = np.argsort(worst_case)[:min(6, len(archive_X))]
            
            for idx in top_idx:
                if used >= budget or time.time() > deadline:
                    break
                
                x0 = np.array(archive_X[idx])
                f0 = np.array(archive_F[idx])
                
                for weights in tcheb_weights:
                    if used >= budget or time.time() > deadline:
                        break
                    
                    def obj_func_tcheb(x, w=weights, ref=F_ref):
                        try:
                            f = np.array(problem(np.clip(x, lo, hi)), dtype=np.float64)
                            # Tchebycheff: minimize max_i w_i * (f_i - ref_i)
                            return np.max(w * (f - ref))
                        except:
                            return 1e10
                    
                    try:
                        res = minimize(obj_func_tcheb, x0, method='Nelder-Mead', 
                                     options={'maxiter': 100, 'xatol': 1e-10, 'fatol': 1e-12})
                        if res.success:
                            x_new = np.clip(res.x, lo, hi)
                            f_new = np.array(problem(x_new), dtype=np.float64)
                            used += 1
                            update_archive(x_new, f_new)
                    except:
                        pass

        # Phase 4: Final high-precision polish on remaining archive
        # Tighter tolerance, more iterations for each archive point
        if used < budget and len(archive_F) > 0:
            F_arr = np.array(archive_F)
            F_ref = np.max(F_arr, axis=0)
            
            # Use all archive points for final polish
            for idx in range(len(archive_X)):
                if used >= budget or time.time() > deadline:
                    break
                
                x0 = np.array(archive_X[idx])
                f0 = np.array(archive_F[idx])
                
                # Tchebycheff with fine-tuned weights around this point's position
                w1 = f0[0] / (F_ref[0] + 1e-30)
                w1 = np.clip(w1, 0.05, 0.95)
                w2 = 1.0 - w1
                w = np.array([w1, w2])
                
                def obj_func_final(x, w=w, ref=F_ref):
                    try:
                        f = np.array(problem(np.clip(x, lo, hi)), dtype=np.float64)
                        return np.max(w * (f - ref))
                    except:
                        return 1e10
                
                try:
                    res = minimize(obj_func_final, x0, method='Nelder-Mead',
                                 options={'maxiter': 150, 'xatol': 1e-12, 'fatol': 1e-14})
                    if res.success:
                        x_new = np.clip(res.x, lo, hi)
                        f_new = np.array(problem(x_new), dtype=np.float64)
                        used += 1
                        update_archive(x_new, f_new)
                except:
                    pass
        
        # Return the best points (return value is ignored, but we return something sensible)
        if len(archive_X) > 0:
            return np.array(archive_X)
        else:
            return np.array([])
