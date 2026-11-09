import numpy as np
import time
from scipy.optimize import minimize

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        timeout = 320  # Safety margin for 330s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        span = hi - lo
        dim = self.dim
        n_obj = self.n_obj
        
        rng = np.random.default_rng(42)
        
        # Track all evaluated points and their objective values
        all_points = []  # list of (x, f)
        all_f = []       # list of f vectors
        
        def record_point(x, f):
            """Record a point and its objective values."""
            x = np.array(x, dtype=float)
            f = np.array(f, dtype=float)
            all_points.append(x)
            all_f.append(f)
            return len(all_points) - 1
        
        def get_pareto_front():
            """Extract Pareto front from all recorded points."""
            n = len(all_f)
            if n == 0:
                return []
            
            F = np.array(all_f)
            pareto_mask = np.ones(n, dtype=bool)
            
            for i in range(n):
                if not pareto_mask[i]:
                    continue
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        pareto_mask[i] = False
                        break
            
            front = []
            for i in range(n):
                if pareto_mask[i]:
                    front.append((all_points[i], all_f[i]))
            return front
        
        def compute_hypervolume(front, ref):
            """Compute 2D hypervolume for a Pareto front."""
            if not front:
                return 0.0
            
            pts = [f for _, f in front if all(f[i] < ref[i] for i in range(len(ref)))]
            if not pts:
                return 0.0
            
            pts.sort(key=lambda p: p[0])
            hv = 0.0
            prev_y = ref[1]
            for p in pts:
                if p[1] < prev_y:
                    hv += (ref[0] - p[0]) * (prev_y - p[1])
                    prev_y = p[1]
            return hv
        
        # Phase 1: Initial sampling using Latin Hypercube + corners
        init_n = min(30, self.budget // 3)
        
        # LHS sampling
        def lhs_sample(n, d, rng):
            points = np.zeros((n, d))
            for j in range(d):
                perm = rng.permutation(n)
                points[:, j] = (perm + rng.random(n)) / n
            return points
        
        lhs_pts = lhs_sample(init_n, dim, rng)
        for pt in lhs_pts:
            x = pt * span + lo
            f = problem(x)
            record_point(x, f)
        
        # Add corner points
        corners = []
        for i in range(dim):
            for j in range(i+1, dim):
                x = np.array([lo[k] if k not in (i,j) else hi[k] for k in range(dim)])
                corners.append(x)
                x2 = np.array([hi[k] if k not in (i,j) else lo[k] for k in range(dim)])
                corners.append(x2)
        
        for x in corners[:min(8, self.budget - len(all_points))]:
            if len(all_points) >= self.budget:
                break
            f = problem(x)
            record_point(x, f)
        
        # Phase 2: Decomposition-based optimization using Tchebycheff aggregation
        # Generate a dense set of weight vectors for systematic coverage
        # For 2 objectives, weights are (w1, w2) with w1 + w2 = 1
        n_weights = 20  # Number of weight vectors
        weights = []
        for i in range(n_weights):
            w1 = i / (n_weights - 1) if n_weights > 1 else 0.5
            w2 = 1 - w1
            weights.append(np.array([w1, w2]))
        
        # Also include extreme weights
        weights.append(np.array([1.0, 0.0]))
        weights.append(np.array([0.0, 1.0]))
        weights.append(np.array([0.5, 0.5]))
        
        # For each weight vector, run a local optimization
        # Use the best point from initial sampling as starting point
        current_front = get_pareto_front()
        
        if current_front:
            # Find best point for each weight using Tchebycheff distance
            best_starts = {}
            for w in weights:
                best_idx = None
                best_val = np.inf
                for i, (x, f) in enumerate(current_front):
                    # Tchebycheff distance to ideal point (use min of objectives)
                    ideal = np.array([min(f_obj[0] for _, f_obj in current_front),
                                     min(f_obj[1] for _, f_obj in current_front)])
                    val = np.max(w * np.abs(f - ideal))
                    if val < best_val:
                        best_val = val
                        best_idx = i
                if best_idx is not None:
                    best_starts[w] = current_front[best_idx][0]
        
        # Run L-BFGS-B for each weight vector
        evals_per_opt = max(10, (self.budget - len(all_points)) // len(weights))
        
        for w in weights:
            if time.time() - start_time > timeout:
                break
            if len(all_points) >= self.budget:
                break
            
            if w not in best_starts:
                continue
            
            x0 = best_starts[w]
            
            def tchebycheff_objective(x, w=w):
                f = problem(x)
                ideal = np.array([min(f_obj[0] for _, f_obj in current_front),
                                 min(f_obj[1] for _, f_obj in current_front)])
                return np.max(w * np.abs(f - ideal))
            
            try:
                res = minimize(
                    tchebycheff_objective,
                    x0,
                    method='L-BFGS-B',
                    bounds=list(zip(lo, hi)),
                    options={'maxiter': 50, 'maxfev': evals_per_opt, 'gtol': 1e-4}
                )
                
                if res.x is not None:
                    f = problem(res.x)
                    record_point(res.x, f)
            except Exception:
                pass
        
        # Phase 3: Refinement - run additional local optimizations from promising points
        current_front = get_pareto_front()
        
        if current_front and len(all_points) < self.budget:
            # Select diverse points from the front for further refinement
            front_pts = [(x, f) for x, f in current_front]
            
            # Sort by first objective and pick a few
            sorted_front = sorted(front_pts, key=lambda x: x[1][0])
            n_refine = min(10, len(sorted_front), (self.budget - len(all_points)) // 15)
            
            # Pick evenly spaced points
            if n_refine > 0:
                indices = np.linspace(0, len(sorted_front)-1, n_refine, dtype=int)
                for idx in indices:
                    if time.time() - start_time > timeout:
                        break
                    if len(all_points) >= self.budget:
                        break
                    
                    x0 = sorted_front[idx][0]
                    
                    # Try multiple weight vectors around this point
                    w_center = np.array([0.5, 0.5])
                    for dw in [-0.2, -0.1, 0.0, 0.1, 0.2]:
                        if len(all_points) >= self.budget:
                            break
                        
                        w = np.array([0.5 + dw, 0.5 - dw])
                        w = np.clip(w, 0.01, 0.99)
                        w = w / np.sum(w)
                        
                        def tchebycheff_obj(x, w=w):
                            f = problem(x)
                            ideal = np.array([min(f_obj[0] for _, f_obj in current_front),
                                             min(f_obj[1] for _, f_obj in current_front)])
                            return np.max(w * np.abs(f - ideal))
                        
                        try:
                            res = minimize(
                                tchebycheff_obj,
                                x0,
                                method='L-BFGS-B',
                                bounds=list(zip(lo, hi)),
                                options={'maxiter': 30, 'maxfev': 20, 'gtol': 1e-5}
                            )
                            
                            if res.x is not None:
                                f = problem(res.x)
                                record_point(res.x, f)
                        except Exception:
                            pass
        
        # Phase 4: Final polish - run Nelder-Mead on the best points
        current_front = get_pareto_front()
        
        if current_front and len(all_points) < self.budget:
            # Sort by hypervolume contribution (approximate by distance to extremes)
            front_pts = [(x, f) for x, f in current_front]
            
            # Pick top 5 diverse points
            sorted_by_f1 = sorted(front_pts, key=lambda x: x[1][0])
            sorted_by_f2 = sorted(front_pts, key=lambda x: x[1][1])
            
            candidates = []
            candidates.append(sorted_by_f1[0])
            candidates.append(sorted_by_f2[0])
            if len(sorted_by_f1) > 1:
                candidates.append(sorted_by_f1[-1])
            if len(sorted_by_f2) > 1:
                candidates.append(sorted_by_f2[-1])
            if len(front_pts) > 2:
                mid = len(front_pts) // 2
                candidates.append(front_pts[mid])
            
            # Remove duplicates
            unique_candidates = []
            for c in candidates:
                if c not in unique_candidates:
                    unique_candidates.append(c)
            
            for x0, f0 in unique_candidates[:5]:
                if time.time() - start_time > timeout:
                    break
                if len(all_points) >= self.budget:
                    break
                
                # Nelder-Mead with a scalar objective (sum of objectives)
                def sum_objective(x):
                    f = problem(x)
                    return np.sum(f)
                
                try:
                    res = minimize(
                        sum_objective,
                        x0,
                        method='Nelder-Mead',
                        options={'maxiter': 30, 'maxfev': 15, 'xatol': 1e-5, 'fatol': 1e-5}
                    )
                    
                    if res.x is not None:
                        f = problem(res.x)
                        record_point(res.x, f)
                except Exception:
                    pass
        
        # Return the Pareto front
        return get_pareto_front()
