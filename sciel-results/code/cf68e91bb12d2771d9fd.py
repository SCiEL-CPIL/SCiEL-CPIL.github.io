import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # 20s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # State
        X_all = []
        F_all = []
        
        def evaluate(x):
            nonlocal X_all, F_all
            x = np.clip(x, lo, hi)
            f = problem(x)
            X_all.append(x)
            F_all.append(f)
            return f

        # 1. Initialization: Sobol sequence for good coverage
        n_init = 40
        try:
            from scipy.stats import qmc
            sampler = qmc.Sobol(d=self.dim, scramble=True, seed=42)
            samples = sampler.random(n_init)
        except ImportError:
            samples = np.random.random((n_init, self.dim))
        
        X_init = lo + (hi - lo) * samples
        for x in X_init:
            evaluate(x)
        
        # Helper: Get non-dominated indices
        def get_non_dominated_indices(F):
            n = len(F)
            nd = []
            for i in range(n):
                dominated = False
                for j in range(n):
                    if i == j:
                        continue
                    if np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        dominated = True
                        break
                if not dominated:
                    nd.append(i)
            return nd

        # Helper: Compute Hypervolume Contribution for 2D
        # Reference point: [3000.0, 0.06]
        ref = np.array([3000.0, 0.06])
        
        def hv_contributions(F_nd):
            # F_nd: list of 2D vectors (non-dominated)
            # Sort by f1
            if not F_nd:
                return []
            
            # Filter valid (strictly better than ref)
            valid = [(f[0], f[1]) for f in F_nd if f[0] < ref[0] and f[1] < ref[1]]
            if not valid:
                return [0.0] * len(F_nd)
                
            valid_sorted = sorted(valid, key=lambda x: x[0])
            
            # The HV contribution of a point in a 2D non-dominated set is the area of the rectangle
            # (f1_i, f2_i) to (f1_next, ref[1]) or (ref[0], ref[1]) if it's the last one.
            # Actually, standard definition:
            # Sort by f1 ascending.
            # Contribution of point i: (f1_{i+1} - f1_i) * (ref[1] - f2_i)
            # Wait, this assumes f2 is decreasing as f1 increases (which is true for non-dominated).
            # Let's verify.
            # If we have points (1, 10), (2, 5), (3, 1) and ref (10, 10).
            # Point (1, 10): covers x in [1, 2], y in [10, 10]? No.
            # The region dominated by (1, 10) but not by others is x in [1, 2] and y in [10, 10]?
            # No, the region is x in [1, 2] and y in [10, 10] is empty.
            # The region is x in [1, 2] and y in [10, 10] ... wait.
            # A point (f1, f2) dominates (x, y) if f1 <= x and f2 <= y.
            # The unique contribution of (f1_i, f2_i) is the set of points (x, y) such that
            # f1_i <= x, f2_i <= y, and NOT (f1_j <= x and f2_j <= y for any other j).
            
            # In 2D, sorted by f1:
            # Point 0: (f1_0, f2_0). Next point is (f1_1, f2_1) with f1_1 > f1_0 and f2_1 < f2_0.
            # The region uniquely dominated by Point 0 is x in [f1_0, f1_1) and y in [f2_0, ref[1]).
            # Area = (f1_1 - f1_0) * (ref[1] - f2_0).
            # For the last point k: x in [f1_k, ref[0]) and y in [f2_k, ref[1]).
            # Area = (ref[0] - f1_k) * (ref[1] - f2_k).
            
            contribs = [0.0] * len(valid_sorted)
            for i in range(len(valid_sorted)):
                f1, f2 = valid_sorted[i]
                if i < len(valid_sorted) - 1:
                    next_f1 = valid_sorted[i+1][0]
                else:
                    next_f1 = ref[0]
                
                width = next_f1 - f1
                height = ref[1] - f2
                contribs[i] = width * height
            
            # Map back to original indices
            # This is tricky because we lost the original indices.
            # For selection, we just need to know which points in F_nd have high contribution.
            # We can return a list of (original_index_in_F_nd, contribution)
            # But F_nd is a list of values, not indices.
            # Let's change strategy: return contributions corresponding to the sorted list,
            # then we need to map back.
            
            # Simpler: Just return the contributions for the sorted list.
            # The caller can identify the point with max contribution by checking which point in F_nd matches the sorted point.
            # Or better: pass indices.
            
            return contribs

        def select_best_contributor(F_nd_indices, F_all):
            # F_nd_indices: indices into F_all
            # Returns index into F_all of the point with max HV contribution
            
            F_nd = [F_all[i] for i in F_nd_indices]
            
            # Filter valid
            valid_indices_in_nd = []
            valid_points = []
            for i, f in enumerate(F_nd):
                if f[0] < ref[0] and f[1] < ref[1]:
                    valid_indices_in_nd.append(i)
                    valid_points.append(f)
            
            if not valid_points:
                # Fallback: pick random
                return F_nd_indices[np.random.randint(len(F_nd_indices))]
            
            valid_points_sorted = sorted(valid_points, key=lambda x: x[0])
            
            # Calculate contributions
            max_contrib = -1.0
            best_point = None
            for i in range(len(valid_points_sorted)):
                f1, f2 = valid_points_sorted[i]
                if i < len(valid_points_sorted) - 1:
                    next_f1 = valid_points_sorted[i+1][0]
                else:
                    next_f1 = ref[0]
                
                contrib = (next_f1 - f1) * (ref[1] - f2)
                if contrib > max_contrib:
                    max_contrib = contrib
                    best_point = (f1, f2)
            
            # Find the index in F_nd_indices corresponding to best_point
            # Note: There might be duplicates? Unlikely with floats.
            for i in valid_indices_in_nd:
                if np.allclose(F_nd[i], best_point):
                    return F_nd_indices[i]
            
            # Fallback
            return F_nd_indices[np.random.randint(len(F_nd_indices))]

        # 2. Main Loop: Interpolation and Local Refinement
        # We have budget - n_init left.
        
        # Phase A: Interpolation
        # Identify the "gaps" in the Pareto front.
        # We will iteratively pick the point with the largest HV contribution (which corresponds to the largest gap)
        # and try to find a point in between it and its neighbor (in objective space) by interpolating in decision space?
        # No, interpolating in decision space doesn't guarantee objective space interpolation.
        # Instead, we can use the "HypE" idea: sample a target point in the gap, and use a local optimizer to find x that maps to it.
        # But we don't have a surrogate.
        
        # Alternative: 
        # 1. Pick the point with max HV contribution (call it P_max).
        # 2. Pick a neighbor P_neigh (the one defining the gap).
        # 3. Create a new point by averaging their decision vectors? 
        #    x_new = 0.5 * (x_P_max + x_P_neigh)
        # 4. Evaluate.
        # 5. If it improves HV, keep it.
        
        # This is a simple "crossover" or "interpolation" strategy.
        
        # Phase B: Local Refinement
        # For the top K contributors, run Nelder-Mead to minimize a scalarized objective (e.g., Tchebycheff or weighted sum)
        # or just minimize the distance to the "ideal" point in the gap.
        
        # Let's implement a hybrid:
        # For each step:
        # 1. Get ND set.
        # 2. If ND set size < 2, do random exploration.
        # 3. Identify the "critical" point (max HV contribution) and its neighbor.
        # 4. Generate candidate via interpolation of decision vectors.
        # 5. Also generate candidate via Nelder-Mead on a surrogate? No, expensive.
        # 6. Just evaluate the interpolation candidate.
        # 7. If budget allows, also do a local search (Nelder-Mead) from the best contributor to minimize a weighted sum of objectives, where weights are chosen to target the gap.
        
        # Actually, Nelder-Mead is deterministic and fast for 4D.
        # Let's allocate 50% of remaining budget to Interpolation/Crossover and 50% to Local Search.
        
        remaining_budget = self.budget - len(X_all)
        n_interp = remaining_budget // 2
        n_local = remaining_budget - n_interp
        
        # Phase A: Interpolation
        for _ in range(n_interp):
            if time.time() > deadline:
                break
            if len(X_all) >= self.budget:
                break
                
            nd_indices = get_non_dominated_indices(F_all)
            if len(nd_indices) < 2:
                # Random exploration
                x_new = lo + (hi - lo) * np.random.random(self.dim)
                evaluate(x_new)
                continue
            
            # Find point with max HV contribution
            # We need to identify the gap.
            # Sort ND points by f1.
            nd_points = [(F_all[i][0], F_all[i][1], i) for i in nd_indices if F_all[i][0] < ref[0] and F_all[i][1] < ref[1]]
            if not nd_points:
                x_new = lo + (hi - lo) * np.random.random(self.dim)
                evaluate(x_new)
                continue
                
            nd_points.sort(key=lambda x: x[0])
            
            # Calculate contributions
            best_contrib = -1
            best_i = 0
            best_j = 0 # neighbor
            
            for k in range(len(nd_points)):
                f1, f2, idx = nd_points[k]
                if k < len(nd_points) - 1:
                    next_f1 = nd_points[k+1][0]
                    next_idx = nd_points[k+1][2]
                else:
                    next_f1 = ref[0]
                    next_idx = -1
                
                contrib = (next_f1 - f1) * (ref[1] - f2)
                if contrib > best_contrib:
                    best_contrib = contrib
                    best_i = idx
                    if next_idx != -1:
                        best_j = next_idx
                    else:
                        # If it's the last point, the neighbor is the previous one
                        if k > 0:
                            best_j = nd_points[k-1][2]
                        else:
                            best_j = idx # self? no, pick random other
                
            # If best_j is -1 or same as best_i, pick another
            if best_j == -1 or best_j == best_i:
                # Pick a random other ND point
                candidates = [i for i in nd_indices if i != best_i]
                if candidates:
                    best_j = candidates[np.random.randint(len(candidates))]
                else:
                    best_j = best_i
            
            # Interpolate in decision space
            x1 = X_all[best_i]
            x2 = X_all[best_j]
            
            # Try a few interpolation coefficients
            alpha = np.random.uniform(0.2, 0.8)
            x_new = alpha * x1 + (1 - alpha) * x2
            x_new = np.clip(x_new, lo, hi)
            
            evaluate(x_new)

        # Phase B: Local Refinement (Nelder-Mead)
        # For the remaining budget, run local searches from the best contributors.
        # Each Nelder-Mead run takes ~50-100 evals for 4D.
        # So we can run ~1-2 local searches.
        
        # We will run local search from the top 2 HV contributors.
        
        # Helper: Nelder-Mead for multi-objective?
        # We can minimize a scalarized objective: f_scalar = w1*f1 + w2*f2
        # Weights w can be chosen to target the "gap".
        # If we are refining point i which has a large gap to the right (higher f1), we want to decrease f1.
        # So weight on f1 should be high?
        
        # Simpler: Just minimize the sum of objectives from the best starting points.
        # Or minimize distance to a target point in objective space.
        
        # Let's use scipy.optimize.minimize with Nelder-Mead.
        from scipy.optimize import minimize
        
        # Get top contributors
        nd_indices = get_non_dominated_indices(F_all)
        if not nd_indices:
            return None
            
        # Sort by HV contribution (approximate: just use the previous logic)
        # Re-use the logic to find top 2
        nd_points = [(F_all[i][0], F_all[i][1], i) for i in nd_indices if F_all[i][0] < ref[0] and F_all[i][1] < ref[1]]
        nd_points.sort(key=lambda x: x[0])
        
        contribs = []
        for k in range(len(nd_points)):
            f1, f2, idx = nd_points[k]
            if k < len(nd_points) - 1:
                next_f1 = nd_points[k+1][0]
            else:
                next_f1 = ref[0]
            contrib = (next_f1 - f1) * (ref[1] - f2)
            contribs.append((contrib, idx))
            
        contribs.sort(reverse=True)
        
        # Take top 2
        top_indices = [idx for _, idx in contribs[:2]]
        
        # Run local search from each
        for start_idx in top_indices:
            if time.time() > deadline:
                break
            if len(X_all) >= self.budget:
                break
                
            x0 = X_all[start_idx]
            
            # Define objective function for Nelder-Mead
            # We want to minimize the weighted sum.
            # Let's use equal weights for simplicity, or adjust based on which objective is "worse".
            # Since we want to improve HV, and HV is area, moving towards the origin (0,0) is generally good.
            # So minimizing f1 + f2 is a reasonable proxy.
            
            def obj(x):
                if len(X_all) >= self.budget:
                    return float('inf')
                x = np.clip(x, lo, hi)
                f = problem(x)
                X_all.append(x)
                F_all.append(f)
                return f[0] + f[1]
            
            # Nelder-Mead options
            # maxiter is not exactly evals, but we can limit it.
            # For 4D, 100 iterations is usually enough.
            # Each iteration does ~5 evals. So 500 evals? Too many.
            # Let's limit maxiter to 10.
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 20, 'xatol': 1e-4, 'fatol': 1e-6})
            except Exception:
                pass
            
        return None
