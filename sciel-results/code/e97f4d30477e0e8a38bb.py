import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.ref_point = np.array([1.1, 1.3, 1.1])
        
    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        rng = np.random.default_rng(42)
        
        # Archive: list of (x, f)
        archive = []
        evals = 0
        
        def update_archive(new_x, new_f):
            nonlocal archive
            # Check if new_f is dominated by any in archive
            dominated = False
            for _, f_old in archive:
                if np.all(f_old <= new_f) and np.any(f_old < new_f):
                    dominated = True
                    break
            if dominated:
                return False
            
            # Remove points in archive dominated by new_f
            new_archive = []
            for x_old, f_old in archive:
                if not (np.all(new_f <= f_old) and np.any(new_f < f_old)):
                    new_archive.append((x_old, f_old))
            
            new_archive.append((new_x, new_f))
            archive = new_archive
            return True

        # Hypervolume calculation (2D slice method for 3D)
        def hv_2d(points_2d, ref_2d):
            if not points_2d:
                return 0.0
            pts = sorted(points_2d, key=lambda x: x[0]) # Sort by first obj asc
            # Filter non-dominated in 2D
            nd_pts = []
            min_f1 = np.inf
            for p in reversed(pts): # Iterate from lowest f1 to highest
                if p[1] < min_f1: # If f2 is better than all previous (which have higher f1)
                    nd_pts.append(p)
                    min_f1 = p[1]
            
            if not nd_pts:
                return 0.0
            
            hv = 0.0
            r1, r2 = ref_2d
            n = len(nd_pts)
            for i in range(n):
                f1, f2 = nd_pts[i]
                next_f1 = nd_pts[i+1][0] if i < n - 1 else r1
                # Wait, standard HV integration for minimization:
                # Sort by f1 ascending.
                # The contribution of point i is (f1[i+1] - f1[i]) * (r2 - f2[i])?
                # No, let's stick to the previous logic which was:
                # Sort by f1 descending.
                # w1 = f1[i] - f1[i+1] (where f1[n] = 0 or min_bound)
                # This logic assumes the "floor" is 0.
                # Let's use a robust O(N log N) or O(N^2) HV for 3D directly in Python for small N.
                
            # Actually, let's just implement a simple O(N^2) HV for 3D for the archive size ~50-100.
            # It's safer than deriving the 2D slice logic incorrectly again.
            
        def calculate_hv_3d(points, ref):
            # points: list of np.array of shape (3,)
            # ref: np.array of shape (3,)
            if not points:
                return 0.0
            
            # Filter points strictly better than ref
            valid_points = [p for p in points if np.all(p < ref)]
            if not valid_points:
                return 0.0
            
            # Remove dominated points
            nd_points = []
            for i, p_i in enumerate(valid_points):
                dominated = False
                for j, p_j in enumerate(valid_points):
                    if i == j:
                        continue
                    if np.all(p_j <= p_i) and np.any(p_j < p_i):
                        dominated = True
                        break
                if not dominated:
                    nd_points.append(p_i)
            
            if not nd_points:
                return 0.0
            
            # Sort by first objective
            nd_points.sort(key=lambda x: x[0])
            
            # Hypervolume calculation using the "slice" method
            # Volume = sum over slices in f1
            # For each unique f1 value, calculate the 2D HV of the points with f1 <= current_f1?
            # No, standard algorithm:
            # Sort by f1 ascending.
            # Iterate from last to first (highest f1 to lowest).
            # Maintain a list of "active" points in the (f2, f3) plane.
            # This is complex. Let's use a simpler O(N^2) approach for N<200.
            
            # Inclusion-Exclusion Principle is too slow.
            # Let's use the property that HV is the volume of the union of boxes [p, ref].
            # For 3D, we can project to 2D.
            
            # Sort by f1 descending.
            pts_desc = sorted(nd_points, key=lambda x: -x[0])
            
            hv = 0.0
            r1, r2, r3 = ref
            
            # We iterate through points sorted by f1 descending.
            # For each point i, the unique contribution in the f1 direction is the width 
            # (f1[i] - f1[i+1]) times the 2D HV of the points 0..i in the (f2, f3) plane.
            
            n = len(pts_desc)
            for i in range(n):
                f1_i = pts_desc[i][0]
                next_f1 = pts_desc[i+1][0] if i < n - 1 else 0.0
                # Note: The "floor" for f1 is 0 because bounds are [0,1] and we minimize.
                # The box extends from f1_i to r1.
                # The width of the slice is f1_i - next_f1.
                # The height of this slice is the 2D HV of the projection of points[0..i] onto (f2, f3).
                
                width = f1_i - next_f1
                if width <= 0:
                    continue
                
                # Get 2D points
                pts_2d = [(pts_desc[k][1], pts_desc[k][2]) for k in range(i + 1)]
                
                # Calculate 2D HV
                # Sort by f2 descending
                pts_2d_sorted = sorted(pts_2d, key=lambda x: -x[0])
                nd_2d = []
                min_f3 = np.inf
                for p in pts_2d_sorted:
                    if p[1] < min_f3:
                        nd_2d.append(p)
                        min_f3 = p[1]
                        
                if not nd_2d:
                    continue
                    
                hv_2d_val = 0.0
                m = len(nd_2d)
                for k in range(m):
                    f2_k = nd_2d[k][0]
                    f3_k = nd_2d[k][1]
                    next_f2 = nd_2d[k+1][0] if k < m - 1 else 0.0
                    w2 = f2_k - next_f2
                    h3 = r3 - f3_k
                    hv_2d_val += w2 * h3
                    
                hv += width * hv_2d_val
                
            return hv

        def get_marginal_hvs(archive_f_list, ref):
            if not archive_f_list:
                return []
            
            total_hv = calculate_hv_3d(archive_f_list, ref)
            marginals = []
            n = len(archive_f_list)
            for i in range(n):
                reduced = [archive_f_list[j] for j in range(n) if j != i]
                hv_minus = calculate_hv_3d(reduced, ref)
                marginals.append(total_hv - hv_minus)
            return marginals

        # 1. Initialization
        init_n = 30
        X_init = lo + (hi - lo) * rng.random((init_n, self.dim))
        for x in X_init:
            f = problem(x)
            if update_archive(x, f):
                evals += 1
        
        # 2. Main Loop
        while evals < self.budget:
            # Get current non-dominated points
            if not archive:
                x = lo + (hi - lo) * rng.random(self.dim)
                f = problem(x)
                update_archive(x, f)
                evals += 1
                continue
            
            # Calculate marginal HVs to identify "gap fillers"
            # To save time, we only calculate exact marginal HVs if archive is small (< 50)
            # Otherwise, we approximate or just use random + local search.
            # Given budget 200, archive might grow to ~50-100.
            # O(N^2) HV is fine for N=100 (10k ops per HV, 100 HVs = 1M ops).
            
            archive_f_list = [f for _, f in archive]
            marginals = get_marginal_hvs(archive_f_list, self.ref_point)
            
            # Select top K parents
            k_parents = min(5, len(archive))
            top_indices = np.argsort(marginals)[::-1][:k_parents]
            
            candidates = []
            
            # Generate candidates from top parents
            for idx in top_indices:
                parent_x = archive[idx][0]
                parent_f = archive[idx][1]
                
                # Multi-scale perturbation
                # 1. Large jump (exploration)
                sigma_large = 0.5
                x_cand = parent_x + sigma_large * rng.standard_normal(self.dim)
                x_cand = np.clip(x_cand, lo, hi)
                candidates.append(x_cand)
                
                # 2. Medium jump
                sigma_med = 0.2
                for _ in range(2):
                    x_cand = parent_x + sigma_med * rng.standard_normal(self.dim)
                    x_cand = np.clip(x_cand, lo, hi)
                    candidates.append(x_cand)
                    
                # 3. Small jump (exploitation)
                sigma_small = 0.05
                for _ in range(2):
                    x_cand = parent_x + sigma_small * rng.standard_normal(self.dim)
                    x_cand = np.clip(x_cand, lo, hi)
                    candidates.append(x_cand)

            # Add some random candidates for diversity
            n_random = 5
            for _ in range(n_random):
                x_cand = lo + (hi - lo) * rng.random(self.dim)
                candidates.append(x_cand)
            
            # Evaluate candidates
            new_points = []
            for x_cand in candidates:
                if evals >= self.budget:
                    break
                f_cand = problem(x_cand)
                evals += 1
                # Only keep if it improves the archive (non-dominated and not dominated)
                # We do a quick check here to avoid adding dominated points to the "new" list
                # before updating the main archive, which is expensive.
                # Actually, update_archive handles it.
                if update_archive(x_cand, f_cand):
                    new_points.append((x_cand, f_cand))
            
            # Local Refinement on the best new point
            if new_points:
                # Find the new point with the highest marginal HV in the updated archive
                # This is expensive. Instead, just pick the one that was just added and had best raw fitness?
                # Or simply pick the first one.
                # Let's pick the one that is "best" in terms of sum of objectives (simple proxy)
                # or just the first one.
                best_new_x, best_new_f = new_points[0]
                
                # Perform a few steps of Nelder-Mead or similar
                # Since we have budget, let's do a simple coordinate descent or random local search
                # for the best new point.
                current_x = best_new_x.copy()
                current_f = best_new_f.copy()
                
                # 5 steps of local search
                for _ in range(5):
                    if evals >= self.budget:
                        break
                    # Try moving in each dimension
                    improved = False
                    for d in range(self.dim):
                        # Try + and - step
                        for direction in [1, -1]:
                            step = 0.05 * direction
                            trial_x = current_x.copy()
                            trial_x[d] += step
                            trial_x = np.clip(trial_x, lo, hi)
                            
                            trial_f = problem(trial_x)
                            evals += 1
                            
                            # Check if trial_f dominates current_f
                            if np.all(trial_f <= current_f) and np.any(trial_f < current_f):
                                current_x = trial_x
                                current_f = trial_f
                                improved = True
                                # Don't break, keep trying other directions? 
                                # Or break to move to next dim?
                                # Let's keep exploring this dim if improved? 
                                # Simple hill climb: take the step.
                                
                    # If no improvement in any dim, stop local search for this point
                    if not improved:
                        break
                
                # Update archive with the refined point
                update_archive(current_x, current_f)

        return archive
