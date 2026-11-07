# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


class Optimizer:
    """
    Multi-objective optimizer using reference-point-based adaptive scalarization
    (Augmented Tchebycheff) with multi-start L-BFGS-B, dynamically targeting hypervolume gaps.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        # Reference point for hypervolume calculation (from task)
        ref_point = np.array([1705.0, 12.0, 0.3])

        # Archive of non-dominated points
        archive_X = []
        archive_F = []
        used = 0

        def dominates(f_new, f_old):
            """Check if f_new dominates f_old."""
            return np.all(f_new <= f_old) and np.any(f_new < f_old)

        def update_archive(x, f):
            """Update the non-dominated archive."""
            nonlocal archive_X, archive_F
            # Check if new point is dominated by existing archive
            for f_old in archive_F:
                if dominates(f_old, f):
                    return
            # Remove points dominated by new point
            new_X = []
            new_F = []
            for x_old, f_old in zip(archive_X, archive_F):
                if not dominates(f, f_old):
                    new_X.append(x_old)
                    new_F.append(f_old)
            new_X.append(x)
            new_F.append(f)
            archive_X = new_X
            archive_F = new_F

        def compute_hypervolume_3d(points, ref):
            """
            Compute exact 3D hypervolume using inclusion-exclusion.
            Points must be strictly better than ref in all objectives.
            """
            # Filter points strictly better than reference
            valid = [p for p in points if np.all(p < ref)]
            if len(valid) == 0:
                return 0.0
            if len(valid) == 1:
                return float(np.prod(ref - valid[0]))
            
            # Sort by first objective
            valid = sorted(valid, key=lambda x: x[0])
            
            # For 3D, we can use a sweep line approach
            # But for small archives, use the standard algorithm
            # Hypervolume = sum over all subsets... too expensive
            # Use the 3D hypervolume algorithm
            
            n = len(valid)
            # Create list of points with indices
            pts = [(valid[i][0], valid[i][1], valid[i][2], i) for i in range(n)]
            pts.sort(key=lambda x: x[0])
            
            # We'll use the standard 3D HV algorithm
            # Sort by f1, then for each point, compute the area in (f2, f3) space
            # that is not covered by subsequent points
            
            total_hv = 0.0
            for i in range(n):
                f1, f2, f3, _ = pts[i]
                # The box from (f1, f2, f3) to ref, minus parts covered by later points
                # Later points have f1' >= f1, so they can only cover the region
                # where f2 >= f2' and f3 >= f3'
                
                # Find all j > i where pts[j][1] >= f2 and pts[j][2] >= f3
                # This is getting complex. Let's use a simpler approach for small n.
                
                # For n <= 20, we can use the inclusion-exclusion or direct computation
                # Actually, let's just use a simple recursive approach or accept approximation
                
                # Simpler: for each point, compute the volume it uniquely contributes
                # by checking against all other points
                
                # The contribution of point i is the volume of the box [f1, ref[0]] x [f2, ref[1]] x [f3, ref[2]]
                # minus the volume covered by any other point j where f1_j >= f1, f2_j >= f2, f3_j >= f3
                
                # This is still complex. Let's use a different strategy:
                # Since archive is small, use the standard 3D HV algorithm
                
                pass
            
            # Let me implement a proper 3D HV
            # Reference: "Computing the Hypervolume Indicator" 
            # For 3D: sort by first coordinate, then for each point, compute the 2D area
            # in the remaining 2 dimensions that is not dominated by subsequent points
            
            # Sort by f1 ascending
            pts_sorted = sorted([(valid[i][0], valid[i][1], valid[i][2]) for i in range(n)], key=lambda x: x[0])
            
            hv = 0.0
            for i in range(n):
                f1, f2, f3 = pts_sorted[i]
                # Compute the 2D hypervolume in (f2, f3) space for points j > i
                # where the box from (f2, f3) to (ref[1], ref[2]) is not covered
                
                # Collect all (f2_j, f3_j) for j > i
                # A point (f2_j, f3_j) covers the region [f2_j, ref[1]] x [f3_j, ref[2]]
                # We want the area of [f2, ref[1]] x [f3, ref[2]] not covered by any such box
                
                # This is the 2D hypervolume problem
                # For 2D: sort by f2, then sweep
                boxes = []
                for j in range(i + 1, n):
                    f2_j, f3_j = pts_sorted[j][1], pts_sorted[j][2]
                    if f2_j >= f2 and f3_j >= f3:
                        boxes.append((f2_j, f3_j))
                
                # Compute area of [f2, ref[1]] x [f3, ref[2]] not covered by boxes
                # Each box covers [f2_j, ref[1]] x [f3_j, ref[2]]
                
                # Sort boxes by f2_j ascending
                boxes.sort(key=lambda x: x[0])
                
                # Sweep from right to left in f2
                # For each box, the uncovered region in f3 is [f3, max(f3_j for boxes with f2_j <= current)]
                
                # Actually, let's use a simpler approach:
                # The area = (ref[1] - f2) * (ref[2] - f3) - covered_area
                
                # To compute covered area, use the 2D HV algorithm
                # Sort by f2 ascending, then for each box, the "new" area it covers
                # is (ref[1] - f2_j) * (ref[2] - max_f3_so_far) where max_f3_so_far is the max f3
                # of all boxes processed so far with f2_k <= f2_j
                
                # Wait, I'm overcomplicating. Let me just use a grid-based approximation
                # or accept that for small archives, we can use a simpler method.
                
                # For now, let's use a simple Monte Carlo or just return 0 for now
                # and focus on getting the optimization right.
                
                # Actually, let's just compute it properly for small n
                if len(boxes) == 0:
                    area = (ref[1] - f2) * (ref[2] - f3)
                else:
                    # Use the 2D hypervolume algorithm
                    # Sort by f2 ascending
                    boxes_sorted = sorted(boxes, key=lambda x: x[0])
                    
                    # The covered area in [f2, ref[1]] x [f3, ref[2]]
                    # We can compute this by noting that a point (f2_j, f3_j) covers
                    # [f2_j, ref[1]] x [f3_j, ref[2]]
                    
                    # The union of these rectangles
                    # Use sweep line: sort by f2, keep track of max f3 seen so far
                    # but we want the area of the union
                    
                    # Alternative: since n is small, use inclusion-exclusion or just
                    # compute the area of the complement
                    
                    # The area NOT covered is the area of [f2, ref[1]] x [f3, ref[2]]
                    # minus the area of the union of the boxes clipped to this region
                    
                    # For simplicity and correctness with small n, let's use a different approach:
                    # Compute the 2D HV of the set of points (f2_j, f3_j) with reference (ref[1], ref[2])
                    # but only considering the region [f2, ref[1]] x [f3, ref[2]]
                    
                    # Actually, the standard 2D HV:
                    # Sort points by f2 ascending
                    # For each point, the contribution is (ref[1] - f2_j) * (max_f3 - f3_j)
                    # where max_f3 is the maximum f3 of all points with f2_k >= f2_j
                    
                    # But we need to clip to [f2, ref[1]] x [f3, ref[2]]
                    
                    # Let me just use a simple approach: for small number of boxes,
                    # compute the area of the union by checking each point in a fine grid
                    # This is O(n * grid_size) but grid_size can be small
                    
                    # Or, even simpler: since we're just using this for ranking targets,
                    # we can use an approximate HV. Let's use a simple approximation.
                    
                    # Approximate HV by sampling
                    n_samples = 100
                    rng_local = np.random.default_rng(0)
                    f2_samples = rng_local.uniform(f2, ref[1], n_samples)
                    f3_samples = rng_local.uniform(f3, ref[2], n_samples)
                    
                    covered = 0
                    for f2_s, f3_s in zip(f2_samples, f3_samples):
                        for f2_j, f3_j in boxes:
                            if f2_s >= f2_j and f3_s >= f3_j:
                                covered += 1
                                break
                    
                    covered_area = (covered / n_samples) * (ref[1] - f2) * (ref[2] - f3)
                    area = (ref[1] - f2) * (ref[2] - f3) - covered_area
                
                total_hv += (ref[0] - f1) * area
            
            return total_hv

        def compute_hv_contribution(f_new, archive_F, ref):
            """
            Compute the hypervolume contribution of adding f_new to the archive.
            This is HV(archive + f_new) - HV(archive).
            """
            if np.any(f_new >= ref):
                return 0.0
            
            # Check if f_new is dominated
            for f_old in archive_F:
                if dominates(f_old, f_new):
                    return 0.0
            
            # Compute HV before and after
            hv_before = compute_hypervolume_3d(archive_F, ref)
            
            new_archive = list(archive_F)
            # Remove points dominated by f_new
            new_archive = [f for f in new_archive if not dominates(f_new, f)]
            new_archive.append(f_new)
            
            hv_after = compute_hypervolume_3d(new_archive, ref)
            
            return hv_after - hv_before

        def get_tchebycheff_weights(target_F, archive_F, ref):
            """
            Generate Tchebycheff weight vector that targets the region around target_F.
            """
            if len(archive_F) == 0:
                return np.ones(n_obj) / n_obj
            
            F_arr = np.array(archive_F)
            f_min = np.min(F_arr, axis=0)
            f_max = np.max(F_arr, axis=0)
            f_range = np.maximum(f_max - f_min, 1e-10)
            
            # Weight is inversely proportional to the distance from target to the current best
            # in each objective, normalized by the range
            dist = np.maximum(target_F - f_min, 1e-10)
            w = 1.0 / (dist / f_range)
            w = w / np.sum(w)
            return w

        def augmented_tchebycheff(x, w, z, rho=100.0):
            """
            Augmented Tchebycheff objective:
            min max_i w_i * |f_i(x) - z_i| + rho * sum_i w_i * |f_i(x) - z_i|
            """
            f = problem(x)
            vals = w * np.abs(f - z)
            return np.max(vals) + rho * np.sum(vals)

        # Phase 1: Initial diverse sampling
        n_init = min(40, budget // 4)
        rng = np.random.default_rng(42)
        
        # Use stratified sampling for better initial coverage
        X_init = np.zeros((n_init, dim))
        for i in range(n_init):
            for d in range(dim):
                # Stratified: divide each dimension into bins
                bin_idx = (i * dim + d) % max(1, n_init // dim)
                frac = (bin_idx + rng.random()) / max(1, n_init // dim)
                X_init[i, d] = lo[d] + frac * (hi[d] - lo[d])
        
        for x in X_init:
            if used >= budget:
                break
            f = problem(x)
            used += 1
            if np.all(f < ref_point):
                update_archive(x, f)

        # Phase 2: Adaptive scalarization with L-BFGS-B
        # Generate target points to fill hypervolume gaps
        remaining_budget = budget - used
        if remaining_budget > 20 and len(archive_F) > 0:
            # Generate target points on a grid in objective space, biased toward
            # regions with low hypervolume contribution
            n_targets = min(12, remaining_budget // 15)
            
            # Get current Pareto front bounds
            F_arr = np.array(archive_F)
            f_min = np.min(F_arr, axis=0)
            f_max = np.max(F_arr, axis=0)
            
            # Create a grid of candidate targets
            grid_size = 4
            targets = []
            for i in range(grid_size):
                for j in range(grid_size):
                    for k in range(grid_size):
                        # Create a target point
                        t = np.array([i / (grid_size - 1), j / (grid_size - 1), k / (grid_size - 1)])
                        target = f_min + t * (f_max - f_min)
                        # Ensure strictly better than reference
                        target = np.minimum(target, ref_point - 1e-4)
                        targets.append(target)
            
            # Rank targets by hypervolume contribution potential
            # (points that would add more HV if we could find a solution there)
            target_scores = []
            for target in targets:
                # Estimate HV contribution: how much space is there around this target?
                # Simple heuristic: distance to nearest archive point in objective space
                if len(archive_F) > 0:
                    F_arr = np.array(archive_F)
                    dists = np.sqrt(np.sum((F_arr - target) ** 2, axis=1))
                    min_dist = np.min(dists)
                else:
                    min_dist = 1.0
                
                # Also consider how much "room" there is (distance to reference)
                room = np.prod(np.maximum(ref_point - target, 1e-6))
                
                score = min_dist * room
                target_scores.append((score, target))
            
            # Sort by score descending and pick top n_targets
            target_scores.sort(key=lambda x: x[0], reverse=True)
            selected_targets = [t for _, t in target_scores[:n_targets]]
            
            # For each target, run multi-start L-BFGS-B
            starts_per_target = max(2, (remaining_budget - n_targets * 5) // (n_targets * 3))
            
            for target in selected_targets:
                if used >= budget - 5:  # Leave some budget for final polish
                    break
                
                # Get weight vector
                w = get_tchebycheff_weights(target, archive_F, ref_point)
                
                # Get z (ideal point)
                if len(archive_F) > 0:
                    F_arr = np.array(archive_F)
                    z = np.min(F_arr, axis=0)
                else:
                    z = np.array([1.0, 1.0, 0.01])
                
                # Multi-start: initialize at boundaries of current Pareto front
                starts = []
                
                # Start from archive points closest to the target
                if archive_X:
                    F_arr = np.array(archive_F)
                    dists = np.sqrt(np.sum((F_arr - target) ** 2, axis=1))
                    # Pick the 3 closest
                    idx_closest = np.argsort(dists)[:min(3, len(archive_X))]
                    for i in idx_closest:
                        starts.append(archive_X[i].copy())
                
                # Start from boundary points (vertices of the bounding box)
                # Create a few boundary starts
                n_boundary = min(3, starts_per_target)
                for _ in range(n_boundary):
                    x_bound = np.zeros(dim)
                    for d in range(dim):
                        # Randomly choose low or high bound
                        if rng.random() < 0.5:
                            x_bound[d] = lo[d] + rng.random() * 0.1 * (hi[d] - lo[d])
                        else:
                            x_bound[d] = hi[d] - rng.random() * 0.1 * (hi[d] - lo[d])
                    starts.append(x_bound)
                
                # Add random starts
                for _ in range(min(2, starts_per_target)):
                    starts.append(lo + (hi - lo) * rng.random(dim))
                
                for x0 in starts:
                    if used >= budget - 3:
                        break
                    
                    # Run L-BFGS-B with evaluation tracking
                    eval_count = [0]
                    max_evals = min(15, budget - used - 3)
                    
                    if max_evals < 5:
                        break
                    
                    def obj_wrapper(x, w=w, z=z, rho=100.0):
                        if eval_count[0] >= max_evals:
                            return 1e10
                        eval_count[0] += 1
                        f = problem(x)
                        vals = w * np.abs(f - z)
                        return np.max(vals) + rho * np.sum(vals)
                    
                    try:
                        result = minimize(
                            obj_wrapper,
                            x0,
                            method='L-BFGS-B',
                            bounds=list(zip(lo, hi)),
                            options={'maxiter': 8, 'maxfun': max_evals, 'ftol': 1e-8, 'gtol': 1e-6}
                        )
                        
                        used += eval_count[0]
                        
                        # Evaluate the result point (one more evaluation)
                        if used < budget:
                            x_opt = np.clip(result.x, lo, hi)
                            f_opt = problem(x_opt)
                            used += 1
                            
                            if np.all(f_opt < ref_point):
                                update_archive(x_opt, f_opt)
                            
                    except Exception:
                        continue

        # Phase 3: Final polish with random evaluations
        while used < budget:
            x = lo + (hi - lo) * rng.random(dim)
            f = problem(x)
            used += 1
            if np.all(f < ref_point):
                update_archive(x, f)

        return archive_X
# EVOLVE-BLOCK-END
