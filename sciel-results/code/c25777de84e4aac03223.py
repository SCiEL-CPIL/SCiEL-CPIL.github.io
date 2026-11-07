"""Multi-objective optimizer on RE34: Random Forest Surrogate with Hypervolume-Weighted Uncertainty."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import random

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = int(budget)
        self.dim = int(dim)
        self.n_obj = int(n_obj)

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 295  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        used = 0
        X_list = []
        F_list = []

        def evaluate(x):
            nonlocal used
            x = np.clip(x, lo, hi)
            f = np.asarray(problem(x), dtype=np.float64)
            used += 1
            return x, f

        # Use a fixed seed for reproducibility within the run, 
        # but distinct from the global seed to ensure diversity in RF initialization
        # Note: The task says evaluator calls np.random.seed(seed) before creating Optimizer.
        # We should use that global state or a derived one. 
        # Using np.random directly allows us to inherit the evaluator's seed.
        
        # Reference point for hypervolume
        ref_point = np.array([1705.0, 12.0, 0.3])

        # 1. Initialization: Space-filling design
        # Use a stratified random sampling or simple uniform random for first 20 points
        n_init = 20
        for _ in range(n_init):
            x = np.random.uniform(lo, hi)
            x, f = evaluate(x)
            X_list.append(x)
            F_list.append(f)
        
        X = np.array(X_list)
        F = np.array(F_list)

        # Helper: Check if a point dominates another
        def dominates(f1, f2):
            return np.all(f1 <= f2) and np.any(f1 < f2)

        # Helper: Get non-dominated points
        def get_non_dominated(F):
            n = len(F)
            is_nd = np.ones(n, dtype=bool)
            for i in range(n):
                if not is_nd[i]:
                    continue
                for j in range(n):
                    if i == j or not is_nd[j]:
                        continue
                    if dominates(F[j], F[i]):
                        is_nd[i] = False
                        break
            return np.where(is_nd)[0]

        def calc_hv_3d(points, ref):
            """
            Exact hypervolume for 3 objectives.
            points: (N, 3) array of non-dominated points.
            ref: (3,) array.
            """
            if len(points) == 0:
                return 0.0
            
            # Filter valid points
            valid = points[np.all(points < ref, axis=1)]
            if len(valid) == 0:
                return 0.0
            
            # Sort by objective 0
            valid = valid[np.argsort(valid[:, 0])]
            
            hv = 0.0
            # For 3D, we can use a sweep line or simple decomposition.
            # Given small N (<= 200), O(N^2) or O(N^3) is fine.
            # Simple decomposition:
            # Sort by f0.
            # For each point i, the unique contribution is the box defined by:
            # [f0_i, f0_{i+1}] x [f1_i, min(f1_j for j>i where f0_j > f0_i and f1_j < f1_i? No, complex)]
            # Let's use a simpler O(N^2) approach for 3D:
            # Sort by f0.
            # Maintain a list of "active" segments in the 2D plane (f1, f2).
            # Actually, for N<=200, even O(N^3) is < 10^7 ops, which is fast in Python if vectorized or simple loops.
            
            # A robust simple method for 3D:
            # Sort by f0 ascending.
            # For each point i, it contributes volume between f0_i and f0_{i+1} (or ref[0] if last).
            # The cross-section at f0_i is the area in (f1, f2) plane dominated by point i and not by any previous points (with smaller f0).
            # Since points are non-dominated and sorted by f0, for a given f0, the "skyline" in (f1, f2) is determined by the points with f0 <= current f0.
            
            # Let's just use a standard 3D HV algorithm.
            # Sort by f0.
            # prev_f1 = ref[1]
            # prev_f2 = ref[2]
            # This specific decomposition works if we sweep f0 and maintain the "max" f1 and f2? No.
            
            # Let's use the "inclusion-exclusion" or simple box subtraction for small N.
            # For each point p, vol = prod(ref - p).
            # Subtract overlap with points that dominate it? No, p is non-dominated.
            # Subtract overlap with points that are "below" it in the sense of the box.
            
            # Efficient 3D HV:
            # Sort by f0.
            # Keep track of the "Pareto front" in the (f1, f2) plane for the points processed so far.
            # Actually, let's just iterate.
            sorted_idx = np.argsort(valid[:, 0])
            sorted_valid = valid[sorted_idx]
            
            hv = 0.0
            # We maintain a list of "active" points that form the skyline in f1-f2 for the current f0 range.
            # This is getting complex. Let's use a simpler O(N^2) heuristic that is exact for 3D non-dominated sets:
            # Sort by f0.
            # For i in range(N):
            #   f0_range = min(f0_{i+1}, ref[0]) - f0_i
            #   Area in (f1, f2) plane:
            #   The region is [f1_i, ref[1]] x [f2_i, ref[2]] MINUS regions covered by points j < i that have f1_j < f1_i and f2_j < f2_i?
            #   No, points j < i have smaller f0. They dominate the region [0, f0_j]...
            #   Actually, the unique contribution of point i (sorted by f0 asc) is:
            #   Width in f0: (next_f0 or ref[0]) - f0_i
            #   Area in (f1, f2): The area of the box [f1_i, ref[1]] x [f2_i, ref[2]] that is NOT covered by any point j < i.
            #   Since j < i has f0_j < f0_i, it does NOT cover any volume in the f0 slice [f0_i, next_f0].
            #   Wait, the volume is 3D. The box for point i is [f0_i, ref[0]] x [f1_i, ref[1]] x [f2_i, ref[2]].
            #   Points j < i have boxes [f0_j, ref[0]] x ...
            #   The intersection of box i and box j is [max(f0_i, f0_j), min(ref[0], ref[0])] ...
            #   Since f0_j < f0_i, the intersection in f0 is [f0_i, ref[0]].
            #   So we must subtract the volume of the intersection in (f1, f2) for all j < i where the boxes overlap in f1 and f2.
            
            # This is effectively:
            # For each i:
            #   f0_len = (sorted_valid[i+1, 0] if i < N-1 else ref[0]) - sorted_valid[i, 0]
            #   area = 0
            #   # Calculate area of [f1_i, ref[1]] x [f2_i, ref[2]] minus overlaps with previous points' (f1, f2) boxes?
            #   # No, the previous points' boxes extend to ref[0] in f0. So they overlap in the full f0_len range.
            #   # So we need the area in (f1, f2) plane of the union of boxes from j < i that overlap with i's box?
            #   # Actually, it's easier to calculate the "skyline" area.
            
            # Let's use a known simple O(N^2) for 3D:
            # Sort by f0.
            # Maintain a list of "active" points in the (f1, f2) plane.
            # This is too complex to implement correctly in a short time.
            
            # Alternative: Use a simple Monte Carlo estimate? No, we need exact.
            # Alternative: Use the fact that N is small.
            # For each point i, calculate the volume of the box [f_i, ref].
            # Subtract the volume of the intersection with the union of boxes of all other points.
            # This is hard.
            
            # Let's use a library if available? No, stdlib/numpy/scipy only.
            # Let's use a simple recursive decomposition or just a decent approximation if exact is too hard?
            # The scoring uses "exact hypervolume". I should try to get it exact.
            
            # Here is a working O(N^2) for 3D:
            # Sort by f0.
            # hv = 0
            # prev_f1 = ref[1]
            # prev_f2 = ref[2]
            # This only works for 2D.
            
            # For 3D, we can use the following:
            # Sort by f0.
            # For each point i, the unique contribution is:
            # (f0_{i+1} - f0_i) * Area_2D( [f1_i, ref[1]] x [f2_i, ref[2]] \ Union_{j<i} [f1_j, ref[1]] x [f2_j, ref[2]] )
            # Wait, the union is not quite right because the boxes extend to ref[0].
            # Yes, since all boxes extend to ref[0] in the f0 dimension, the overlap in f0 is the full length of the slice for point i.
            # So we just need the 2D area of the box of i in (f1, f2) that is not covered by the boxes of j < i in (f1, f2).
            
            # To calculate this area efficiently:
            # We maintain a set of "active" rectangles in (f1, f2) from previous points.
            # Since N is small, we can just compute the area of the box of i, and subtract the area of intersection with each previous box j, and add back intersections of intersections... (Inclusion-Exclusion).
            # With N=200, Inclusion-Exclusion is impossible (2^200).
            
            # However, the "active" rectangles in (f1, f2) for non-dominated points sorted by f0 have a special structure.
            # They form a "staircase" or "skyline" in the (f1, f2) plane?
            # If points are non-dominated in 3D, their projections on (f1, f2) are not necessarily non-dominated.
            # Example: (0, 10, 10) and (1, 11, 9). Both non-dominated in 3D.
            # Box 1: [10, ref1]x[10, ref2]. Box 2: [11, ref1]x[9, ref2].
            # Overlap: [11, ref1]x[10, ref2].
            
            # Let's just use a simple O(N^3) method which is fine for N=200 (8 million ops).
            # For each point i:
            #   f0_slice = (next_f0 or ref[0]) - f0_i
            #   # Calculate 2D area of box_i minus union of box_j (j<i)
            #   # We can do this by discretizing? No.
            #   # We can calculate the area of box_i, and subtract the area of the intersection of box_i with the UNION of previous boxes.
            #   # Union of rectangles area can be calculated in O(M^2) for M rectangles.
            #   # Total O(N^3).
            
            def rect_area_intersections(rects):
                """Calculate area of union of rectangles (list of (x1, y1, x2, y2) where x1<x2, y1<y2)."""
                if not rects:
                    return 0.0
                # Sweep line or simple inclusion-exclusion for small M?
                # For M up to 200, O(M^2) sweep line is good.
                # But simpler: just sum areas and subtract pairwise? No, that's not union.
                # Let's use a simple grid-based approximation? No, exact required.
                # Let's use the "shoelace" like approach for union of rectangles?
                # Actually, for M <= 200, we can use a simple O(M^2) algorithm for union of rectangles.
                
                # Flatten all x coordinates
                xs = sorted(set([r[0] for r in rects] + [r[2] for r in rects]))
                # For each strip [xs[i], xs[i+1]], calculate the height covered by rectangles
                total_area = 0.0
                for k in range(len(xs) - 1):
                    x_left = xs[k]
                    x_right = xs[k+1]
                    if x_right <= x_left:
                        continue
                    # Find all rectangles that cover this x-range
                    # A rect covers [x_left, x_right] if rect.x1 <= x_left and rect.x2 >= x_right
                    # Since xs are from rect boundaries, any rect that overlaps this strip must contain it?
                    # No, a rect might start inside the strip. But xs are all boundaries.
                    # So if a rect overlaps [x_left, x_right], it must be that rect.x1 <= x_left and rect.x2 >= x_right?
                    # Yes, because if rect.x1 > x_left, then rect.x1 would be in xs and define a boundary.
                    # So we just check which rects have x1 <= x_left and x2 >= x_right.
                    
                    y_intervals = []
                    for r in rects:
                        if r[0] <= x_left and r[2] >= x_right:
                            y_intervals.append((r[1], r[3]))
                    
                    if not y_intervals:
                        continue
                    
                    # Calculate length of union of y_intervals
                    y_intervals.sort()
                    current_max_y = -np.inf
                    union_len = 0.0
                    for y1, y2 in y_intervals:
                        if y1 > current_max_y:
                            union_len += (y2 - y1)
                            current_max_y = y2
                        else:
                            if y2 > current_max_y:
                                union_len += (y2 - current_max_y)
                                current_max_y = y2
                    
                    total_area += (x_right - x_left) * union_len
                
                return total_area

            N_valid = len(sorted_valid)
            for i in range(N_valid):
                f0_i = sorted_valid[i, 0]
                f1_i = sorted_valid[i, 1]
                f2_i = sorted_valid[i, 2]
                
                next_f0 = sorted_valid[i+1, 0] if i < N_valid - 1 else ref[0]
                f0_len = next_f0 - f0_i
                if f0_len <= 0:
                    continue
                
                # Box i in (f1, f2): [f1_i, ref[1]] x [f2_i, ref[2]]
                box_i = (f1_i, f2_i, ref[1], ref[2])
                
                # Previous boxes:
                prev_boxes = []
                for j in range(i):
                    # Box j in (f1, f2): [f1_j, ref[1]] x [f2_j, ref[2]]
                    prev_boxes.append((sorted_valid[j, 1], sorted_valid[j, 2], ref[1], ref[2]))
                
                # Area of box_i
                area_i = (ref[1] - f1_i) * (ref[2] - f2_i)
                
                # Area of union of (box_i intersect prev_boxes)
                # This is the area of box_i that is covered by any previous box.
                # Intersection of box_i and box_j is [max(f1_i, f1_j), ref[1]] x [max(f2_i, f2_j), ref[2]]
                # We want the area of box_i MINUS the area of (box_i ∩ Union(prev_boxes)).
                # This is equivalent to the area of box_i that is NOT covered by any prev_box.
                
                # Let's calculate the area of the union of intersections.
                # Intersected rects:
                inter_rects = []
                for pb in prev_boxes:
                    # pb = (f1_j, f2_j, ref[1], ref[2])
                    # box_i = (f1_i, f2_i, ref[1], ref[2])
                    # Intersection:
                    x1 = max(f1_i, pb[0])
                    y1 = max(f2_i, pb[1])
                    x2 = ref[1]
                    y2 = ref[2]
                    if x1 < x2 and y1 < y2:
                        inter_rects.append((x1, y1, x2, y2))
                
                if not inter_rects:
                    unique_area = area_i
                else:
                    covered_area = rect_area_intersections(inter_rects)
                    unique_area = area_i - covered_area
                
                hv += f0_len * unique_area
            
            return hv

        # Main Loop
        n_trees = 50
        n_cand = 100
        
        # Train initial RF
        # RFs for each objective
        rfs = []
        for k in range(n_obj):
            rf = RandomForestRegressor(n_estimators=n_trees, random_state=42+k, n_jobs=-1)
            rf.fit(X, F[:, k])
            rfs.append(rf)
        
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate candidates
            # Mix of random and local search around best points
            X_cand = np.random.uniform(lo, hi, size=(n_cand, dim))
            
            # Predict
            mu = np.zeros((n_cand, n_obj))
            std = np.zeros((n_cand, n_obj))
            for k in range(n_obj):
                # RF predict returns (n_cand, 1) or (n_cand,)
                pred = rfs[k].predict(X_cand)
                pred = np.asarray(pred).flatten()
                mu[:, k] = pred
                # Estimate uncertainty: std of predictions from individual trees
                # This is expensive. Alternative: use the OOB score or just assume fixed uncertainty?
                # Better: Use the variance of the trees.
                # To save time, we can approximate uncertainty by the distance to nearest neighbors?
                # Or just use a simple heuristic: uncertainty is higher in sparse regions.
                # Let's use the "max distance to k-nearest neighbors" as a proxy for uncertainty.
                # This is fast and correlates with exploration.
                
            # Calculate simple HV contribution estimate for each candidate
            # We need to check if the candidate would improve the HV.
            # Approximation: Calculate HV of (Current Front + Candidate) - HV(Current Front).
            # This is expensive to do for all candidates.
            
            # Alternative Acquisition:
            # Score = mu_hv_potential + beta * uncertainty
            # mu_hv_potential: How good is the point in terms of objectives?
            # Let's use a "Weighted Sum" or "Hypervolume Indicator" approximation.
            
            # Let's use a simpler, effective strategy:
            # 1. Identify the current non-dominated front.
            # 2. For each candidate, calculate the "Potential HV Contribution" assuming it is the true value.
            # 3. Add an exploration bonus based on distance to existing points (KDE or kNN).
            
            nd_idx = get_non_dominated(F)
            F_nd = F[nd_idx]
            current_hv = calc_hv_3d(F_nd, ref_point)
            
            scores = np.zeros(n_cand)
            
            # Precompute distances for uncertainty
            # Use kNN with k=5
            from sklearn.neighbors import KNeighborsRegressor
            # Reusing a simple distance metric
            # For speed, let's just use random uniform sampling and pick the one with best "Optimistic HV"
            # Optimistic HV: Assume the candidate has value = mu - k*std (best case).
            
            # Let's calculate the "Optimistic" objective values
            # We need std. Let's compute it properly for the RF.
            # It might be slow, but N_cand=100, N_train=200, n_trees=50.
            # 100 * 50 * 200 = 1M ops per objective. 3 objectives = 3M ops. Fast enough.
            
            for k in range(n_obj):
                # Get predictions from each tree
                trees = rfs[k].estimators_
                preds = np.array([tree.predict(X_cand).flatten() for tree in trees])
                std[:, k] = np.std(preds, axis=0)
            
            # Exploration factor
            # Distance to nearest neighbor in X space
            # Normalize X
            X_norm = (X - lo) / (hi - lo)
            X_cand_norm = (X_cand - lo) / (hi - lo)
            
            # Simple kNN using cdist
            from scipy.spatial.distance import cdist
            dists = cdist(X_cand_norm, X_norm, metric='euclidean')
            min_dists = np.min(dists, axis=1)
            
            # Normalize distances
            max_dist = np.sqrt(dim)
            min_dists_norm = min_dists / max_dist
            
            # Acquisition: 
            # We want to maximize HV.
            # Let's try a candidate by adding it to the front and calculating HV.
            # To make it fast, we only do this for the top 10 candidates based on a cheap proxy.
            
            # Cheap proxy: Min of objectives (normalized)
            F_norm = (F - lo_f) / (hi_f - lo_f) # Need to normalize F?
            # Let's just use the raw values.
            
            # Proxy Score: Sum of (best_observed - mu) / (range)
            # This favors points that are better than current best.
            
            # Let's just pick the candidate that maximizes:
            # HV( Front ∪ {candidate} ) 
            # But we do it for only top 20 candidates based on "min objective" or "distance".
            
            # Sort candidates by "optimism": mu - 1.5 * std
            optimistic_F = mu - 1.5 * std
            # Filter out those worse than ref
            valid_mask = np.all(optimistic_F < ref_point, axis=1)
            if not np.any(valid_mask):
                # If none are valid, pick the one with smallest distance to current front
                scores = min_dists_norm
                idx = np.argmax(scores)
            else:
                # For top 20 candidates by "optimistic sum"
                opt_sum = np.sum(optimistic_F, axis=1)
                # Invert because we minimize
                opt_rank = np.argsort(opt_sum)
                top_candidates = opt_rank[:20]
                
                best_score = -np.inf
                best_idx = top_candidates[0]
                
                for idx in top_candidates:
                    f_cand = optimistic_F[idx]
                    # Add to front
                    F_new = np.vstack([F_nd, f_cand])
                    # Get non-dominated of new set
                    nd_new_idx = get_non_dominated(F_new)
                    F_new_nd = F_new[nd_new_idx]
                    hv_new = calc_hv_3d(F_new_nd, ref_point)
                    score = hv_new - current_hv
                    
                    # Add small exploration bonus
                    score += 0.01 * min_dists_norm[idx]
                    
                    if score > best_score:
                        best_score = score
                        best_idx = idx
            
            x_new = X_cand[best_idx]
            x_new, f_new = evaluate(x_new)
            
            X = np.vstack([X, x_new])
            F = np.vstack([F, f_new])
            
            # Retrain RF every 5 evaluations
            if used % 5 == 0:
                for k in range(n_obj):
                    rfs[k].fit(X, F[:, k])

        return X
# EVOLVE-BLOCK-END
