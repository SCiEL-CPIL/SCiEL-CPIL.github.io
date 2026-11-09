import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        # Safety margin: 290s out of 300s limit
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        n_obj = self.n_obj
        
        # Reference point for hypervolume calculation (from task description)
        ref_point = np.array([3.0, 5.0, 7.0])
        
        # 1. Initialize CMA-ES parameters
        # Start population in the center of the bounds
        x0 = (lo + hi) / 2.0
        # Initial sigma: 10% of the range of each dimension
        sigma = 0.1 * (hi - lo)
        
        # Population size
        # Standard CMA-ES uses lambda = 4 + floor(3 * ln(n))
        # For 12 dims, ln(12) ~ 2.5, so lambda ~ 11.5 -> 12
        # We can go slightly larger for multi-objective diversity, e.g., 20
        lambda_pop = max(12, 4 + int(3 * np.log(dim)))
        
        # We have a fixed budget of 200 evaluations.
        # Generations = budget // lambda_pop
        # If lambda_pop = 20, generations = 10.
        
        # Archive of non-dominated points
        archive = []
        
        # Helper to calculate hypervolume contribution in 3D
        def hypervolume_3d(points, ref):
            # points: Nx3 array
            # ref: 3x1 array
            # Returns total hypervolume dominated by points w.r.t ref
            if len(points) == 0:
                return 0.0
            
            # Sort by first objective
            points = points[np.argsort(points[:, 0])]
            
            # We can use a simple recursive or iterative approach.
            # For small N (population size ~20), O(N^2) or O(N^3) is fine.
            # Let's use a simple box decomposition or inclusion-exclusion.
            # Actually, for 3D, we can sort by x, then project to 2D.
            
            # Sort by x
            pts = points[np.argsort(points[:, 0])]
            
            vol = 0.0
            # Iterate through points sorted by x
            # For each point, calculate the volume of the box it uniquely dominates
            # in the "skyline" sense.
            
            # A more robust simple method for 3D:
            # 1. Sort by f1
            # 2. For each point i, find the "envelope" of f2, f3 for points j > i (worse in f1)
            #    that are dominated by i in f2 and f3? No, that's not quite right for HV.
            
            # Let's stick to a standard efficient 3D HV algorithm.
            # Algorithm:
            # 1. Filter points dominated by ref.
            # 2. Sort by f1.
            # 3. Sweep from best f1 to worst f1.
            #    Maintain the "skyline" of f2, f3 for the points seen so far?
            #    No, standard algorithm:
            #    Sort by f1.
            #    For each point i (from best to worst f1):
            #       Calculate the area in f2-f3 plane dominated by point i, 
            #       excluding areas already dominated by points j with f1_j < f1_i.
            
            # Let's implement a simple O(N^3) check for small N, or a better O(N^2 log N).
            # Given N~20, O(N^3) is 8000 ops, trivial.
            
            # However, we need to be careful with the "contribution".
            # Let's use a recursive partitioning or a simple loop.
            
            # Simple approach:
            # Total HV = sum of contributions.
            # Contribution of point p_i:
            #   Volume of box defined by p_i and ref, MINUS volume of that box dominated by other points.
            # This is hard to compute directly without inclusion-exclusion.
            
            # Let's use the standard sweep line for 3D.
            # Sort by f1.
            # Maintain a list of "active" points in f2-f3 space.
            
            # Actually, for N=20, we can just do:
            # 1. Sort by f1
            # 2. For each point i, determine the "frontier" of f2, f3 values that are NOT dominated by any point j with f1_j < f1_i.
            #    This is complex.
            
            # Let's use a simpler, slightly less efficient but correct method for small N:
            # 1. Generate all unique f1, f2, f3 values.
            # 2. Construct a grid? No, that's overkill.
            
            # Let's use the algorithm from "A simple and efficient algorithm for hypervolume calculation in 3D".
            # Or just use scipy? No, standard lib + numpy.
            
            # Let's try a simple recursive approach or just sort and sweep.
            # Sort by f1.
            # pts = sorted(points, key=lambda p: p[0])
            # vol = 0
            # for i in range(len(pts)):
            #    # Calculate the volume of the box defined by pts[i] and ref
            #    # But subtract the part that is already covered by pts[0..i-1]
            
            # Actually, there is a very simple O(N^2) algorithm for 3D HV:
            # 1. Sort points by f1.
            # 2. For each point i, find the maximum f2 and f3 "envelope" of points j < i? No.
            
            # Let's use this logic:
            # The hypervolume is the volume of the union of boxes B_i = [f1_i, ref1] x [f2_i, ref2] x [f3_i, ref3].
            
            # We can compute this by sorting by f1, and for each point, calculating the area in the (f2, f3) plane that is "newly" covered.
            # The "newly covered" area for point i is the area in [f2_i, ref2] x [f3_i, ref3] that is NOT dominated by any point j with f1_j < f1_i.
            
            # Let's implement this.
            
            pts = points[np.argsort(points[:, 0])]
            
            # active_envelope: list of (f2, f3) points that form the skyline of the "covered" region?
            # No, it's the region of (f2, f3) that is dominated by at least one previous point.
            # A point (f2, f3) is dominated by a previous point j if f2_j <= f2 and f3_j <= f3.
            
            # So for point i, we want the area of the rectangle [f2_i, ref2] x [f3_i, ref3] minus the area dominated by previous points.
            
            # This is equivalent to:
            # Area = (ref2 - f2_i) * (ref3 - f3_i) - Area_Dominated_By_Previous(f2_i, f3_i)
            
            # Where Area_Dominated_By_Previous(f2, f3) is the area of the region in [f2, ref2] x [f3, ref3] dominated by any previous point.
            
            # This is still complex to compute exactly without a data structure.
            
            # Given the small N, let's use a grid-based approximation? No, "exact hypervolume" is requested for scoring, but we only use it for selection.
            # Approximation is fine for selection as long as it's consistent.
            # But let's try to be exact if possible.
            
            # Alternative: Since N is small, we can use a simple recursive subdivision or just sum the boxes and subtract overlaps?
            # Inclusion-exclusion for 20 boxes is 2^20, too slow.
            
            # Let's use a simple O(N^3) check for dominance to find the non-dominated set first, then calculate HV of the non-dominated set.
            # For the non-dominated set, the boxes don't overlap in a "nested" way, but they do overlap.
            
            # Actually, for selection, we just need a score.
            # Let's use a simple heuristic: Sum of (ref - f)^3 for all non-dominated points? No, that's not HV.
            
            # Let's use a simple exact HV calculation for 3D.
            # Reference: "Hypervolume Calculation in 3 Dimensions"
            
            def hv_3d_exact(pts, ref):
                # pts: Nx3
                # ref: 3
                # Sort by f1
                pts = pts[np.argsort(pts[:, 0])]
                n = len(pts)
                if n == 0: return 0.0
                
                # We will iterate from best f1 to worst f1.
                # Maintain a list of "segments" in the f2-f3 plane that are covered.
                # Actually, let's use the property that the HV of a set of points is the sum of the contributions.
                
                # A simpler exact algorithm for 3D:
                # 1. Sort by f1.
                # 2. For each point i, calculate the area in f2-f3 plane that is dominated by point i but not by any point j with f1_j < f1_i.
                
                # Let's maintain a list of points that are "relevant" for the f2-f3 skyline.
                # Actually, the region dominated by previous points in the f2-f3 plane is the union of rectangles [f2_j, ref2] x [f3_j, ref3] for all j < i.
                
                # We can compute the area of a rectangle [a, b] x [c, d] minus the area covered by a set of rectangles.
                
                # Given N=20, we can just check all pairs?
                
                # Let's try a different approach:
                # The HV is the volume of the union of boxes.
                # We can compute this by sorting by f1, and for each slice between f1_{i-1} and f1_i, calculating the area of the cross-section in f2-f3.
                
                # The cross-section at f1 is the union of rectangles [f2_j, ref2] x [f3_j, ref3] for all j such that f1_j <= f1.
                
                # So:
                # 1. Sort points by f1.
                # 2. Create a list of unique f1 values: f1_0 < f1_1 < ... < f1_{n-1} < ref1.
                # 3. For each interval [f1_k, f1_{k+1}], the set of active points is {j | f1_j <= f1_k}.
                #    The area of the cross-section is the area of the union of rectangles R_j = [f2_j, ref2] x [f3_j, ref3] for j in active set.
                #    The volume contribution is Area * (f1_{k+1} - f1_k).
                
                # How to compute Area of Union of Rectangles in 2D for N<=20?
                # We can use a grid of unique f2 and f3 values.
                # Unique f2 values: {f2_1, ..., f2_n, ref2}. Sort them.
                # Unique f3 values: {f3_1, ..., f3_n, ref3}. Sort them.
                # This creates a grid of at most (n+1)^2 cells.
                # For each cell, check if it is covered by any active rectangle.
                # If covered, add cell area to total area.
                
                # This is O(N^3) per slice, and there are N slices. Total O(N^4).
                # For N=20, 160,000 ops. Very fast.
                
                f1_vals = np.unique(np.concatenate([pts[:, 0], [ref[0]]]))
                f2_vals = np.unique(np.concatenate([pts[:, 1], [ref[1]]]))
                f3_vals = np.unique(np.concatenate([pts[:, 2], [ref[2]]]))
                
                # Sort ascending
                f1_vals.sort()
                f2_vals.sort()
                f3_vals.sort()
                
                total_vol = 0.0
                
                # Iterate through f1 intervals
                for i in range(len(f1_vals) - 1):
                    f1_low = f1_vals[i]
                    f1_high = f1_vals[i+1]
                    width = f1_high - f1_low
                    if width <= 0: continue
                    
                    # Active points: those with f1 <= f1_low
                    # Actually, if f1_j is in [f1_low, f1_high), it's active for the whole interval?
                    # Yes, because the rectangle extends from f1_j to ref1.
                    # So if f1_j <= f1_high, it is active in the interval [f1_low, f1_high] ONLY IF f1_j <= f1_high.
                    # But the rectangle starts at f1_j.
                    # So for the interval [f1_low, f1_high], the active points are those with f1_j <= f1_high.
                    # Wait, if f1_j > f1_low, it doesn't cover the part [f1_low, f1_j].
                    # So we should split the interval at every f1_j.
                    # Since we are iterating over unique f1 values from the points, the intervals are exactly between consecutive f1 values.
                    # So for interval [f1_k, f1_{k+1}], the active points are those with f1_j <= f1_k?
                    # No. The rectangle for point j covers [f1_j, ref1].
                    # So in the interval [f1_k, f1_{k+1}], point j is active if f1_j <= f1_k.
                    # Because if f1_j > f1_k, it starts later.
                    
                    active_mask = pts[:, 0] <= f1_low + 1e-9 # Use f1_low as the representative of the start of the interval
                    
                    # Actually, let's be precise.
                    # The interval is [f1_vals[i], f1_vals[i+1]].
                    # A point j is active in this interval if f1_j <= f1_vals[i].
                    # Because if f1_j is in (f1_vals[i], f1_vals[i+1]), it is not in our unique list?
                    # f1_vals contains all point f1s. So f1_j is one of the boundaries.
                    # So for the interval [f1_k, f1_{k+1}], the points with f1_j = f1_k are active.
                    # Points with f1_j < f1_k are also active.
                    # So active points are those with f1_j <= f1_k.
                    
                    active_pts = pts[pts[:, 0] <= f1_low + 1e-9]
                    
                    if len(active_pts) == 0:
                        continue
                        
                    # Calculate area of union of rectangles in f2-f3 plane for active_pts
                    # Rectangles are [f2_j, ref2] x [f3_j, ref3]
                    
                    # Grid method
                    # f2_grid = f2_vals
                    # f3_grid = f3_vals
                    
                    area = 0.0
                    # Iterate over cells in f2-f3 grid
                    for u in range(len(f2_vals) - 1):
                        f2_lo = f2_vals[u]
                        f2_hi = f2_vals[u+1]
                        if f2_lo >= ref[1]: continue
                        # Cell width in f2
                        w2 = min(f2_hi, ref[1]) - f2_lo
                        if w2 <= 0: continue
                        
                        for v in range(len(f3_vals) - 1):
                            f3_lo = f3_vals[v]
                            f3_hi = f3_vals[v+1]
                            if f3_lo >= ref[2]: continue
                            # Cell height in f3
                            h3 = min(f3_hi, ref[2]) - f3_lo
                            if h3 <= 0: continue
                            
                            # Check if this cell is covered by any active rectangle
                            # A cell [f2_lo, f2_hi] x [f3_lo, f3_hi] is covered by rect j
                            # if f2_j <= f2_lo and f3_j <= f3_lo.
                            # (Since the rect extends to ref, which is >= f2_hi, f3_hi)
                            
                            covered = False
                            for p in active_pts:
                                if p[1] <= f2_lo + 1e-9 and p[2] <= f3_lo + 1e-9:
                                    covered = True
                                    break
                            
                            if covered:
                                area += w2 * h3
                    
                    total_vol += area * width
                
                return total_vol
            
            # Main Loop
            while self.budget > 0:
                # 1. Sample population
                # CMA-ES sampling: x = mean + sigma * z
                # But we need to handle bounds.
                # Simple approach: Sample in unconstrained space, clip to bounds.
                # This can cause issues if the front is at the boundary.
                # Better: Sample in the box [lo, hi] directly?
                # CMA-ES assumes Gaussian. Clipping a Gaussian is okay for now.
                
                # Generate lambda samples
                Z = np.random.randn(lambda_pop, dim)
                # Standard CMA-ES uses a step size.
                # Let's use the current sigma.
                # However, since we are clipping, the effective step size changes.
                
                # To improve efficiency, let's adapt sigma based on the spread of the population?
                # Or just keep it fixed?
                # Fixed sigma might be too large or too small.
                # Let's try to adapt sigma.
                # A simple adaptation: if the population spread is small, decrease sigma.
                
                # For simplicity and robustness, let's use a fixed sigma that is reasonable.
                # Or, better, use the bounds to define the initial sigma and keep it constant?
                # If sigma is constant, it's just a random search with a Gaussian kernel.
                # That's not CMA-ES.
                
                # Let's implement a basic CMA-ES adaptation.
                # 1. Sample
                # 2. Evaluate
                # 3. Select best mu points based on HV
                # 4. Update mean to weighted average of selected points
                # 5. Update covariance matrix? (Too complex for this time)
                #    Instead, let's just update the mean and keep a fixed covariance (diagonal)?
                #    That's a variant.
                
                # Let's stick to a simpler "Mean-Shift" approach with a fixed step size, but adapt the step size.
                
                # Actually, let's use a standard CMA-ES implementation logic but simplified.
                # We will maintain a mean `m` and a step size `sigma`.
                # We will NOT update the covariance matrix (C) to keep it simple, just use diagonal?
                # No, let's just use the full covariance if possible?
                # With N=20 and dim=12, we can't estimate a 12x12 covariance well.
                # So let's use a diagonal covariance (independent step sizes per dimension).
                
                # Initialize diagonal step sizes
                # We already have sigma as a vector.
                
                # Sample
                samples = m + sigma * Z
                # Clip to bounds
                samples = np.clip(samples, lo, hi)
                
                # Evaluate
                fvals = np.zeros((lambda_pop, n_obj))
                for i in range(lambda_pop):
                    if self.budget <= 0 or time.time() > deadline:
                        break
                    fvals[i] = problem(samples[i])
                    self.budget -= 1
                    
                    # Add to archive if non-dominated?
                    # Or just keep all and filter later?
                    # Let's keep all evaluated points in a list for now.
                    # Actually, we should add to archive if it's non-dominated by current archive.
                    # But checking dominance against a growing archive is O(N_archive).
                    # Let's just store all points and filter at the end?
                    # No, we need to select the best for the next generation.
                    # So we need the HV of the current population.
                    
                if self.budget <= 0 or time.time() > deadline:
                    break
                
                # 2. Calculate HV for each point in the population
                # We want to select the top `mu` points based on their contribution to the HV.
                # A common proxy is the HV of the set of points {p_i, p_1, ..., p_{i-1}} minus HV of {p_1, ..., p_{i-1}}?
                # That's expensive.
                
                # Simpler: Sort by a simple fitness, e.g., sum of normalized objectives?
                # Or just use the HV of the whole population and assign credit?
                
                # Let's use a simple dominance-based selection.
                # Find the non-dominated set in the population.
                # If the non-dominated set is smaller than mu, fill with dominated points that are close to the front?
                
                # For now, let's just select the `mu` points with the lowest sum of objectives (normalized).
                # This is a simple surrogate.
                
                # Normalize objectives to [0, 1] using the current population min/max?
                # Or using the ref point?
                # Let's use the ref point.
                # f_norm = f / ref_point
                # Sum = sum(f_norm)
                
                f_norm = fvals / ref_point
                scores = np.sum(f_norm, axis=1)
                
                # Select best mu
                mu = max(1, lambda_pop // 2)
                selected_indices = np.argsort(scores)[:mu]
                selected_pts = samples[selected_indices]
                selected_fvals = fvals[selected_indices]
                
                # Update mean
                # Weighted average? Or simple average?
                # Simple average is fine for a basic version.
                m = np.mean(selected_pts, axis=0)
                
                # Update sigma?
                # If the selected points are close to each other, decrease sigma.
                # If they are spread out, increase sigma?
                # Let's keep sigma constant for now to avoid instability.
                # Or decrease it slowly.
                sigma *= 0.95
                
                # Add selected points to archive
                for i in range(mu):
                    p = selected_pts[i]
                    f = selected_fvals[i]
                    # Check if f is dominated by any point in archive
                    # And if p is not already in archive (approximate check)
                    dominated = False
                    for af in archive:
                        if np.all(af <= f) and np.any(af < f):
                            dominated = True
                            break
                    if not dominated:
                        # Remove points in archive dominated by f
                        new_archive = []
                        for af, ap in archive:
                            if not (np.all(f <= af) and np.any(f < af)):
                                new_archive.append((af, ap))
                        new_archive.append((f, p))
                        archive = new_archive
            
        # Return the archive points
        # The evaluator records all returned objective vectors.
        # The task says "Every returned objective vector is recorded".
        # Wait, the task says: "The return value of __call__ is ignored."
        # "Every returned objective vector is recorded" -- this implies the `problem(x)` calls are recorded?
        # "More than 200 evaluations... makes the whole result invalid"
        # "Scoring... exact hypervolume of all recorded vectors"
        # The evaluator likely records the output of `problem(x)` for every call.
        # So we don't need to return the points, the evaluator has them.
        # We just need to make sure we don't exceed the budget.
        
        return None

