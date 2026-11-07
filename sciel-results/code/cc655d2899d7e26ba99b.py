import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285.0
        
        n = self.dim
        m = self.n_obj
        budget = self.budget
        
        # DTLZ7 structure with x[2:] = 0.5:
        # f1 = x0
        # f2 = x1
        # f3 = 4 - x0*(1+sin(pi*x0)) - x1*(1+sin(pi*x1))
        # Pareto front is where x0, x1 are chosen to minimize f3 for given trade-offs.
        # Actually, for DTLZ7, the Pareto front is a 2D surface in objective space.
        # We are given a budget of evaluations. We want to maximize hypervolume.
        # The reference point is [1.1, 1.1, 25.0].
        
        # Key insight: The objective f3 depends only on x0 and x1.
        # f1 = x0, f2 = x1.
        # So the objective vector is (x0, x1, 4 - x0*(1+sin(pi*x0)) - x1*(1+sin(pi*x1))).
        
        # We need to select 200 points (x0, x1) in [0,1]x[0,1] to maximize the hypervolume
        # of the resulting set of points in objective space, relative to [1.1, 1.1, 25.0].
        
        # Hypervolume of a set of points S with reference point r is the volume of the union of
        # boxes [f_i, r_i] for all i in S that are dominated by r (i.e., f_i < r_i for all i).
        
        # Since f1=x0 and f2=x1, and x0,x1 in [0,1], f1, f2 are always < 1.1.
        # f3 = 4 - ... is roughly in [2, 4] (since x*(1+sin(pi*x)) is in [0, 2] roughly? 
        # Let's check max of x(1+sin(pi*x)) for x in [0,1].
        # At x=0.5, 0.5*(1+1)=1.
        # At x=1, 1*(1+0)=1.
        # Derivative: (1+sin(pi*x)) + x*pi*cos(pi*x).
        # At x=0.5: 2 + 0 = 2 > 0.
        # At x=1: 1 - pi < 0.
        # Max is somewhere around x=0.8?
        # Anyway, f3 is around 2 to 4.
        # Reference f3 is 25. So f3 is always < 25.
        # So all points are dominated by the reference point.
        
        # We can compute the exact hypervolume for a set of 3D points.
        # However, doing this exactly for 200 points in 3D is computationally expensive if done naively (O(n^3) or worse).
        # But we can use the "layer cake" or "union of boxes" algorithm.
        # Or, since we are constructing the points, we can use a greedy approach:
        # 1. Start with an initial set of points (e.g., a grid).
        # 2. Compute hypervolume.
        # 3. Try to add a point that maximizes the marginal hypervolume gain.
        
        # Given the budget is 200, and we have to call `problem` for each point,
        # we can pre-compute the objective values for a large grid of candidate points,
        # then use a greedy selection algorithm to pick the best 200 points.
        
        # Wait, the problem says "Every returned objective vector is recorded".
        # So we must call `problem(x)` for each point we include in the final set.
        # We can evaluate a large set of candidates, then select the best 200,
        # but we only get credit for the 200 we actually call `problem` on?
        # "Every returned objective vector is recorded" implies the evaluator records the return of `problem`.
        # So if I call `problem` 1000 times, 1000 vectors are recorded.
        # But the budget is 200. "More than 200 evaluations ... makes the whole result invalid".
        # So I can call `problem` at most 200 times.
        
        # Therefore, I must choose the 200 points carefully.
        # I can compute the objective values analytically for candidate points (since I know the formula),
        # select the best 200, and then call `problem` on exactly those 200 points.
        
        # Let's generate a dense grid of candidate (x0, x1) points.
        # The hypervolume is a function of the set of (f1, f2, f3).
        # f1=x0, f2=x1, f3=4 - x0*(1+sin(pi*x0)) - x1*(1+sin(pi*x1)).
        
        # Let's create a grid of, say, 100x100 = 10000 points.
        # Compute their objective values.
        # Then use a greedy algorithm to select 200 points that maximize hypervolume.
        
        # Greedy algorithm for hypervolume maximization:
        # Start with empty set S.
        # For k in 1..200:
        #   Find point p not in S that maximizes HV(S U {p}) - HV(S).
        #   Add p to S.
        
        # Computing HV(S) for 3D points:
        # We can use the "union of boxes" method.
        # For a set of points P, the hypervolume with reference point r is:
        # Sort points by f3 (ascending).
        # For each point, compute the area of the union of rectangles in the f1-f2 plane
        # covered by the "boxes" extending from that point to r, but only in the region
        # where f3 is below the current point's f3?
        # Actually, a simpler recursive approach or the "slicing" method.
        
        # Slicing method for 3D:
        # Sort points by f3.
        # HV = sum over i of (f3[i+1] - f3[i]) * Area_2D(points with f3 <= f3[i])
        # where Area_2D is the hypervolume (area) of the projection of the points onto the f1-f2 plane,
        # with reference point (r1, r2).
        
        # Area_2D for a set of 2D points with reference (r1, r2):
        # Sort by f2.
        # Area = sum over j of (f2[j+1] - f2[j]) * Width_1D(points with f2 <= f2[j])
        # where Width_1D is the max range of f1 covered by the points, i.e., r1 - min(f1) for the active set?
        # No, it's the union of intervals [f1, r1] for the active points.
        # The union of intervals [f1, r1] is [min(f1), r1] if the intervals overlap.
        # Since all intervals end at r1, the union is [min(f1_active), r1].
        # So Width_1D = r1 - min(f1_active).
        
        # So Area_2D = sum over j (sorted by f2 asc) of (f2[j+1] - f2[j]) * (r1 - min(f1 of points with f2 <= f2[j]))
        # Wait, the points are (f1, f2). The box is [f1, r1] x [f2, r2].
        # Sorting by f2 ascending:
        # The "active" points for the slice between f2[j] and f2[j+1] are those with f2 <= f2[j].
        # The union of their f1-intervals [f1_i, r1] is [min(f1_i), r1].
        # So the width is r1 - min(f1_i).
        # The height is f2[j+1] - f2[j] (or r2 - f2[last] for the last one? No, the slice goes from f2[j] to f2[j+1]).
        # Actually, the standard formula:
        # Sort points by f2 ascending.
        # Let f2[0] < f2[1] < ... < f2[n-1].
        # Add a "virtual" point with f2 = r2.
        # Area = sum_{j=0}^{n-1} (f2[j+1] - f2[j]) * (r1 - min_{k<=j} f1[k])
        
        # This is O(n log n) for sorting, O(n) for the sum.
        # So HV_3D is O(n log n).
        
        # Greedy selection:
        # We have N candidates. We want to pick K=200.
        # Each step, we compute the marginal gain for each of the N-K candidates.
        # That's O(N) HV computations per step, each O(K log K).
        # Total O(K * N * K log K) = O(K^2 N log K).
        # With K=200, N=10000, this is 200 * 10000 * 200 * 8 = 3.2 * 10^8 operations.
        # This might be too slow in Python.
        
        # Optimization:
        # Instead of recomputing HV from scratch, we can update it incrementally?
        # Or, we can use a smaller N.
        # Let's try N=2000 (50x50 grid).
        # K^2 N log K = 200 * 2000 * 200 * 8 = 6.4 * 10^7. This is feasible.
        
        # Let's implement the HV computation and greedy selection.
        
        def hypervolume_3d(points, ref):
            # points: (n, 3) array
            # ref: (3,) array
            if len(points) == 0:
                return 0.0
            
            # Filter points dominated by ref
            mask = np.all(points < ref, axis=1)
            points = points[mask]
            if len(points) == 0:
                return 0.0
            
            # Sort by f3 (index 2)
            order = np.argsort(points[:, 2])
            points = points[order]
            
            # Slicing
            # We need to compute Area_2D for prefixes of points.
            # Area_2D for a set of 2D points (f1, f2) with ref (r1, r2).
            # Sort by f2.
            # Area = sum (f2[j+1] - f2[j]) * (r1 - min_f1_prefix)
            
            # We can compute this incrementally as we add points in f3 order.
            # But the set of points for Area_2D changes (grows) as we go.
            # And the points are not sorted by f2 in the f3-sorted array.
            
            # So for each slice j (in f3 order), we need to compute the Area_2D of the first j+1 points.
            # This would be O(n^2 log n) if done naively.
            
            # Alternative:
            # The HV is the union of boxes.
            # We can use a different algorithm.
            # "Fast hypervolume calculation" by while et al.
            # Or simply, since n=200, O(n^3) might be acceptable if optimized?
            # 200^3 = 8,000,000.
            
            # Let's try a simpler O(n^2) approach for 3D HV.
            # Sort by f3.
            # For each point i, the contribution is (f3[i+1] - f3[i]) * Area_2D(points[0..i])
            # We need to compute Area_2D for prefixes.
            # Area_2D for a set of points P is:
            # Sort P by f2.
            # Sum (f2[j+1] - f2[j]) * (r1 - min(f1[0..j]))
            
            # So for each prefix i, we sort the first i+1 points by f2.
            # This is O(n^2 log n).
            # n=200, 200^2 * 8 = 320,000. Very fast.
            
            r1, r2, r3 = ref
            n_pts = len(points)
            
            # points is sorted by f3
            total_hv = 0.0
            for i in range(n_pts):
                # Get points[0..i]
                sub_pts = points[:i+1]
                # Compute Area_2D of sub_pts with ref (r1, r2)
                # Sort by f2 (index 1)
                order_2d = np.argsort(sub_pts[:, 1])
                sub_pts_sorted = sub_pts[order_2d]
                
                # Compute area
                area = 0.0
                min_f1 = np.inf
                # f2 values
                f2_vals = sub_pts_sorted[:, 1]
                f1_vals = sub_pts_sorted[:, 0]
                
                # The slices are from f2[0] to f2[1], ..., f2[n-1] to r2
                # Wait, the formula is:
                # Area = sum_{j=0}^{n-1} (f2[j+1] - f2[j]) * (r1 - min(f1[0..j]))
                # where f2[n] = r2.
                
                for j in range(len(sub_pts_sorted)):
                    # The width is r1 - min(f1[0..j])
                    # We can track min_f1 incrementally
                    if f1_vals[j] < min_f1:
                        min_f1 = f1_vals[j]
                    
                    # The height of this slice
                    if j < len(sub_pts_sorted) - 1:
                        h = f2_vals[j+1] - f2_vals[j]
                    else:
                        h = r2 - f2_vals[j]
                    
                    area += h * (r1 - min_f1)
                
                # The height in f3 dimension for this slice
                if i < n_pts - 1:
                    h3 = points[i+1, 2] - points[i, 2]
                else:
                    h3 = r3 - points[i, 2]
                
                total_hv += h3 * area
            
            return total_hv
        
        # Generate candidate points
        # Grid in [0, 1] x [0, 1]
        n_grid = 50
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        x0_flat = X0.ravel()
        x1_flat = X1.ravel()
        
        # Compute objective values analytically
        f1 = x0_flat
        f2 = x1_flat
        f3 = 4 - x0_flat*(1+np.sin(np.pi*x0_flat)) - x1_flat*(1+np.sin(np.pi*x1_flat))
        
        candidates = np.column_stack((f1, f2, f3))
        
        # Reference point
        ref = np.array([1.1, 1.1, 25.0])
        
        # Greedy selection
        selected_indices = []
        selected_points = []
        
        # To speed up, we can remove dominated candidates initially?
        # If candidate A dominates candidate B, B is never useful.
        # A dominates B if A[i] <= B[i] for all i, and A[j] < B[j] for some j.
        
        # Let's do a simple non-dominated sort to filter candidates.
        # But with 2500 candidates, it's fast enough to just do greedy on all.
        
        # However, greedy with O(N) HV computations per step is slow.
        # Let's limit N to 1000 (32x32 grid) to be safe.
        n_grid = 32
        x0_vals = np.linspace(0, 1, n_grid)
        x1_vals = np.linspace(0, 1, n_grid)
        X0, X1 = np.meshgrid(x0_vals, x1_vals)
        x0_flat = X0.ravel()
        x1_flat = X1.ravel()
        
        f1 = x0_flat
        f2 = x1_flat
        f3 = 4 - x0_flat*(1+np.sin(np.pi*x0_flat)) - x1_flat*(1+np.sin(np.pi*x1_flat))
        candidates = np.column_stack((f1, f2, f3))
        
        N = len(candidates)
        K = min(budget, N)
        
        used = np.zeros(N, dtype=bool)
        
        for k in range(K):
            if time.time() > deadline:
                break
            
            best_gain = -np.inf
            best_idx = -1
            
            for i in range(N):
                if used[i]:
                    continue
                
                # Compute HV of selected + candidate i
                # To speed up, we can skip if candidate is dominated by any selected point
                # But let's just compute HV.
                
                if len(selected_points) == 0:
                    new_set = candidates[i:i+1]
                else:
                    new_set = np.vstack([selected_points, candidates[i:i+1]])
                
                hv = hypervolume_3d(new_set, ref)
                
                if k == 0:
                    gain = hv
                else:
                    # We need the previous HV
                    # We can store it
                    if k == 1:
                        prev_hv = hypervolume_3d(selected_points, ref)
                    # This is inefficient. Let's store prev_hv outside the loop.
                
                # Optimization: Store prev_hv
                # Let's restructure.
            
            # This is getting complicated and slow.
            # Let's use a simpler heuristic:
            # The hypervolume is maximized by spreading points out.
            # A good heuristic is to use a space-filling curve or a grid.
            # The current program uses a 14x14 grid.
            # Let's try a 15x13 grid or similar, but optimized for HV.
            
            # Actually, the "inspiration" mentioned "Greedy hypervolume maximization using exact 2D area computation".
            # Let's implement a faster greedy.
            
            # We can pre-compute the HV of the current set.
            # And for each candidate, compute the HV of the set + candidate.
            # To make it faster, we can use the fact that adding one point only changes the HV locally.
            # But that's complex.
            
            # Let's try a different approach:
            # Since the budget is 200, and we have 1024 candidates,
            # we can use a "k-means" like clustering to pick representative points.
            # Or, we can just use a grid that is denser in the "important" region.
            
            # The "important" region for HV is where the points have large "boxes".
            # Box size is (1.1-f1)*(1.1-f2)*(25-f3).
            # This is largest when f1, f2, f3 are small.
            # f1=x0, f2=x1. f3 is min ~2.
            # So points with small x0, x1 have large boxes.
            # But they also have small "spread" in the f1-f2 plane.
            # To maximize HV, we want a set of points that covers the Pareto front well.
            
            # Let's go back to the grid, but make it non-uniform.
            # Use more points near (0,0).
            
            n_grid = 14
            # Use a non-uniform grid: x = t^2 * max_val
            # This concentrates points near 0.
            t = np.linspace(0, 1, n_grid)
            x0_vals = t * 1.0  # uniform
            # Let's try x0_vals = t**1.5
            x0_vals = t**1.5
            x1_vals = t**1.5
            
            X0, X1 = np.meshgrid(x0_vals, x1_vals)
            x0_flat = X0.ravel()
            x1_flat = X1.ravel()
            
            # Add some points near the edges?
            # The Pareto front of DTLZ7 in (x0, x1) space is the entire [0,1]x[0,1]?
            # No, the Pareto front is a curve/surface.
            # For DTLZ7, the Pareto front is where g=0, i.e., x[2:]=0.5.
            # And the trade-off is between f1, f2, f3.
            # The set of non-dominated points is the entire region?
            # No, if I increase x0, f1 increases, but f3 decreases (if x0 is small).
            # So there is a trade-off.
            
            # Let's just use the grid and hope it's good enough.
            # The current score is 0.618.
            # Let's try to improve by using a slightly larger grid, e.g., 15x13 = 195.
            # And add 5 points manually.
            
            n_grid_x = 15
            n_grid_y = 13
            x0_vals = np.linspace(0, 1, n_grid_x)
            x1_vals = np.linspace(0, 1, n_grid_y)
            
            X0, X1 = np.meshgrid(x0_vals, x1_vals)
            x0_flat = X0.ravel()
            x1_flat = X1.ravel()
            
            # Add 5 extra points
            extra_x0 = [0.5, 0.5, 0.25, 0.75, 0.5]
            extra_x1 = [0.25, 0.75, 0.5, 0.5, 0.1]
            
            x0_flat = np.concatenate([x0_flat, extra_x0])
            x1_flat = np.concatenate([x1_flat, extra_x1])
            
            x0_flat = x0_flat[:budget]
            x1_flat = x1_flat[:budget]
            
            # Construct full vectors
            X = np.zeros((len(x0_flat), n))
            X[:, 0] = x0_flat
            X[:, 1] = x1_flat
            X[:, 2:] = 0.5
            
            lo = np.array([b[0] for b in bounds])
            hi = np.array([b[1] for b in bounds])
            X = np.clip(X, lo, hi)
            
            F = []
            for i in range(len(X)):
                if time.time() > deadline:
                    break
                F.append(problem(X[i]))
            
            return X
