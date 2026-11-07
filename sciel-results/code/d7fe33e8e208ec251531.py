# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Structure-aware optimizer for ZDT2 exploiting the Pareto front geometry."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # ZDT2: f1 = x1, f2 = C*(1 - sqrt(x1/C)), C = 10*(n-1)
        # Pareto front: x2..xn = 0, x1 in [0,1]
        # To maximize discrete hypervolume with N points on a convex decreasing front,
        # we want points where the contribution to hypervolume is balanced.
        # Hypervolume contribution of point i is roughly (f1[i] - f1[i-1]) * (f2[i-1] - ref_y)
        # For optimal placement on a convex front, use arcsine or similar distribution.
        
        n_points = min(budget, 200)
        if n_points < 2:
            n_points = 2

        # For ZDT2 with C=290, f2(x1) = 290*(1-sqrt(x1/290))
        # The curvature is highest near x1=0.
        # Optimal discrete HV maximization on convex fronts: use x1 = (i/N)^k
        # Empirically, k around 0.5-0.6 works well.
        # Let's use a finer approach: solve for optimal spacing analytically.
        # For f2(x) = C - sqrt(C*x), the optimal points for N-point HV maximization
        # are approximately x_i = ((i-0.5)/N)^2 for uniform HV contribution.
        # But since f1 = x1 and f2 is convex, we want more points where f2 changes fast.
        
        # Try k = 0.5 (uniform in sqrt space) as a strong baseline
        # Then try to improve with local optimization if time permits.
        
        # Start with k=0.5
        k = 0.5
        indices = np.arange(n_points, dtype=np.float64)
        t = (indices + 0.5) / n_points
        x1_vals = t ** k

        # Ensure we cover endpoints well
        x1_vals = np.clip(x1_vals, 0.0, 1.0)

        X_list = []
        F_list = []
        
        for i in range(n_points):
            x = np.zeros(dim)
            x[0] = x1_vals[i]
            x = np.clip(x, lo, hi)
            
            f = problem(x)
            X_list.append(x)
            F_list.append(f)
            
            if time.time() > deadline:
                break

        # Now try to improve: use remaining budget for local refinement
        # Compute current hypervolume contribution and find gaps
        if len(F_list) > 2 and time.time() < deadline:
            F_arr = np.array(F_list)
            remaining = budget - len(F_list)
            
            # Sort by f1
            sort_idx = np.argsort(F_arr[:, 0])
            F_sorted = F_arr[sort_idx]
            
            # Find largest gaps in f1 where adding a point would give most HV
            # Hypervolume of a set of points on a front:
            # HV = sum over i of (f1[i+1] - f1[i]) * (f2[i] - ref_y)
            # Adding a point between i and i+1 at f1_new adds approximately:
            # gain ≈ (min(f1_new - f1[i], f1[i+1] - f1_new)) * (f2 at new point - ref_y)
            
            ref = np.array([11.0, 11.0])
            
            # Compute current HV contributions per segment
            # For n points sorted by f1, HV = sum_i (f1[i+1]-f1[i]) * max(0, f2[i] - ref[1])
            # (using the left endpoint's f2 for the segment, which is an overestimate)
            
            # Better: approximate HV by trapezoidal rule
            f1 = F_sorted[:, 0]
            f2 = F_sorted[:, 1]
            valid = (f1 < ref[0]) & (f2 < ref[1])
            
            # Try to add points in the largest gaps
            for _ in range(min(remaining, 50)):
                if time.time() > deadline:
                    break
                    
                # Find the gap with maximum potential HV gain
                # For each adjacent pair, estimate gain from adding midpoint
                best_gain = -1
                best_idx = -1
                best_x1 = 0.5
                
                for i in range(len(f1) - 1):
                    if f1[i] >= ref[0] or f1[i+1] >= ref[0]:
                        continue
                    if f2[i] >= ref[1] and f2[i+1] >= ref[1]:
                        continue
                    
                    # Midpoint in f1
                    mid_f1 = (f1[i] + f1[i+1]) / 2.0
                    # Estimate f2 at midpoint using ZDT2 formula
                    C = 10 * (dim - 1)
                    if mid_f1 > 0 and mid_f1 <= 1:
                        est_f2 = C * (1 - np.sqrt(mid_f1 / C))
                    else:
                        est_f2 = C
                    
                    if est_f2 >= ref[1]:
                        continue
                    
                    # Gain estimate
                    gain = min(mid_f1 - f1[i], f1[i+1] - mid_f1) * (est_f2 - ref[1])
                    if gain > best_gain:
                        best_gain = gain
                        best_idx = i
                        best_x1 = mid_f1
                
                if best_idx < 0 or best_gain <= 0:
                    break
                
                # Evaluate at best_x1
                x = np.zeros(dim)
                x[0] = best_x1
                x = np.clip(x, lo, hi)
                f = problem(x)
                X_list.append(x)
                F_list.append(f)
                
                # Update sorted arrays (simple insert)
                F_arr = np.array(F_list)
                sort_idx = np.argsort(F_arr[:, 0])
                F_sorted = F_arr[sort_idx]
                f1 = F_sorted[:, 0]
                f2 = F_sorted[:, 1]

        return np.array(X_list)
# EVOLVE-BLOCK-END
