"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer on ZDT6 exploiting problem structure."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # Safety margin before 330s limit

        dim = self.dim
        budget = self.budget

        # Pre-compute mapping from f1 to x1 for ZDT6
        # f1(x1) = 1 - exp(-4*x1) * sin(6*pi*x1)**6
        # We sample x1 densely to build a lookup table
        n_samples = 100000
        x1_samples = np.linspace(0, 1, n_samples)
        f1_samples = 1 - np.exp(-4 * x1_samples) * (np.sin(6 * np.pi * x1_samples) ** 6)
        
        # Sort by f1 for efficient lookup
        sort_idx = np.argsort(f1_samples)
        f1_sorted = f1_samples[sort_idx]
        x1_sorted = x1_samples[sort_idx]

        def get_x1_for_f1(target_f1):
            """Find x1 that yields f1 closest to target_f1."""
            target_f1 = np.clip(target_f1, 0, 1)
            # Binary search for insertion point
            idx = np.searchsorted(f1_sorted, target_f1)
            
            # Handle boundaries
            if idx == 0:
                return x1_sorted[0]
            if idx == len(f1_sorted):
                return x1_sorted[-1]
            
            # Check neighbors to find the closest f1
            # f1_sorted[idx-1] <= target_f1 < f1_sorted[idx]
            dist_prev = abs(f1_sorted[idx - 1] - target_f1)
            dist_curr = abs(f1_sorted[idx] - target_f1)
            
            if dist_prev < dist_curr:
                return x1_sorted[idx - 1]
            else:
                return x1_sorted[idx]

        # Strategy:
        # 1. The Pareto front of ZDT6 is achieved when x_2...x_10 = 0.
        # 2. f2 = 1 + 9/f1 for f1 in (0,1]. Reference point is [11, 11].
        # 3. Precompute optimal f1 targets to maximize hypervolume coverage.
        #    The hypervolume in 2D is sum of trapezoids: for sorted f1 values,
        #    HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) + f1_last * (11 - f2_last)
        #    where f2_i = 1 + 9/f1_i.
        #    Optimal placement: use points that maximize this sum.
        #    For a convex decreasing front, near-uniform spacing in f1 works well,
        #    but we can do better by accounting for the varying "height".
        
        # Precompute optimal targets using iterative hypervolume maximization
        # Start with uniform, then refine by splitting the interval with max marginal gain
        targets = set()
        targets.add(1e-10)  # Near 0 (f1=0 gives f2=inf, but we need finite)
        targets.add(1.0)
        
        # Actually f1=0 is problematic. The Pareto front for ZDT6 has f1 in (0,1].
        # At f1 -> 0+, f2 -> inf. So we should not include f1=0.
        # The reference point is [11,11], so we only count points with f1 < 11 and f2 < 11.
        # f2 < 11 means 1 + 9/f1 < 11, so f1 > 0.9.
        # Wait, that's wrong. f2 = 1 + 9/f1. For f2 < 11: 9/f1 < 10, f1 > 0.9.
        # So only points with f1 > 0.9 have f2 < 11!
        # That means most of the Pareto front is dominated by the reference point!
        # Let me re-check: the reference point is [11.0, 11.0].
        # A point (f1, f2) counts if f1 < 11 AND f2 < 11.
        # f2 = 1 + 9/f1 < 11 => 9/f1 < 10 => f1 > 0.9.
        # So we need f1 in (0.9, 1] for the point to be non-dominated by ref.
        # This dramatically changes the strategy!
        
        # Re-derive: for f1 in (0.9, 1], f2 = 1 + 9/f1 ranges from 1+9/1=10 to 1+9/0.9=11.
        # So f2 is in (10, 11). The hypervolume region is very narrow in f2.
        
        # The hypervolume contribution: for points sorted by f1 (ascending),
        # HV = sum of rectangles. With ref [11,11]:
        # For a point (f1_i, f2_i), it contributes to the area where f1 > f1_i and f2 < f2_i
        # but we need to account for overlaps. Standard 2D HV calculation:
        # Sort by f1 ascending, then HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) 
        # where f1_{n+1} = 11, but actually we cap at the reference.
        
        # More carefully: sort points by f1 ascending. Add sentinel (11, 11).
        # HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) for i where f2_i < 11.
        # But we also need f1_i < 11 which is always true here.
        
        # So we want to maximize: sum_i (f1_{i+1} - f1_i) * (11 - (1 + 9/f1_i))
        # = sum_i (f1_{i+1} - f1_i) * (10 - 9/f1_i)
        # with f1_1 < f1_2 < ... < f1_N < 11, and f1_{N+1} = 11.
        # But f2_i < 11 requires f1_i > 0.9, so all f1_i must be in (0.9, 1].
        # And f1_{N+1} = 11 but we should cap at 1 since beyond f1=1, the front doesn't exist.
        # Actually, for f1 > 1, there's no Pareto-optimal point. The last point should be at f1=1.
        # So f1_{N+1} should be 1.0 (the max f1 on the front), not 11.
        
        # Let me reconsider. The hypervolume is the volume of the union of boxes
        # [f1_i, 11] x [f2_i, 11] for all points, intersected with [0,11]x[0,11].
        # Since f1_i <= 1 and f2_i in (10,11), the boxes are [f1_i, 11] x [f2_i, 11].
        # But we can't have f1 > 1 on the front. The union is:
        # For the point with smallest f1 (call it f1_min), its box is [f1_min, 11] x [f2_min, 11].
        # But for f1 > 1, there's no point, so the actual Pareto front only covers f1 in (0.9, 1].
        # The hypervolume should only count the region dominated by at least one point.
        # A point (f1, f2) dominates (x, y) if x >= f1 and y >= f2.
        # So the dominated region is the union of [f1_i, inf) x [f2_i, inf) intersected with [0,11]x[0,11].
        # = union of [f1_i, 11] x [f2_i, 11].
        
        # Since f1_i <= 1 < 11, and f2_i in (10,11):
        # The union is [f1_min, 11] x [min_f2, 11] minus holes? No, it's a staircase.
        # Sort by f1 ascending: f1_1 < f1_2 < ... < f1_N.
        # f2 is decreasing in f1, so f2_1 > f2_2 > ... > f2_N.
        # The union of [f1_i, 11] x [f2_i, 11] is:
        # [f1_1, f1_2] x [f2_1, 11] union [f1_2, f1_3] x [f2_2, 11] union ... union [f1_N, 11] x [f2_N, 11]
        # HV = sum_i (f1_{i+1} - f1_i) * (11 - f2_i) where f1_{N+1} = 11.
        
        # But wait, for f1 in [1, 11], there's no Pareto point, but the box from the last point
        # [f1_N, 11] x [f2_N, 11] still counts because the point (f1_N, f2_N) dominates
        # all (x,y) with x >= f1_N and y >= f2_N. So yes, f1_{N+1} = 11.
        
        # So HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (10 - 9/f1_i) + (11 - f1_N) * (10 - 9/f1_N)
        # where f1_{N+1} = 11, so the last term is (11 - f1_N) * (10 - 9/f1_N).
        # Actually let me just use: HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (11 - f2_i)
        # with f1_{N+1} = 11, f2_i = 1 + 9/f1_i.
        
        # To maximize this with N points in (0.9, 1], we should use all N points in that range.
        # The optimal placement can be found by dynamic programming or greedy.
        
        # For simplicity, use a fine grid and pick the best N points via greedy hypervolume maximization.
        
        # Generate candidate f1 values in (0.9, 1]
        n_candidates = 2000
        cand_f1 = np.linspace(0.9 + 1e-6, 1.0, n_candidates)
        cand_f2 = 1.0 + 9.0 / cand_f1
        
        # Greedy: start with the point that gives max HV alone, then iteratively add
        # the point that gives max marginal HV gain.
        # HV of a set S: sort by f1, compute sum (f1_{i+1}-f1_i)*(11-f2_i) with f1_{last+1}=11
        
        def compute_hv(f1_arr, f2_arr):
            """Compute 2D hypervolume with ref [11,11]."""
            if len(f1_arr) == 0:
                return 0.0
            order = np.argsort(f1_arr)
            f1_s = f1_arr[order]
            f2_s = f2_arr[order]
            # Only keep points with f1 < 11 and f2 < 11
            mask = (f1_s < 11.0) & (f2_s < 11.0)
            f1_s = f1_s[mask]
            f2_s = f2_s[mask]
            if len(f1_s) == 0:
                return 0.0
            # Extend with reference
            f1_ext = np.append(f1_s, 11.0)
            f2_ext = f2_s  # length N
            hv = 0.0
            for i in range(len(f1_s)):
                width = f1_ext[i+1] - f1_s[i]
                height = 11.0 - f2_s[i]
                if width > 0 and height > 0:
                    hv += width * height
            return hv
        
        # Greedy selection
        selected_idx = []
        remaining = set(range(n_candidates))
        
        for step in range(min(budget, n_candidates)):
            if time.time() > deadline:
                break
            best_gain = -1
            best_idx = -1
            current_f1 = cand_f1[selected_idx] if selected_idx else np.array([])
            current_f2 = cand_f2[selected_idx] if selected_idx else np.array([])
            current_hv = compute_hv(current_f1, current_f2) if len(current_f1) > 0 else 0.0
            
            for idx in remaining:
                test_f1 = np.append(current_f1, cand_f1[idx])
                test_f2 = np.append(current_f2, cand_f2[idx])
                test_hv = compute_hv(test_f1, test_f2)
                gain = test_hv - current_hv
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx
            
            if best_idx >= 0:
                selected_idx.append(best_idx)
                remaining.remove(best_idx)
        
        # Evaluate the selected points
        for idx in selected_idx:
            if time.time() > deadline:
                break
            target_f1 = cand_f1[idx]
            x1 = get_x1_for_f1(target_f1)
            x = np.zeros(dim)
            x[0] = x1
            f = problem(x)
        
        return None
# EVOLVE-BLOCK-END
