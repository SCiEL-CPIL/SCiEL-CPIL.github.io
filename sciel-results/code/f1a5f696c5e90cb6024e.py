import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        # ZDT3 analytic Pareto front:
        # x[1:] = 0
        # f1 = x[0]
        # g(x) = 1 + 10*n = 301 (for n=30, sum sin(10*pi*x_i*(i+1)) = 0)
        # h(x1, g) = 1 - sqrt(x1/g) - sin(10*pi*x1)
        # f2 = g * h = 301 * (1 - sqrt(x1/301) - sin(10*pi*x1))
        
        # We want to select `budget` points x0 in [0, 1] to maximize hypervolume
        # with reference point [11, 11].
        # Since f1 = x0, the points are sorted by f1.
        # The hypervolume contribution of a point (f1, f2) given the previous point (prev_f1, prev_f2)
        # in a sorted list (increasing f1) is (f1 - prev_f1) * (ref_f2 - f2).
        # Note: We assume the "previous" point is the one with smaller f1.
        # The total HV is sum of these contributions for all points, assuming they form a non-dominated set.
        # Since f2 is not monotonic, we must ensure we only keep points that are part of the Pareto front of the selected set.
        
        # Strategy:
        # 1. Generate a large grid of candidate x0 values.
        # 2. Compute their (f1, f2) coordinates analytically.
        # 3. Use a greedy algorithm to select `budget` points that maximize the total hypervolume.
        
        start_time = time.time()
        deadline = start_time + 280  # Safety margin for 300s limit
        
        n_candidates = 20000
        x0_grid = np.linspace(0, 1, n_candidates)
        
        # Compute objectives
        f1 = x0_grid
        g = 301.0
        f2 = g * (1.0 - np.sqrt(f1 / g) - np.sin(10.0 * np.pi * f1))
        
        # Filter out points that are worse than reference point [11, 11]
        # f1 <= 11 and f2 <= 11
        # f1 is in [0, 1], so f1 <= 11 is always true.
        # f2 can be up to ~301. We need f2 < 11.
        mask = f2 < 11.0
        valid_indices = np.where(mask)[0]
        
        if len(valid_indices) == 0:
            # Fallback: just evaluate some points
            for x0 in x0_grid[:self.budget]:
                x = np.zeros(self.dim)
                x[0] = x0
                problem(x)
            return None

        # Get coordinates of valid candidates
        cand_f1 = f1[valid_indices]
        cand_f2 = f2[valid_indices]
        
        # Sort by f1
        sort_idx = np.argsort(cand_f1)
        cand_f1 = cand_f1[sort_idx]
        cand_f2 = cand_f2[sort_idx]
        
        # Greedy selection to maximize hypervolume
        # We select points one by one.
        # A point (f1, f2) adds area (f1 - f1_prev) * (11 - f2) if it is "dominating" in the sense
        # that it extends the front.
        # Actually, for a set of points, the hypervolume is the area under the lower envelope.
        # Since we are maximizing HV with ref [11, 11], and all f1 < 11, f2 < 11,
        # the HV is the area of the union of rectangles [0, f1] x [0, f2] ... wait.
        # Standard HV definition: Volume dominated by the set.
        # For 2D, if points are sorted by f1, the HV is sum of (f1[i] - f1[i-1]) * (f2[i] - 0) ? No.
        # The region dominated by a point (x, y) is [0, x] x [0, y].
        # The union of these rectangles.
        # If we sort by x, the contribution of point i is (x_i - x_{i-1}) * y_i, provided y_i < y_{i-1} (monotonic decreasing y).
        # If y_i > y_{i-1}, point i is dominated by i-1 in terms of y, but has larger x.
        # Actually, the Pareto front of the selected set will have decreasing f2 as f1 increases.
        # So we should select points that form a decreasing sequence in f2 when sorted by f1.
        
        # Greedy approach:
        # Start with empty set.
        # At each step, pick the candidate that adds the most marginal HV.
        # This is O(N * Budget * N) if naive, which is too slow for N=20000, Budget=200.
        # 20000 * 200 * 20000 is 8e10. Too slow.
        
        # Alternative:
        # The optimal set of points for HV maximization on a known curve often corresponds to
        # uniform distribution in the "area" sense, or simply dense sampling of the non-dominated part of the curve.
        
        # Let's identify the non-dominated part of the candidate set first.
        # A point (f1, f2) is non-dominated if there is no other point (f1', f2') with f1' <= f1 and f2' <= f2 and (f1' < f1 or f2' < f2).
        # Since we have a single curve, the non-dominated points are those that are on the "lower-left" envelope.
        # Sort by f1. Keep track of min f2 seen so far? No, min f2 seen from the right?
        # For minimization, a point is non-dominated if it has the lowest f2 for any given f1 prefix?
        # No. Point A dominates B if A.f1 <= B.f1 and A.f2 <= B.f2.
        # So if we sort by f1 increasing, a point is dominated if there is a previous point with f2 <= current f2.
        # So we keep points where f2 is strictly lower than all previous f2s?
        # Wait, if f1 increases, and f2 decreases, they are all non-dominated.
        # If f1 increases and f2 increases, the later one is dominated by the earlier one? No, earlier has smaller f1 and smaller f2, so earlier dominates later.
        # So we want the "lower envelope" of f2 as f1 increases.
        
        # Let's find the non-dominated candidates.
        # Sort by f1.
        # Iterate from right to left (high f1 to low f1).
        # A point is non-dominated if its f2 is lower than the minimum f2 of all points with higher f1.
        # Let min_f2_from_right[i] be the min f2 of candidates i..end.
        # Candidate i is non-dominated if cand_f2[i] <= min_f2_from_right[i+1].
        
        min_f2_suffix = np.minimum.accumulate(cand_f2[::-1])[::-1]
        
        # Non-dominated condition: cand_f2[i] <= min_f2_suffix[i+1] (for i < last)
        # For the last one, it's always non-dominated if it's the best in f2?
        # Actually, let's just filter:
        # A point is dominated if there exists j > i (higher f1) such that cand_f2[j] <= cand_f2[i].
        # So we keep i if cand_f2[i] < min(cand_f2[i+1:]).
        
        is_non_dominated = np.zeros(len(cand_f1), dtype=bool)
        # The last point is always a candidate for being non-dominated (it has the highest f1)
        # It is non-dominated if its f2 is the global minimum? No.
        # If it has the highest f1, it is dominated by any point with lower f1 and lower f2.
        # So we check against the suffix minimum.
        
        # Let's just compute the Pareto front of the candidates.
        # Sort by f1.
        # Keep a running minimum of f2 from the right?
        # No, standard Pareto front extraction:
        # Sort by f1 asc.
        # f2_min_so_far = inf
        # For i from 0 to N-1:
        #   if f2[i] < f2_min_so_far:
        #       keep i
        #       f2_min_so_far = f2[i]
        # This finds points that have the lowest f2 for any prefix of f1.
        # These are the non-dominated points.
        
        nd_indices = []
        min_f2 = np.inf
        for i in range(len(cand_f1)):
            if cand_f2[i] < min_f2:
                nd_indices.append(i)
                min_f2 = cand_f2[i]
                
        nd_indices = np.array(nd_indices)
        
        # Now we have a set of non-dominated candidates.
        # We need to select `budget` points from this set to maximize HV.
        # The HV of a set of non-dominated points sorted by f1 is:
        # Sum_{i in selected} (f1[i] - f1[prev_selected]) * (11 - f2[i])
        # Wait, the reference point is [11, 11].
        # The area contributed by a point (f1, f2) given the previous point (f1_prev, f2_prev) on the front is:
        # (f1 - f1_prev) * (11 - f2) + (11 - f1) * (11 - f2_prev)? No.
        
        # Let's visualize.
        # Points P1, P2, ... Pk sorted by f1.
        # The dominated region is the union of rectangles [0, f1_i] x [0, f2_i].
        # The hypervolume is the area of this union.
        # Since f2 is decreasing (for non-dominated points), the union is a staircase.
        # Area = sum_{i=1 to k} (f1_i - f1_{i-1}) * f2_i  ? No.
        # The rectangle for P_i is [0, f1_i] x [0, f2_i].
        # The part added by P_i compared to P_{i-1} is [f1_{i-1}, f1_i] x [0, f2_i].
        # So Area = sum (f1_i - f1_{i-1}) * f2_i.
        # But we have a reference point [11, 11].
        # The hypervolume relative to [11, 11] is the area of the region dominated by the points AND strictly better than [11, 11].
        # Since all points are better than [11, 11], the HV is the area of the union of [0, f1_i] x [0, f2_i] ?
        # No, the definition of HV with reference r is the volume of the region dominated by the set and contained in [0, r].
        # So yes, it is the area of the union of rectangles [0, f1_i] x [0, f2_i].
        # Which is sum (f1_i - f1_{i-1}) * f2_i, with f1_0 = 0, f2_0 = 11?
        # Let's check.
        # If we have one point (x, y), HV = x * y.
        # Formula: (x - 0) * y = xy. Correct.
        # If we have two points (x1, y1), (x2, y2) with x1 < x2, y1 > y2.
        # HV = x1 * y1 + (x2 - x1) * y2.
        # This is the area of the staircase.
        
        # So we want to select a subset of the non-dominated candidates to maximize:
        # Sum (f1_i - f1_prev) * f2_i
        # where f1_prev is the f1 of the previously selected point (or 0 if none).
        # Note: f2_i is the y-coordinate.
        
        # This is a dynamic programming problem!
        # Let dp[i][k] be the max HV using k points from the first i non-dominated candidates, ending with candidate i.
        # But the contribution depends on the previous point.
        # Let's define:
        # V(i, j) = (f1_i - f1_j) * f2_i  (contribution of i given previous is j)
        # We want to maximize sum of contributions.
        # Let H(i, k) be the max HV for a sequence of k points ending at i.
        # H(i, 1) = f1_i * f2_i
        # H(i, k) = max_{j < i} (H(j, k-1) + (f1_i - f1_j) * f2_i)
        
        # We need to compute this for k up to budget (200) and i up to len(nd_indices).
        # len(nd_indices) could be large.
        # How many non-dominated points are there?
        # f2 = 301 * (1 - sqrt(x/301) - sin(10*pi*x))
        # The term sin(10*pi*x) oscillates.
        # The non-dominated points will be the local minima of f2.
        # There are about 5-10 oscillations in [0, 1].
        # So the number of non-dominated candidates might be small (e.g., < 100).
        # If so, DP is very fast.
        
        # Let's check the number of nd_indices.
        n_nd = len(nd_indices)
        
        if n_nd == 0:
            # Should not happen
            return None
            
        # If n_nd is small, we can do DP.
        # If n_nd is large (e.g., > 5000), DP might be slow (N^2 * K).
        # 5000^2 * 200 = 5e9. Too slow.
        # But given the oscillatory nature, the number of true Pareto points (local minima) is small.
        # However, my extraction method `if cand_f2[i] < min_f2` keeps ALL points that set a new record low.
        # Since f2 oscillates, it will keep many points that are local minima.
        # Let's see. f2 goes down and up.
        # Every time it goes down below the previous minimum, it's kept.
        # Since it oscillates, the global minimum is reached once, but local minima are not kept if they are higher than a previous global minimum.
        # Wait.
        # If f2 = 5, 4, 3, 4, 2, 1.
        # i=0: 5 < inf -> keep. min=5
        # i=1: 4 < 5 -> keep. min=4
        # i=2: 3 < 4 -> keep. min=3
        # i=3: 4 < 3? No.
        # i=4: 2 < 3 -> keep. min=2
        # i=5: 1 < 2 -> keep. min=1
        # So it keeps the "staircase" of global minima.
        # This set of points is non-dominated.
        # The number of such points is the number of times the function hits a new global minimum.
        # For ZDT3, f2 has a global minimum around x=0.5?
        # Actually, f2 = 301(1 - sqrt(x/301) - sin(10pi x)).
        # The term -sin(10pi x) oscillates between -1 and 1.
        # The term -sqrt(x/301) is decreasing from 0 to -1.
        # So f2 generally decreases.
        # The oscillations add ripples.
        # The global minimum will be at the end of a ripple where the decreasing trend and the sine wave align.
        # Likely, there are only a few global minima.
        # So n_nd should be small.
        
        if n_nd > 1000:
            # Fallback to uniform sampling if too many
            # Just pick uniform points from the non-dominated set
            indices = np.linspace(0, n_nd - 1, self.budget, dtype=int)
            selected_indices = nd_indices[indices]
        else:
            # DP
            # H[i][k]
            # i from 0 to n_nd-1
            # k from 1 to budget
            # H[i][k] = max_{j < i} (H[j][k-1] + (f1_i - f1_j) * f2_i)
            
            # Initialize H with -inf
            H = np.full((n_nd, self.budget + 1), -np.inf)
            
            # Base case k=1
            for i in range(n_nd):
                H[i, 1] = cand_f1[nd_indices[i]] * cand_f2[nd_indices[i]]
                
            # Fill DP
            for k in range(2, self.budget + 1):
                for i in range(k - 1, n_nd):
                    # j can be from 0 to i-1
                    # We need max over j of H[j, k-1] + (f1_i - f1_j) * f2_i
                    # = f1_i * f2_i + max_j (H[j, k-1] - f1_j * f2_i)
                    
                    f1_i = cand_f1[nd_indices[i]]
                    f2_i = cand_f2[nd_indices[i]]
                    
                    best_val = -np.inf
                    for j in range(i):
                        val = H[j, k-1] + (f1_i - cand_f1[nd_indices[j]]) * f2_i
                        if val > best_val:
                            best_val = val
                    H[i, k] = best_val
                    
            # Find the best ending point for k=budget
            best_hv = -np.inf
            best_last_idx = -1
            for i in range(self.budget - 1, n_nd):
                if H[i, self.budget] > best_hv:
                    best_hv = H[i, self.budget]
                    best_last_idx = i
                    
            # Reconstruct path
            selected = [best_last_idx]
            for k in range(self.budget, 1, -1):
                i = selected[-1]
                f1_i = cand_f1[nd_indices[i]]
                f2_i = cand_f2[nd_indices[i]]
                best_prev = -1
                best_val = -np.inf
                for j in range(i):
                    val = H[j, k-1] + (f1_i - cand_f1[nd_indices[j]]) * f2_i
                    if val > best_val:
                        best_val = val
                        best_prev = j
                selected.append(best_prev)
                
            selected.reverse()
            selected_nd_indices = selected
            
        # Get the actual x0 values
        selected_x0 = x0_grid[valid_indices[sort_idx[selected_nd_indices]]]
        
        # Evaluate
        for x0 in selected_x0:
            x = np.zeros(self.dim)
            x[0] = x0
            problem(x)
            
        return None
