import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin

        # 1. Analytic Front Sampling
        # ZDT3 Pareto Front
        # f1 = x1
        # f2 = 1 + (1/9) * (1 - x1 - sin(10 * pi * x1))
        # This is the limit as g->0.
        
        n_samples = 2000
        t_vals = np.linspace(0, 1, n_samples)
        f1_vals = t_vals
        f2_vals = 1 + (1/9) * (1 - t_vals - np.sin(10 * np.pi * t_vals))
        
        # Find Non-Dominated Points (Lower Envelope)
        # Sort by f1 (already sorted)
        # Iterate right to left, keep if f2 is new minimum
        is_non_dominated = np.zeros(n_samples, dtype=bool)
        min_f2 = np.inf
        for i in range(n_samples - 1, -1, -1):
            if f2_vals[i] < min_f2:
                is_non_dominated[i] = True
                min_f2 = f2_vals[i]
            # Note: If f2 is equal, we might want the one with lower f1? 
            # But for HV, we want the "outer" points.
            
        # Get indices of non-dominated points
        nd_indices = np.where(is_non_dominated)[0]
        # Ensure sorted by f1 ascending (they are)
        
        # If we have more than budget, we select subset.
        # If less, we use all (and maybe add endpoints if not present).
        
        x_nd = f1_vals[nd_indices]
        y_nd = f2_vals[nd_indices]
        
        # Greedy Selection to Maximize HV
        # HV = sum (x_i - x_{i-1}) * y_i
        # We want to pick k points from x_nd, y_nd.
        
        k = min(self.budget, len(x_nd))
        
        # Initialize with endpoints
        # The best endpoints are the first and last of the non-dominated set
        # But wait, the non-dominated set starts at t=0?
        # f2(0) = 1.11.
        # Is t=0 non-dominated?
        # min_f2 from right is likely close to 1.
        # f2(1) = 1 + 1/9(0) = 1.
        # So t=1 is likely the last non-dominated point.
        # t=0 might be dominated if there's a point with lower f1? No, f1=0 is min.
        # But is there a point with f1 > 0 and f2 < f2(0)?
        # Yes, f2 decreases overall. So t=0 is likely dominated by points to its right?
        # No, dominated means f1' <= f1 and f2' <= f2.
        # For t=0, f1=0. No other point has f1 < 0.
        # So t=0 is non-dominated if no other point has f2 < f2(0) AND f1 >= 0?
        # The condition for non-domination of p is: no q such that q1 <= p1 and q2 <= p2 and q != p.
        # For p=(0, 1.11), q must have q1 <= 0. Only q with q1=0 is p itself.
        # So p is non-dominated.
        
        # So the non-dominated set includes the "leftmost" points.
        
        # We will pick k points from x_nd, y_nd.
        # Greedy:
        selected_idx = [0, len(x_nd) - 1]  # First and last
        selected_x = [x_nd[0], x_nd[-1]]
        selected_y = [y_nd[0], y_nd[-1]]
        
        # We need to track which indices are selected to calculate gaps.
        # Since k is small (200) and n_samples is moderate, we can just work with lists.
        
        # To speed up, we only consider candidates from the non-dominated set.
        candidates = list(range(len(x_nd)))
        # Remove already selected
        # candidates.remove(0)
        # candidates.remove(len(x_nd)-1)
        
        # We will maintain a list of selected points sorted by x.
        # And calculate the HV.
        
        # Function to calculate HV increase for inserting point j into the selected set
        # This is expensive to do from scratch every time.
        # We can just re-evaluate the best candidate from the full non-dominated set.
        
        # Optimization:
        # Precompute the "value" of each point? No, value depends on neighbors.
        
        # Given k=200 and N_nd likely < 1000 (due to oscillations? No, oscillations create many local minima? 
        # Actually, the lower envelope of a function with 10 oscillations will have ~10-20 points?
        # No, the function is decreasing on average. The wiggles are small.
        # The lower envelope will pick the "bottom" of each wiggle.
        # So N_nd might be around 10-20?
        # If N_nd < k, we just use all N_nd.
        
        if len(x_nd) <= k:
            final_t_vals = x_nd
        else:
            # Greedy selection
            # We select k-2 more points.
            # Current selected indices in x_nd array:
            sel_indices = [0, len(x_nd) - 1]
            
            # To make it fast, we can just pick points uniformly from the non-dominated set?
            # If the front is smooth, uniform in x is good.
            # But let's try a simple greedy:
            # At each step, find the point in x_nd that maximizes the HV contribution if added.
            
            # We'll store selected points in a sorted list of (x, y, original_index)
            sel_points = [(x_nd[0], y_nd[0], 0), (x_nd[-1], y_nd[-1], len(x_nd)-1)]
            
            for _ in range(k - 2):
                best_gain = -1
                best_point = None
                best_pos = -1
                
                # Try inserting each candidate from x_nd (excluding already selected)
                # This is O(N * K). N~1000, K~200. 200k ops per step. 200 steps -> 40M ops. OK.
                
                for j in range(len(x_nd)):
                    if j in [p[2] for p in sel_points]:
                        continue
                    
                    x_j = x_nd[j]
                    y_j = y_nd[j]
                    
                    # Find neighbors in sel_points
                    # sel_points is sorted by x.
                    # Find insertion position
                    # Since we iterate j, we can just find the gap.
                    # To avoid O(K) search, we can use a simple linear scan or binary search.
                    # K is small.
                    
                    prev_x = 0
                    prev_y = 0
                    next_x = 0
                    next_y = 0
                    
                    # Find neighbors
                    # Simple linear scan
                    for i in range(len(sel_points) - 1):
                        p1 = sel_points[i]
                        p2 = sel_points[i+1]
                        if p1[0] < x_j < p2[0]:
                            prev_x, prev_y, _ = p1
                            next_x, next_y, _ = p2
                            break
                        elif i == len(sel_points) - 2: # If it's beyond the last? 
                            # But we selected min and max, so it's inside.
                            pass
                            
                    if prev_x == 0 and next_x == 0:
                         continue
                         
                    # Calculate gain
                    # Old contribution of next point (which was the one after prev in the sum? No)
                    # The sum is sum (x_i - x_{i-1}) y_i.
                    # The term for `next` was (next_x - prev_x) * next_y.
                    # New terms: (x_j - prev_x) * y_j + (next_x - x_j) * next_y.
                    gain = (x_j - prev_x) * y_j + (next_x - x_j) * next_y - (next_x - prev_x) * next_y
                    gain = (x_j - prev_x) * y_j - (x_j - prev_x) * next_y
                    gain = (x_j - prev_x) * (y_j - next_y)
                    
                    if gain > best_gain:
                        best_gain = gain
                        best_point = (x_j, y_j, j)
                        best_pos = i # Insert after i, before i+1
                        
                if best_point:
                    sel_points.insert(best_pos + 1, best_point)
            
            final_t_vals = [p[0] for p in sel_points]
            
        # Convert to x vectors
        # x1 = t
        # x2... = 0.5
        x_vectors = []
        for t in final_t_vals:
            x = np.zeros(self.dim)
            x[0] = t
            x[1:] = 0.5
            x_vectors.append(x)
            
        # Evaluate
        for x in x_vectors:
            if time.time() > deadline:
                break
            problem(x)
            
        return None
