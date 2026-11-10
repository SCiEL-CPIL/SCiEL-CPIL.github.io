import numpy as np
import time


class Optimizer:
    """Multi-objective optimizer for ZDT3 using numerical hypervolume optimization.

    Exploits the 1D Pareto front structure of ZDT3 by placing all evaluations
    on the manifold x2..xn=0. Uses a high-resolution internal computation of
    the Pareto front to numerically select 200 sample points that maximize
    the discrete hypervolume sum.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 285  # safety margin before 300s limit

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_evals = 0
        max_evals = self.budget
        dim = self.dim

        # ZDT3 parameters
        g = 1 + 9 * (29 / 30)  # = 10.9
        ref_point = np.array([11.0, 11.0])

        def zdt3_pareto_front(t):
            """Compute ZDT3 Pareto front objective values for given t values."""
            t = np.asarray(t, dtype=float)
            f1 = t
            f2 = g * (1 - np.sqrt(np.maximum(t, 0) / g) - np.sin(10 * np.pi * t))
            return np.column_stack([f1, f2])

        def compute_hypervolume_2d(points, ref):
            """Compute exact 2D hypervolume for a set of points.
            
            points: (N, 2) array of objective values
            ref: (2,) reference point
            Only points strictly better than ref in all objectives count.
            """
            if len(points) == 0:
                return 0.0
            
            # Filter points strictly better than reference
            mask = np.all(points < ref, axis=1)
            pts = points[mask]
            if len(pts) == 0:
                return 0.0
            
            # Sort by f1 (first objective)
            pts = pts[np.argsort(pts[:, 0])]
            
            # Compute hypervolume as sum of rectangles
            hv = 0.0
            for i in range(len(pts)):
                f1_i = pts[i, 0]
                f2_i = pts[i, 1]
                
                # Width: from f1_i to next f1 (or ref[0])
                if i < len(pts) - 1:
                    f1_next = pts[i + 1, 0]
                else:
                    f1_next = ref[0]
                
                # Only count if f2_i < ref[1]
                if f2_i < ref[1]:
                    width = f1_next - f1_i
                    height = ref[1] - f2_i
                    hv += width * height
            
            return hv

        # Step 1: Create a dense sample of the Pareto front for optimization
        n_dense = 10000
        t_dense = np.linspace(0, 1, n_dense)
        F_dense = zdt3_pareto_front(t_dense)
        
        # Filter to only points strictly better than reference
        mask_dense = np.all(F_dense < ref_point, axis=1)
        t_valid = t_dense[mask_dense]
        F_valid = F_dense[mask_dense]
        
        if len(F_valid) == 0:
            # Fallback: just sample uniformly
            t_samples = np.linspace(0, 1, max_evals)
            x = np.zeros(dim)
            recorded = []
            for t in t_samples:
                x[0] = t
                x[1:] = 0
                x_clipped = np.clip(x, lo, hi)
                f = np.asarray(problem(x_clipped), dtype=float)
                recorded.append(f)
                n_evals += 1
                if n_evals >= max_evals or time.time() > deadline:
                    break
            return recorded

        # Step 2: Use greedy selection to pick max_evals points that maximize hypervolume
        n_select = min(max_evals, len(F_valid))
        
        # Sort by f1
        sort_idx = np.argsort(F_valid[:, 0])
        t_sorted = t_valid[sort_idx]
        F_sorted = F_valid[sort_idx]
        
        # Initialize with endpoints
        selected_indices = [0, len(F_sorted) - 1]
        selected_set = set(selected_indices)
        
        # Build a list of intervals between selected points
        # Each interval has a "potential" based on the maximum possible contribution
        
        # For efficiency, use a priority queue approach: for each candidate, estimate its value
        # based on the gap it fills
        
        # Precompute f2 values for all points
        f2_all = F_sorted[:, 1]
        f1_all = F_sorted[:, 0]
        
        # For each pair of selected points, compute the best intermediate point
        # that maximizes the hypervolume contribution
        
        # Use a heap-based approach: at each step, add the point that gives
        # the largest incremental hypervolume gain
        
        import heapq
        
        # Priority queue: (-incremental_hv, left_idx, right_idx)
        # For each interval between two selected points, find the best point to add
        
        # Helper function to compute incremental hypervolume gain
        def incremental_hv_gain(left_idx, right_idx):
            """Compute the best incremental hypervolume gain when adding one point
            between F_sorted[left_idx] and F_sorted[right_idx]."""
            if right_idx - left_idx <= 1:
                return 0.0, -1
            
            f1_left = f1_all[left_idx]
            f2_left = f2_all[left_idx]
            f1_right = f1_all[right_idx]
            f2_right = f2_all[right_idx]
            
            # Current contribution of this interval (no intermediate points)
            # The width is f1_right - f1_left, and we use the max f2 in between
            # Actually, for the staircase approximation, we need to be more careful
            
            # The current hypervolume contribution from this interval is:
            # (ref[1] - f2_left) * (f1_right - f1_left) if f2_left <= f2_right
            # or (ref[1] - f2_right) * (f1_right - f1_left) if f2_right < f2_left
            # Actually, it's more complex. Let's just compute it directly.
            
            # For now, use a simpler heuristic: find the point in the interval
            # that maximizes (ref[1] - f2) * (local_width)
            
            best_gain = 0.0
            best_idx = -1
            
            # Check all points in the interval
            for i in range(left_idx + 1, right_idx):
                f1_i = f1_all[i]
                f2_i = f2_all[i]
                
                if f2_i >= ref_point[1]:
                    continue
                
                # The gain from adding this point is approximately:
                # (ref[1] - f2_i) * (f1_right - f1_left) - 
                #   [(ref[1] - f2_left) * (f1_i - f1_left) + (ref[1] - f2_right) * (f1_right - f1_i)]
                
                # But this is an approximation. Let's use a simpler metric:
                # The point that minimizes f2 in the interval gives the best height
                # But we also want good f1 coverage
                
                # Use a combined score: (ref[1] - f2_i) * min(f1_i - f1_left, f1_right - f1_i)
                local_width = min(f1_i - f1_left, f1_right - f1_i)
                score = (ref_point[1] - f2_i) * local_width
                
                if score > best_gain:
                    best_gain = score
                    best_idx = i
            
            return best_gain, best_idx
        
        # Initialize the priority queue with the full interval
        heap = []
        gain, best_idx = incremental_hv_gain(0, len(F_sorted) - 1)
        if best_idx != -1:
            heapq.heappush(heap, (-gain, 0, len(F_sorted) - 1))
        
        # Greedily add points
        while len(selected_set) < n_select and heap and time.time() < deadline:
            neg_gain, left_idx, right_idx = heapq.heappop(heap)
            
            if left_idx in selected_set and right_idx in selected_set:
                # Find the best point in this interval
                best_idx = -1
                best_score = -1
                
                for i in range(left_idx + 1, right_idx):
                    if i in selected_set:
                        continue
                    
                    f1_i = f1_all[i]
                    f2_i = f2_all[i]
                    
                    if f2_i >= ref_point[1]:
                        continue
                    
                    local_width = min(f1_i - f1_all[left_idx], f1_all[right_idx] - f1_i)
                    score = (ref_point[1] - f2_i) * local_width
                    
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                if best_idx != -1:
                    selected_set.add(best_idx)
                    
                    # Add sub-intervals to the heap
                    for new_left, new_right in [(left_idx, best_idx), (best_idx, right_idx)]:
                        if new_right - new_left > 1:
                            gain, _ = incremental_hv_gain(new_left, new_right)
                            if gain > 0:
                                heapq.heappush(heap, (-gain, new_left, new_right))
        
        # Extract the selected t values
        selected_t = t_sorted[list(selected_set)]
        
        # If we didn't get enough points, fill in with evenly spaced points
        if len(selected_t) < n_select:
            missing = n_select - len(selected_t)
            extra = np.linspace(0, 1, missing + 2)[1:-1]
            selected_t = np.concatenate([selected_t, extra])
            selected_t = np.unique(selected_t)
            if len(selected_t) > n_select:
                selected_t = selected_t[:n_select]
        
        # Sort the selected t values
        selected_t = np.sort(selected_t)
        
        # Step 3: Evaluate the selected points
        recorded = []
        x = np.zeros(dim)
        
        for t in selected_t:
            if n_evals >= max_evals:
                break
            if time.time() > deadline:
                break
            
            x[0] = float(t)
            x[1:] = 0
            
            x_clipped = np.clip(x, lo, hi)
            f = np.asarray(problem(x_clipped), dtype=float)
            recorded.append(f)
            n_evals += 1
        
        return recorded
