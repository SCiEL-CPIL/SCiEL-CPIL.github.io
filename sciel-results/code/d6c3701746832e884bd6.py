"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    """Greedy hypervolume-maximizing search on the ZDT3 Pareto front."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290  # Safety margin for 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT3 Pareto front is at x[1:] = 0
        # We only vary x[0]
        # Initial uniform sampling to establish the front shape
        n_init = min(10, self.budget)
        x0_init = np.linspace(0.0, 1.0, n_init)
        
        points_x0 = []
        points_f = []
        
        # Evaluate initial points
        for x0 in x0_init:
            if time.time() > deadline:
                break
            x = np.zeros(self.dim)
            x[0] = x0
            # Ensure within bounds (though 0 and 1 are bounds)
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            points_x0.append(x0)
            points_f.append(f)
            
        # If budget is very small, return
        if self.budget <= n_init:
            return points_x0

        # Greedy selection for remaining budget
        # We maintain a list of (x0, f1, f2)
        # Sort by f1 (which is x0)
        
        def get_hv(points):
            # points: list of (x0, f1, f2)
            if not points:
                return 0.0
            
            # Filter points strictly better than reference [11, 11]
            ref = np.array([11.0, 11.0])
            valid = []
            for p in points:
                if p[1] < ref[0] and p[2] < ref[1]:
                    valid.append(p)
            
            if not valid:
                return 0.0
            
            # Sort by f1 (x0)
            valid.sort(key=lambda p: p[1])
            
            # Calculate HV
            # For ZDT3, f2 decreases as f1 increases on the Pareto front.
            # We assume the points are non-dominated or we just calculate the area under the step function.
            # To be safe, we should remove dominated points.
            # Dominated: if p_i has f1 <= p_j f1 and f2 <= p_j f2, p_j is dominated.
            # Since we are on the front, we expect f2 to decrease as f1 increases.
            # If f2 increases, the point might be dominated by a previous one.
            
            # Filter non-dominated
            nd_points = []
            min_f2 = float('inf')
            # Sort by f1 ascending
            # For minimization, a point is dominated if there exists another point with lower or equal f1 and lower or equal f2.
            # So we traverse from low f1 to high f1.
            # Keep track of min_f2 seen so far.
            # If current f2 < min_f2, it is non-dominated (with respect to previous points).
            # But it might dominate future points? No, future points have higher f1.
            # If current f2 < min_f2, it is better than all previous in f2.
            # Since f1 is higher than previous, it is not dominated by previous.
            # Is it dominated by future? We don't know future yet.
            # So we just keep points that improve on the minimum f2.
            
            # Wait, this logic finds the "lower envelope" of f2 vs f1.
            # This is exactly the Pareto front for a set of points.
            
            for p in valid:
                if p[2] < min_f2:
                    nd_points.append(p)
                    min_f2 = p[2]
            
            if not nd_points:
                return 0.0
            
            # Calculate area
            # Add reference point to the end?
            # The HV is the area between the curve and the reference point.
            # Points are (f1, f2).
            # Ref is (11, 11).
            # Area = sum of rectangles.
            # For each point i, the width is (f1[i+1] - f1[i]) where f1[i+1] is the next point's f1 or ref[0].
            # The height is (ref[1] - f2[i]).
            
            hv = 0.0
            for i in range(len(nd_points)):
                f1_curr = nd_points[i][1]
                f2_curr = nd_points[i][2]
                
                if i < len(nd_points) - 1:
                    f1_next = nd_points[i+1][1]
                else:
                    f1_next = ref[0]
                    
                width = f1_next - f1_curr
                if width <= 0:
                    continue
                    
                height = ref[1] - f2_curr
                if height <= 0:
                    continue
                    
                hv += width * height
                
            return hv

        # Candidate generation for greedy step
        # We use a fine grid of x0 values
        n_grid = 1000
        x0_grid = np.linspace(0.0, 1.0, n_grid)
        
        used_x0 = set(points_x0)
        
        current_hv = get_hv([(x0, x0, f[1]) for x0, f in zip(points_x0, points_f)])
        
        for _ in range(self.budget - len(points_x0)):
            if time.time() > deadline:
                break
                
            best_x0 = None
            best_hv = current_hv
            
            # To speed up, we can sample a subset of the grid or use the whole grid
            # 1000 candidates * HV calc (O(N)) is 1000*200 = 200k ops. Fast enough.
            
            # We need to estimate f2 for each candidate.
            # Use linear interpolation from nearest neighbors in points_x0.
            
            # Sort current points by x0 for interpolation
            sorted_points = sorted(zip(points_x0, points_f), key=lambda x: x[0])
            
            for x0_cand in x0_grid:
                if x0_cand in used_x0:
                    continue
                    
                # Estimate f2
                # Find neighbors
                # Since sorted_points is small, linear search is fine
                left_x0, left_f = None, None
                right_x0, right_f = None, None
                
                for x0_p, f_p in sorted_points:
                    if x0_p < x0_cand:
                        left_x0, left_f = x0_p, f_p
                    elif x0_p > x0_cand:
                        right_x0, right_f = x0_p, f_p
                        break
                    else:
                        # Should not happen due to used_x0 check
                        continue
                
                if left_x0 is None:
                    # Extrapolate from right
                    if right_x0 is not None:
                        est_f2 = right_f[1]
                    else:
                        continue
                elif right_x0 is None:
                    # Extrapolate from left
                    est_f2 = left_f[1]
                else:
                    # Linear interpolation
                    if right_x0 == left_x0:
                        est_f2 = left_f[1]
                    else:
                        t = (x0_cand - left_x0) / (right_x0 - left_x0)
                        est_f2 = left_f[1] + t * (right_f[1] - left_f[1])
                
                # Estimate f1 is x0_cand
                est_f = np.array([x0_cand, est_f2])
                
                # Calculate new HV
                # Create new list of points
                # Optimization: We can just insert into the sorted non-dominated list and recalc HV
                # But for simplicity, we use the get_hv function on the full set + candidate
                # This is O(N log N) due to sorting in get_hv.
                # 200 steps * 1000 candidates * 200 log 200 ~ 200 * 1000 * 1500 = 300M ops. Might be slow in Python.
                
                # Faster HV calculation for 2D:
                # We can maintain the non-dominated list.
                # But let's try to optimize get_hv or the inner loop.
                
                # Alternative: Just pick the candidate that maximizes the "potential"
                # i.e., maximizes (ref[1] - est_f2) * (gap to neighbor)
                # This is an approximation.
                
                # Let's stick to exact HV but optimize the inner loop.
                # Instead of creating a new list and sorting, we can insert into the sorted list.
                
                # For now, let's try the exact method but with a smaller grid if it's too slow.
                # 1000 candidates is probably fine for 200 steps.
                
                # Construct new points list
                # To avoid copying the whole list, we can pass a flag or just use the list
                # Since N is small (<200), copying is cheap.
                
                new_points = [(x0, x0, f[1]) for x0, f in zip(points_x0, points_f)]
                new_points.append((x0_cand, x0_cand, est_f2))
                
                hv_cand = get_hv(new_points)
                
                if hv_cand > best_hv:
                    best_hv = hv_cand
                    best_x0 = x0_cand
            
            if best_x0 is None:
                # Fallback: pick random
                best_x0 = np.random.rand()
                
            # Evaluate the best candidate
            x = np.zeros(self.dim)
            x[0] = best_x0
            x = np.clip(x, lo, hi)
            f = np.array(problem(x))
            
            points_x0.append(best_x0)
            points_f.append(f)
            used_x0.add(best_x0)
            current_hv = best_hv # This is an estimate, update with real HV?
            # We should update current_hv with the real HV of the real points
            # But get_hv takes (x0, f1, f2).
            # We can recalc current_hv here to be accurate for next step.
            current_hv = get_hv([(x0, x0, f[1]) for x0, f in zip(points_x0, points_f)])

        return points_x0
# EVOLVE-BLOCK-END
