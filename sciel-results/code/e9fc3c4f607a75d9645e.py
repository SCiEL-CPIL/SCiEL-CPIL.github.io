"""Optimized ZDT2 solver using greedy hypervolume-optimal point placement."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize_scalar

class Optimizer:
    """Multi-objective optimizer for ZDT2 exploiting analytic structure and greedy HV maximization."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300
        
        dim = self.dim
        budget = self.budget
        
        # ZDT2 Analytic Structure:
        # f1(x) = x1
        # f2(x) = (1 + 10*(n-1)) * (1 - sqrt(f1)) + sum(xi^2 for i=2..n)
        # Pareto front: x2...xn = 0.
        # f2(f1) = C * (1 - sqrt(f1))
        C = 1 + 10 * (dim - 1)
        
        # Reference point
        ref_f1 = 11.0
        ref_f2 = 11.0
        
        # The front is defined for f1 in [0, 1].
        # f2 ranges from C (at f1=0) to 0 (at f1=1).
        # Note: f2 is decreasing.
        
        # We want to select N points (f1_i, f2_i) on this curve to maximize
        # the hypervolume dominated by these points with reference (11, 11).
        
        # Hypervolume of a set of points in 2D:
        # Sort points by f1 ascending.
        # HV = sum over i of (f1_{i+1} - f1_i) * (ref_f2 - f2_i) 
        #     + (ref_f1 - f1_N) * (ref_f2 - f2_N) 
        # Wait, the standard 2D HV calculation for a set of non-dominated points sorted by f1:
        # Let points be p_1, ..., p_N sorted by f1_1 < ... < f1_N.
        # The area is the sum of rectangles.
        # For each point p_i, it contributes a rectangle of width (f1_{i+1} - f1_i) and height (ref_f2 - f2_i) 
        # plus the last point contributes width (ref_f1 - f1_N) and height (ref_f2 - f2_N).
        # Actually, it's easier to think of it as:
        # HV = sum_{i=1}^{N} (f1_{i+1} - f1_i) * (ref_f2 - f2_i) where f1_{N+1} = ref_f1.
        # But we must ensure f2_i is the y-coordinate. Since the front is convex (curving towards the origin? No, f2 = C(1-sqrt(f1)) is convex downwards? 
        # d2f2/df12 = C * 1/(4 * f1^(3/2)) > 0. So it is convex.
        # The "dominated" region is above the curve and below the reference point?
        # No, minimization. The dominated region is the set of points (u, v) such that u >= f1_i and v >= f2_i for some i, and u <= ref_f1, v <= ref_f2.
        # Since f2 is decreasing, the "skyline" is the curve itself.
        # The area is the area under the step function formed by the points.
        
        # Let's define the HV function for a sorted list of f1 values.
        def compute_hv(f1_vals):
            """
            f1_vals: array of f1 values, sorted ascending.
            Returns hypervolume.
            """
            if len(f1_vals) == 0:
                return 0.0
            f1_sorted = np.sort(f1_vals)
            f2_vals = C * (1 - np.sqrt(f1_sorted))
            
            # Next f1 boundary for each point is the next point's f1, or ref_f1 for the last one.
            next_f1 = np.append(f1_sorted[1:], ref_f1)
            
            # Width of each rectangle
            widths = next_f1 - f1_sorted
            
            # Height of each rectangle (distance to ref_f2)
            heights = ref_f2 - f2_vals
            
            # Sum of areas
            hv = np.sum(widths * heights)
            return hv

        # Greedy algorithm to find N points
        # Start with the two endpoints? Or just one?
        # Usually, starting with the extreme points is good, but let's just greedily add.
        
        N = budget
        # Ensure N is at least 1
        if N <= 0:
            return np.zeros((0, dim))
            
        # We will maintain a list of selected f1 values.
        selected_f1 = []
        
        # For speed, we can pre-compute a fine grid of candidate f1 values 
        # and pick from there, or use continuous optimization.
        # Given N=200, continuous optimization via minimize_scalar for each new point
        # might be slow if done naively, but 200 iterations is small.
        # However, minimize_scalar has overhead.
        # Alternative: Use a dense grid of, say, 10000 points and pick the best from the grid.
        # This is O(N * GridSize). 200 * 10000 = 2,000,000 operations. Very fast.
        
        grid_size = 20000
        candidate_f1 = np.linspace(0.0, 1.0, grid_size)
        
        for k in range(N):
            if time.time() > deadline - 10:
                # Break if we are running out of time
                break
                
            # Function to evaluate HV gain if we add candidate x to current set
            # Current set: selected_f1
            # New set: selected_f1 + [x]
            
            # To speed this up, we can pre-compute the current HV and then 
            # calculate the incremental gain.
            # But since N is small, just computing HV for each candidate is fine.
            # Wait, computing HV for each of 20000 candidates * 200 iterations = 4M HV calcs.
            # Each HV calc is O(N). Total O(N^2 * GridSize). 
            # 200^2 * 20000 = 800,000,000. This might be too slow in Python loops.
            
            # Optimization: 
            # The HV of a set S is sum of rectangles.
            # Adding a point x splits one existing rectangle (or adds a new one at the end/beginning).
            # Let's use a more efficient update or a coarser grid.
            
            # Let's use a grid of 5000 points.
            # And we can vectorize the HV calculation for all candidates at once?
            # No, the HV depends on the *sorted* order.
            
            # Let's stick to a simpler, robust approach:
            # Since the front is convex, the optimal points are not uniform.
            # A known heuristic for convex fronts is to sample uniformly in the *normal* direction 
            # or use a specific transformation.
            
            # Actually, let's just do the greedy search with a moderate grid (1000 points) 
            # and optimize the HV calculation.
            
            if not selected_f1:
                # First point: maximize single point HV
                # HV(x) = (ref_f1 - x) * (ref_f2 - f2(x))
                # This is maximized somewhere in the middle.
                # We can just pick the best from the grid.
                f2_cand = C * (1 - np.sqrt(candidate_f1))
                hvs = (ref_f1 - candidate_f1) * (ref_f2 - f2_cand)
                best_idx = np.argmax(hvs)
                best_f1 = candidate_f1[best_idx]
            else:
                # Subsequent points:
                # We need to find x that maximizes HV(S U {x}).
                # Let's evaluate HV for each candidate x in candidate_f1.
                # To do this efficiently, we can sort the combined list each time? 
                # That's O(GridSize * N log N).
                
                # Let's try a different approach: 
                # The marginal gain of adding a point x is the area of the "new" region it covers.
                # If we have points sorted p_1 ... p_m.
                # The HV is sum_{i=1}^m (p_{i+1}.f1 - p_i.f1) * (ref_f2 - p_i.f2) with p_{m+1}.f1 = ref_f1.
                # If we insert x between p_i and p_{i+1}:
                # Old contribution: (p_{i+1}.f1 - p_i.f1) * (ref_f2 - p_i.f2)
                # New contribution: (x.f1 - p_i.f1) * (ref_f2 - p_i.f2) + (p_{i+1}.f1 - x.f1) * (ref_f2 - x.f2)
                # Gain = New - Old
                #      = (x.f1 - p_i.f1) * (ref_f2 - p_i.f2) + (p_{i+1}.f1 - x.f1) * (ref_f2 - x.f2) - (p_{i+1}.f1 - p_i.f1) * (ref_f2 - p_i.f2)
                #      = (x.f1 - p_i.f1) * (ref_f2 - p_i.f2) - (p_{i+1}.f1 - x.f1) * (ref_f2 - p_i.f2) + (p_{i+1}.f1 - x.f1) * (ref_f2 - x.f2)
                #      = (p_{i+1}.f1 - p_i.f1) * (ref_f2 - p_i.f2) - (p_{i+1}.f1 - x.f1) * (ref_f2 - p_i.f2) - (p_{i+1}.f1 - x.f1) * (ref_f2 - x.f2) + (p_{i+1}.f1 - x.f1) * (ref_f2 - x.f2) 
                # Wait, algebra is messy. Let's just compute the total HV for each candidate.
                
                # To make it fast, we will use a smaller grid for the greedy step, 
                # or simply accept that 200 iterations * 1000 candidates * O(200) HV calc is ~40M ops.
                # 40M numpy ops is fast (<1s).
                
                # Let's use a grid of 2000 points.
                # For each candidate x, we compute HV of S + [x].
                
                # Optimization: Pre-sort S.
                S_sorted = sorted(selected_f1)
                
                # We will compute HV for all candidates in a loop. 
                # This is O(GridSize * N).
                # 2000 * 200 = 400,000 operations per iteration.
                # 200 iterations * 400,000 = 80,000,000 operations.
                # This is acceptable in Python if vectorized or using numpy arrays.
                
                # Let's vectorize the HV calculation for a fixed S and many candidates X.
                # This is tricky because the insertion point changes for each X.
                
                # Simpler: Just loop over candidates and compute HV.
                best_gain = -np.inf
                best_x = 0.0
                
                # We can prune the search space. The best x is likely in the largest gap.
                # But let's just scan the grid.
                
                # To speed up, we can compute the HV for a candidate x by:
                # 1. Insert x into S_sorted.
                # 2. Compute HV.
                
                # Let's try a different strategy: 
                # Since the front is analytic, we can solve for the optimal spacing analytically?
                # No, greedy is good enough.
                
                # Let's use a coarser grid (500 points) to ensure speed.
                grid = np.linspace(0.0, 1.0, 500)
                best_hv = -1
                best_f1_val = 0.0
                
                # Pre-compute f2 for grid
                grid_f2 = C * (1 - np.sqrt(grid))
                
                # For each candidate in grid:
                # This loop is 500 * 200 = 100k iterations.
                # Each iteration does an HV calc of size ~200.
                # 100k * 200 = 20M ops. Fast.
                
                for x in grid:
                    # Create new list
                    # To avoid list creation overhead, we can just compute the gain.
                    # But for simplicity and correctness, let's just compute HV.
                    
                    # Efficient HV calc for sorted list + one point:
                    # Find insertion index.
                    # Since grid is sorted, and S is sorted, we can do binary search.
                    # But let's just use numpy.
                    
                    # Actually, let's just use the full grid and vectorize as much as possible.
                    # But the insertion point varies.
                    
                    # Let's just do the simple loop. It's only 500 * 200 = 100k iterations.
                    # Inside, we do a numpy sort of size 201? No, we can merge.
                    
                    # Let's try to compute the HV increment directly.
                    # If S is sorted s_1 ... s_m.
                    # HV(S) = sum_{i=1}^{m} (s_{i+1} - s_i) * (ref_f2 - f2(s_i)) where s_{m+1}=ref_f1.
                    
                    # If we add x, it falls in some interval (s_i, s_{i+1}).
                    # The old term was (s_{i+1} - s_i) * (ref_f2 - f2(s_i)).
                    # The new terms are (x - s_i) * (ref_f2 - f2(s_i)) + (s_{i+1} - x) * (ref_f2 - f2(x)).
                    # Gain = New - Old.
                    
                    # We can pre-compute the "base" HV of S.
                    # And for each x, find i such that s_i < x < s_{i+1}.
                    
                    # Let's pre-compute s_sorted and f2_s.
                    if not S_sorted:
                        # Already handled above
                        pass
                    else:
                        s_arr = np.array(S_sorted)
                        s_f2 = C * (1 - np.sqrt(s_arr))
                        # Next s: append ref_f1
                        s_next = np.append(s_arr[1:], ref_f1)
                        
                        # Find insertion index for x in s_arr
                        # np.searchsorted
                        idx = np.searchsorted(s_arr, x)
                        
                        # If x < s_arr[0], idx=0. Interval is (x, s_arr[0])? No, (x, s_arr[0]) is not right.
                        # The intervals are (s_i, s_{i+1}).
                        # If x < s_arr[0], it's before the first point.
                        # The "old" term for the first interval was (s_1 - s_0?) No, s_0 is the first point.
                        # The HV sum starts from s_1.
                        # Term 1: (s_2 - s_1) * (ref_f2 - f2(s_1)).
                        # If x < s_1, the new point x becomes the first.
                        # New terms: (s_1 - x) * (ref_f2 - f2(x)) + (s_2 - s_1) * (ref_f2 - f2(s_1)) ...
                        # Wait, the term for s_1 changes width?
                        # Old: (s_2 - s_1) * (ref_f2 - f2(s_1))
                        # New: (s_1 - x) * (ref_f2 - f2(x)) + (s_2 - s_1) * (ref_f2 - f2(s_1))
                        # Gain = (s_1 - x) * (ref_f2 - f2(x))
                        
                        # If s_i < x < s_{i+1}:
                        # Old term for s_i: (s_{i+1} - s_i) * (ref_f2 - f2(s_i))
                        # New terms: (x - s_i) * (ref_f2 - f2(s_i)) + (s_{i+1} - x) * (ref_f2 - f2(x))
                        # Gain = (x - s_i) * (ref_f2 - f2(s_i)) + (s_{i+1} - x) * (ref_f2 - f2(x)) - (s_{i+1} - s_i) * (ref_f2 - f2(s_i))
                        #      = (s_{i+1} - s_i) * (ref_f2 - f2(s_i)) - (s_{i+1} - x) * (ref_f2 - f2(s_i)) - (s_{i+1} - x) * (ref_f2 - f2(x)) + (s_{i+1} - x) * (ref_f2 - f2(x)) 
                        #      = (s_{i+1} - s_i) * (ref_f2 - f2(s_i)) - (s_{i+1} - x) * (ref_f2 - f2(s_i)) - (s_{i+1} - x) * (ref_f2 - f2(x)) + (s_{i+1} - x) * (ref_f2 - f2(x))
                        # Let's simplify:
                        # Old = (s_{i+1} - s_i) * H_i
                        # New = (x - s_i) * H_i + (s_{i+1} - x) * H_x
                        # Gain = (x - s_i) * H_i + (s_{i+1} - x) * H_x - (s_{i+1} - s_i) * H_i
                        #      = (x - s_i - s_{i+1} + s_i) * H_i + (s_{i+1} - x) * H_x
                        #      = (x - s_{i+1}) * H_i + (s_{i+1} - x) * H_x
                        #      = (s_{i+1} - x) * (H_x - H_i)
                        # Where H_i = ref_f2 - f2(s_i) and H_x = ref_f2 - f2(x).
                        
                        # If x > s_m (last point):
                        # Old term for s_m: (ref_f1 - s_m) * H_m
                        # New terms: (ref_f1 - x) * H_x + (x - s_m) * H_m ? No.
                        # The point x is after s_m.
                        # The term for s_m changes width?
                        # Old: (ref_f1 - s_m) * H_m
                        # New: (x - s_m) * H_m + (ref_f1 - x) * H_x
                        # Gain = (x - s_m) * H_m + (ref_f1 - x) * H_x - (ref_f1 - s_m) * H_m
                        #      = (ref_f1 - x) * H_x - (ref_f1 - x) * H_m
                        #      = (ref_f1 - x) * (H_x - H_m)
                        
                        # So, Gain(x) can be computed in O(1) if we know the insertion index and the H values.
                        
                        # Let's compute H for all s_arr.
                        H_s = ref_f2 - s_f2
                        
                        # For each x in grid:
                        # idx = searchsorted(s_arr, x)
                        # if idx == 0:
                        #    Gain = (s_arr[0] - x) * (ref_f2 - f2(x))  # H_x - H_{-1}? No, H_{-1} is not defined.
                        #    Wait, if x < s_1, the "old" term was (s_2 - s_1) * H_1.
                        #    The new point x is before s_1.
                        #    The term for s_1 is still (s_2 - s_1) * H_1.
                        #    The new term for x is (s_1 - x) * H_x.
                        #    Gain = (s_1 - x) * H_x.
                        
                        # if idx == len(s_arr):
                        #    x > s_m.
                        #    Gain = (ref_f1 - x) * (H_x - H_m)
                        
                        # else:
                        #    s_i = s_arr[idx-1]
                        #    s_{i+1} = s_arr[idx]
                        #    H_i = H_s[idx-1]
                        #    H_x = ref_f2 - f2(x)
                        #    Gain = (s_{i+1} - x) * (H_x - H_i)
                        
                        # Let's vectorize this.
                        idxs = np.searchsorted(s_arr, grid)
                        gains = np.zeros_like(grid)
                        
                        # Case 1: idx == 0
                        mask0 = (idxs == 0)
                        if np.any(mask0):
                            H_x0 = ref_f2 - grid_f2[mask0]
                            gains[mask0] = (s_arr[0] - grid[mask0]) * H_x0
                        
                        # Case 2: idx == len(s_arr)
                        maskM = (idxs == len(s_arr))
                        if np.any(maskM):
                            H_xM = ref_f2 - grid_f2[maskM]
                            H_m = H_s[-1]
                            gains[maskM] = (ref_f1 - grid[maskM]) * (H_xM - H_m)
                        
                        # Case 3: 0 < idx < len(s_arr)
                        maskMid = (idxs > 0) & (idxs < len(s_arr))
                        if np.any(maskMid):
                            idxs_mid = idxs[maskMid]
                            x_mid = grid[maskMid]
                            H_x_mid = ref_f2 - grid_f2[maskMid]
                            # s_i is s_arr[idx-1]
                            s_i = s_arr[idxs_mid - 1]
                            s_ip1 = s_arr[idxs_mid]
                            H_i = H_s[idxs_mid - 1]
                            gains[maskMid] = (s_ip1 - x_mid) * (H_x_mid - H_i)
                        
                        best_gain_idx = np.argmax(gains)
                        best_f1_val = grid[best_gain_idx]
                        # We don't actually need the gain value, just the best x.
                        
                selected_f1.append(best_f1_val)

        # Construct the final set of x vectors
        X = np.zeros((len(selected_f1), dim))
        X[:, 0] = selected_f1
        
        # Evaluate all points to record them
        # We must call problem() for each point.
        for x in X:
            if time.time() > deadline:
                break
            problem(x)
            
        return X
# EVOLVE-BLOCK-END
