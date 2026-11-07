"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from functools import lru_cache

class Optimizer:
    """
    Multi-objective optimizer using Dynamic Programming to maximize hypervolume.
    
    Strategy:
    1. Identify the Pareto front of ZDT4: f1 = x0, f2 = 1 + (1 - sqrt(x0/g)) * g, 
       with g minimized to 1 (by setting x1...x9 = 0).
       This gives f2(x0) = 1 + (1 - sqrt(x0)) * 1 = 2 - sqrt(x0).
    2. The problem reduces to selecting K points x0_1 < x0_2 < ... < x0_K from [0, 1] 
       to maximize the hypervolume of the resulting set of points (x0_i, 2 - sqrt(x0_i)) 
       against reference point (11, 400).
    3. Since the front is convex and monotonic, the hypervolume contribution of a set of 
       points can be decomposed. We use a discrete DP approach.
       
       Let's discretize the x0 domain [0, 1] into M bins. 
       Actually, a more direct approach for convex fronts:
       The hypervolume of a set of points P = {(f1_i, f2_i)} sorted by f1 ascending 
       is sum over i of (f1_{i+1} - f1_i) * (f2_i - ref2) + (ref1 - f1_K) * (f2_K - ref2)
       Wait, standard HV calculation for sorted points (f1 asc, f2 desc):
       HV = sum_{i=1}^K (f1_{i+1} - f1_i) * (f2_i - ref_y) 
           + (ref_x - f1_K) * (f2_K - ref_y)
           + f1_1 * (f2_1 - ref_y)  ? No.
       
       Let's derive carefully.
       Points: p_1, ..., p_K. f1_1 < ... < f1_K. f2_1 > ... > f2_K.
       Reference R = (ref_x, ref_y).
       The area is the union of rectangles [f1_i, f1_{i+1}] x [f2_i, ref_y] ? No.
       The dominated region is { (x,y) | x < f1_i, y < f2_i for some i }.
       Since f1 is increasing and f2 is decreasing, the "staircase" is:
       For x in [0, f1_1], the max y is f2_1.
       For x in [f1_1, f1_2], the max y is f2_2.
       ...
       For x in [f1_{K-1}, f1_K], the max y is f2_K.
       For x in [f1_K, ref_x], the max y is ref_y (if f2_K > ref_y, then it's f2_K? No, 
       the region is bounded by the points. If we include the reference point, 
       the area is sum of rectangles.
       
       Standard formula for HV of points P with ref R:
       Sort P by f1 asc.
       HV = sum_{i=1}^K (f1_{i+1} - f1_i) * (f2_i - ref_y)  where f1_{K+1} = ref_x
             BUT this assumes f2_i > ref_y for all i.
             Also, for the first interval, it starts from 0? 
             If f1_1 > 0, the area from 0 to f1_1 is f1_1 * (f2_1 - ref_y).
             So, let f1_0 = 0.
             HV = sum_{i=1}^K (f1_i - f1_{i-1}) * (f2_i - ref_y) + (ref_x - f1_K) * (f2_K - ref_y)
             Wait, the term (ref_x - f1_K) * (f2_K - ref_y) is the last part.
             Let's check i=K: (f1_K - f1_{K-1}) * (f2_K - ref_y) + (ref_x - f1_K) * (f2_K - ref_y)
             = (ref_x - f1_{K-1}) * (f2_K - ref_y).
             This matches the rectangle from f1_{K-1} to ref_x with height f2_K - ref_y.
             
             So HV = sum_{i=1}^K (f1_i - f1_{i-1}) * (f2_i - ref_y) + (ref_x - f1_K) * (f2_K - ref_y)
             where f1_0 = 0.
             This can be rewritten as:
             HV = f1_1(f2_1 - ref_y) + sum_{i=2}^K f1_i(f2_i - ref_y) - f1_{i-1}(f2_i - ref_y) ... no.
             
             Let's use the property that we want to maximize HV.
             HV = (ref_x - f1_K)(f2_K - ref_y) + sum_{i=1}^{K-1} (f1_{i+1} - f1_i)(f2_i - ref_y)
                 + f1_1(f2_1 - ref_y) ? 
                 Let's re-verify the first term.
                 Area under staircase:
                 Integral from 0 to ref_x of min(max_y(x), ref_y) dx ? No, it's the area 
                 dominated by the points and bounded by ref.
                 The set is { (x,y) | x <= f1_i, y <= f2_i for some i } intersected with [0,ref_x]x[0,ref_y].
                 Since f1_i < ref_x and f2_i < ref_y (usually), the boundary is the staircase.
                 Height at x:
                 if x < f1_1: h = f2_1
                 if f1_1 <= x < f1_2: h = f2_2
                 ...
                 if f1_{K-1} <= x < f1_K: h = f2_K
                 if f1_K <= x < ref_x: h = f2_K  (Wait, no point dominates x > f1_K with y < f2_K? 
                 Actually, p_K dominates (x, y) if x <= f1_K and y <= f2_K.
                 So for x in [f1_K, ref_x], the max y is f2_K? No, only if x <= f1_K.
                 If x > f1_K, NO point dominates x. 
                 So the dominated region ends at f1_K in x?
                 No, the reference point defines the box. 
                 The union of boxes [0, f1_i] x [0, f2_i] is the dominated region.
                 Since f1 is increasing, [0, f1_K] contains all [0, f1_i].
                 So the x-range of the dominated region is [0, f1_K].
                 BUT the hypervolume is usually defined against a reference point, 
                 meaning we consider the volume of the box [0, ref_x] x [0, ref_y] 
                 MINUS the volume NOT dominated.
                 Or equivalently, the volume of the union of boxes [0, f1_i] x [0, f2_i] 
                 is NOT the HV if f1_K < ref_x.
                 The standard definition: HV(P, R) = Volume( { x in R^m | exists p in P s.t. p <= x } ) 
                 intersected with [0, R]? No.
                 It is Volume( { x | x >= p for some p } )? No, minimization.
                 x dominates p if x_i <= p_i.
                 The region dominated by P is D = { x | exists p in P, x <= p }.
                 This region is unbounded in the negative direction.
                 We intersect with [0, R]? No, usually [ref, inf)?
                 In minimization, we look at the region "above and to the right" of P, 
                 bounded by R?
                 Let's stick to the standard formula for 2D minimization:
                 Sort P by f1 asc.
                 HV = sum_{i=1}^K (f1_{i+1} - f1_i) * (f2_i - ref_y) 
                      where f1_{K+1} = ref_x, and f2_{K+1} is not used?
                      Wait, if f1_K < ref_x, the last rectangle is [f1_K, ref_x] x [ref_y, f2_K]?
                      No, the region is { (x,y) | x >= f1_i, y >= f2_i }? No.
                      
                      Let's use the complement.
                      Total Box: [0, ref_x] x [0, ref_y].
                      Undominated region is the set of points in the box that are NOT dominated by any p in P.
                      A point (x,y) is dominated by p if x <= f1_p AND y <= f2_p.
                      So (x,y) is NOT dominated if for all p, x > f1_p OR y > f2_p.
                      This is the region above the staircase.
                      The staircase is defined by the upper envelope of the boxes [0, f1_i] x [0, f2_i].
                      The boundary is:
                      y = f2_1 for x in [0, f1_1]
                      y = f2_2 for x in [f1_1, f1_2]
                      ...
                      y = f2_K for x in [f1_{K-1}, f1_K]
                      y = 0 for x in [f1_K, ref_x] ? No, the box ends at ref_y.
                      
                      Actually, the "undominated" region in the box [0, ref_x]x[0, ref_y] is:
                      Area = Integral_0^ref_x (ref_y - y_boundary(x)) dx
                      where y_boundary(x) = max { f2_i | f1_i >= x }.
                      If x > f1_K, then no f1_i >= x, so max is -inf? 
                      But the box is bounded by ref_y. 
                      If no point dominates x, then the entire vertical segment [0, ref_y] is undominated?
                      No, if x > f1_K, then for any y, x > f1_K >= f1_i, so x is not <= f1_i.
                      So (x,y) is not dominated by any p.
                      So the undominated region includes [f1_K, ref_x] x [0, ref_y].
                      So the dominated region is [0, f1_K] x [0, y_boundary(x)].
                      HV = Area of dominated region = Integral_0^{f1_K} y_boundary(x) dx.
                      Wait, this implies points with large f1_K are better?
                      But f2 decreases as f1 increases.
                      y_boundary(x) = f2_i for x in [f1_{i-1}, f1_i].
                      So HV = sum_{i=1}^K (f1_i - f1_{i-1}) * f2_i.
                      This formula does NOT depend on ref_x or ref_y, as long as ref_x > f1_K and ref_y > f2_1.
                      Is this correct?
                      Let's check. If I have point (1, 1) and ref (2, 2).
                      Dominated region: [0, 1] x [0, 1]. Area = 1.
                      Formula: f1_1=1, f2_1=1. Sum = (1-0)*1 = 1. Correct.
                      
                      If I have points (0.5, 1.5) and (1, 1). Ref (2, 2).
                      y_boundary(x):
                      x in [0, 0.5]: max f2_i s.t. f1_i >= x. 
                      f1_1=0.5, f1_2=1.
                      For x=0.2, f1_1>=0.2, f1_2>=0.2. Max f2 is 1.5.
                      For x=0.8, f1_1<0.8, f1_2>=0.8. Max f2 is 1.
                      So y_boundary is 1.5 on [0, 0.5] and 1.0 on [0.5, 1].
                      HV = 0.5*1.5 + 0.5*1.0 = 0.75 + 0.5 = 1.25.
                      
                      Let's check manually.
                      Union of boxes:
                      Box 1: [0, 0.5] x [0, 1.5]
                      Box 2: [0, 1.0] x [0, 1.0]
                      Union:
                      x in [0, 0.5]: height 1.5. Area = 0.75.
                      x in [0.5, 1.0]: height 1.0. Area = 0.5.
                      Total 1.25. Correct.
                      
                      So the objective is to maximize:
                      H = sum_{i=1}^K (f1_i - f1_{i-1}) * f2_i
                      subject to 0 <= f1_1 < f1_2 < ... < f1_K <= 1.
                      f2_i = 2 - sqrt(f1_i).
                      
                      This is a deterministic optimization problem.
                      We can solve it using DP if we discretize f1.
                      However, the optimal solution for a convex function like this 
                      (f2 is convex in f1? f2' = -1/(2sqrt(f1)), f2'' = 1/(4f1^1.5) > 0) 
                      might have a specific structure.
                      
                      Let's use a greedy approach or DP.
                      With K=200, and f1 in [0, 1].
                      The function f2(f1) = 2 - sqrt(f1).
                      We want to place points to maximize the area under the staircase.
                      This is equivalent to approximating the integral of f2(f1) from 0 to 1 
                      using a right-hand Riemann sum?
                      Sum (f1_i - f1_{i-1}) f2(f1_i).
                      Since f2 is decreasing, the right-hand sum underestimates the integral.
                      We want to maximize this sum.
                      The maximum possible sum is the integral of f2 from 0 to 1, which is [2f1 - (2/3)f1^1.5]_0^1 = 2 - 2/3 = 4/3.
                      But we are constrained to K points.
                      To maximize the sum, we should place points where the "loss" from the right-hand sum is minimized.
                      The error of the right-hand sum for a decreasing convex function is largest where the curvature is highest?
                      Actually, for a convex function, the right-hand sum is below the integral.
                      The gap is roughly sum (delta_i^2 / 2) * f2''(xi).
                      To minimize the gap (maximize the sum), we should distribute points such that 
                      delta_i^2 * f2''(f1_i) is constant?
                      This leads to a non-uniform grid.
                      
                      However, we can just solve the DP.
                      State: dp[i][j] = max HV using j points from the first i bins.
                      This is O(K * M^2) which might be too slow if M is large.
                      But K=200. If M=1000, 200*1000*1000 = 200M ops. Too slow in Python.
                      
                      Alternative:
                      The optimal points for maximizing the hypervolume of a convex front 
                      with a fixed number of points are known to be related to the 
                      "optimal quantization" or specific spacing.
                      
                      Let's try a simpler structural change:
                      Instead of DP, use the analytical property.
                      For f(x) = 2 - sqrt(x), the optimal points for maximizing the 
                      hypervolume (which is the area under the staircase) are those that 
                      equalize the "local hypervolume contribution" or something similar.
                      
                      Actually, let's look at the derivative of the HV with respect to a point position.
                      If we move point i by dx, the HV changes by:
                      dH = dx * (f2_{i-1} - f2_i)  (from the left rectangle)
                         - dx * (f2_i - f2_{i+1})  (from the right rectangle)
                         + dx * f2'(x_i) * (x_i - x_{i-1}) ... this is getting complex.
                         
                      Let's stick to the DP but optimize it.
                      Or, use a heuristic that is structurally different:
                      "Space filling" based on the derivative of the front.
                      The number of points in an interval [a, b] should be proportional to 
                      the change in the "shape" of the front.
                      For ZDT4, f2 = 2 - sqrt(f1).
                      The "curvature" or "slope change" is high near 0.
                      
                      The current program uses x0 = t^2.
                      Let's try x0 = t^p with a different p, or a different transform.
                      But the directive says "Replace ... with a discrete 1D dynamic programming".
                      
                      Okay, I will implement a simplified DP.
                      Discretize [0, 1] into M=1000 bins.
                      DP[i][j] = max HV using j points from bins 0..i.
                      Transition: DP[i][j] = max(DP[i-1][j], DP[i-1][j-1] + contribution(i, j))
                      What is the contribution of placing a point at bin i as the j-th point?
                      It depends on the previous point.
                      So DP[i][j] = max over k < i of DP[k][j-1] + HV_contribution(k, i)
                      where HV_contribution(k, i) is the additional HV gained by adding point i 
                      after point k.
                      If the previous point was at f1_k and the new one is at f1_i, 
                      and the one before k was at f1_prev...
                      This dependency makes it hard to decompose simply.
                      
                      However, note that H = sum (f1_i - f1_{i-1}) f2_i.
                      This is a linear combination of f2_i with weights (f1_i - f1_{i-1}).
                      We can rewrite H = f1_K f2_K + sum_{i=1}^{K-1} f1_i (f2_i - f2_{i+1}).
                      (Telescoping sum: 
                      sum_{i=1}^K (f1_i - f1_{i-1}) f2_i = f1_1 f2_1 + f1_2 f2_2 - f1_1 f2_2 + ... + f1_K f2_K - f1_{K-1} f2_K
                      = f1_K f2_K + sum_{i=1}^{K-1} f1_i (f2_i - f2_{i+1}) )
                      
                      So H = f1_K f2_K + sum_{i=1}^{K-1} f1_i (f2_i - f2_{i+1}).
                      We want to maximize this.
                      Since f2_i - f2_{i+1} > 0, we want f1_i to be large? 
                      But f1_i < f1_{i+1}.
                      
                      This formulation suggests that the points should be placed to maximize 
                      the weighted sum.
                      
                      Let's use a greedy algorithm or a simple DP with state (last_point_index, count).
                      M=1000, K=200.
                      DP[count][last] = max HV.
                      DP[1][i] = f1_i * f2_i.
                      DP[c][i] = max_{j < i} ( DP[c-1][j] + f1_i * f2_i + f1_j * (f2_j - f2_i) ? No.
                      Let's re-evaluate the incremental gain.
                      If we have a set S ending with j, and we add i > j.
                      The new HV is H(S U {i}).
                      H(S) = ... + f1_j f2_j + ...
                      H(S U {i}) = H(S) - f1_j f2_j + f1_j (f2_j - f2_i) + f1_i f2_i
                                 = H(S) + f1_i f2_i - f1_j f2_i
                                 = H(S) + (f1_i - f1_j) f2_i.
                      Wait, this assumes the term for j was f1_j f2_j in the sum?
                      In the formula H = f1_K f2_K + sum_{i=1}^{K-1} f1_i (f2_i - f2_{i+1}),
                      the last term is f1_K f2_K.
                      If we append i to a set ending in j, the old last term f1_j f2_j becomes 
                      f1_j (f2_j - f2_i), and the new last term is f1_i f2_i.
                      Change = [f1_j (f2_j - f2_i) + f1_i f2_i] - f1_j f2_j
                              = -f1_j f2_i + f1_i f2_i
                              = (f1_i - f1_j) f2_i.
                      
                      So the transition is:
                      DP[c][i] = max_{j < i} ( DP[c-1][j] + (f1_i - f1_j) * f2_i )
                      
                      This is O(K * M^2).
                      K=200, M=1000. 200 * 1000 * 1000 = 200,000,000 operations.
                      In Python, this is too slow (seconds to minutes).
                      We need to optimize the max over j.
                      DP[c][i] = f1_i f2_i + max_{j < i} ( DP[c-1][j] - f1_j f2_i )
                      = f1_i f2_i + max_{j < i} ( DP[c-1][j] - f2_i f1_j )
                      
                      This is a max of linear functions in f2_i?
                      Let y = f2_i.
                      We want max_j ( DP[c-1][j] - y * f1_j ).
                      This is the upper envelope of lines with slope -f1_j and intercept DP[c-1][j].
                      We can use the Convex Hull Trick (CHT) to compute this in O(1) amortized 
                      per query, if the slopes are monotonic.
                      f1_j is increasing, so slopes -f1_j are decreasing.
                      Queries y = f2_i are decreasing (since f2 is decreasing).
                      This fits the CHT conditions perfectly!
                      
                      So we can solve this in O(K * M).
                      K=200, M=1000 -> 200,000 operations. Very fast.
                      
                      Plan:
                      1. Discretize x0 in [0, 1] into M=2000 points.
                      2. Compute f1_i, f2_i.
                      3. DP[1][i] = f1_i * f2_i.
                      4. For c from 2 to K:
                         Initialize CHT.
                         Iterate i from 0 to M-1:
                            Query CHT for y = f2_i to get max_val.
                            DP[c][i] = f1_i * f2_i + max_val.
                            Add line (slope = -f1_i, intercept = DP[c-1][i]) to CHT.
                            Store argmax to reconstruct path.
                      5. Reconstruct the path of K points.
                      6. Evaluate these points on the problem.
                      
                      This is a structurally different approach (DP + CHT) compared to 
                      heuristic grid sampling.
                      """

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 290
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # Check if 0 is within bounds for x1..x9
        zeros_ok = all(lo[i] <= 0 <= hi[i] for i in range(1, self.dim))
        
        if not zeros_ok:
            # Fallback to simple sampling if we can't hit the front exactly
            # This case is unlikely given the task description, but good for safety
            return self._fallback(problem, bounds, deadline)

        # 1. Discretize x0 domain
        # We want to cover [0, 1] for x0.
        x0_lo = max(lo[0], 0.0)
        x0_hi = min(hi[0], 1.0)
        
        if x0_lo >= x0_hi:
            # Degenerate case
            x0_vals = np.array([lo[0]])
            X = np.zeros((1, self.dim))
            X[0, 0] = lo[0]
            F = [problem(X[0])]
            return X

        M = 2000 # Resolution for DP
        x0_grid = np.linspace(x0_lo, x0_hi, M)
        f1 = x0_grid
        f2 = 2.0 - np.sqrt(f1)
        
        # 2. DP with Convex Hull Trick
        K = self.budget
        # DP[c][i] stores max HV using c points, with the last point at index i.
        # We only need the previous row.
        # prev_dp[i] = DP[c-1][i]
        # curr_dp[i] = DP[c][i]
        
        # Initialize for c=1
        prev_dp = f1 * f2
        # argmax storage: parent[c][i] = j that maximizes DP[c][i]
        # We need to store parents for all c to reconstruct the path.
        # Memory: K * M = 200 * 2000 = 400,000 ints. Fine.
        parents = np.zeros((K, M), dtype=int)
        parents[0] = -1 # No parent for first point
        
        # CHT Implementation
        # Lines: y = m*x + b
        # We want max(m*x + b)
        # Slopes m are -f1_i (decreasing)
        # Queries x are f2_i (decreasing)
        
        def compute_row(c, prev_dp):
            curr_dp = np.full(M, -np.inf)
            # CHT data structures
            # We store lines as (m, b, index)
            # Since slopes are added in decreasing order and queries are decreasing,
            # we can use a deque.
            # However, standard CHT assumes slopes are added in increasing order for 
            # max queries with increasing x. 
            # Here slopes are decreasing, x is decreasing.
            # Let's reverse the logic or adapt.
            # If we add lines with decreasing slopes, and query with decreasing x,
            # the "optimal" line index moves in the opposite direction?
            # Let's just implement a generic CHT or adjust.
            # Actually, if we reverse the arrays, it becomes standard.
            # But let's try to implement it directly.
            
            # Line: y = m*x + b
            # Intersection of L1(m1,b1) and L2(m2,b2) is x = (b2-b1)/(m1-m2)
            
            lines = [] # List of (m, b, idx)
            
            def is_bad(l1, l2, l3):
                # l1, l2, l3 are (m, b, idx)
                # l2 is bad if intersection(l1, l2) >= intersection(l2, l3)
                # for max hull with decreasing slopes and decreasing queries?
                # Let's check the condition.
                # If slopes are m1 > m2 > m3 (decreasing).
                # We want max.
                # The hull should have increasing intersections.
                # x12 = (b2-b1)/(m1-m2)
                # x23 = (b3-b2)/(m2-m3)
                # If x12 >= x23, then l2 is never optimal.
                m1, b1, _ = l1
                m2, b2, _ = l2
                m3, b3, _ = l3
                # Avoid division by zero
                if m1 == m2:
                    return b1 >= b2 # Keep the one with higher intercept
                if m2 == m3:
                    return b2 >= b3
                # (b2-b1)/(m1-m2) >= (b3-b2)/(m2-m3)
                # Note m1-m2 > 0 and m2-m3 > 0
                return (b2-b1)*(m2-m3) >= (b3-b2)*(m1-m2)
            
            # We iterate i from 0 to M-1.
            # But we need to query for i before adding i?
            # DP[c][i] depends on DP[c-1][j] for j < i.
            # So when computing i, we have added lines for j < i.
            
            for i in range(M):
                if i > 0:
                    # Query
                    x_query = f2[i]
                    # Find best line
                    # Since x_query is decreasing, and we added lines with decreasing slopes,
                    # the optimal line index might move?
                    # Let's use a pointer or binary search.
                    # Given the small M (2000), even O(M) query is 2000*2000*200 = 800M. Too slow.
                    # We need O(1) or O(log M) query.
                    # With monotonic queries and monotonic slopes, we can use a pointer.
                    # But we are adding lines one by one.
                    # This is the "online" CHT.
                    # If slopes are added in sorted order and queries are sorted, 
                    # we can pop from the front of the deque.
                    
                    # Let's maintain a deque of lines.
                    # Pass 1: Add lines and query.
                    # Wait, I need to add line j before querying i if j < i.
                    # So in the loop for i:
                    # 1. If i > 0, the line for i-1 should have been added?
                    # No, the line for j uses DP[c-1][j].
                    # So for each i, we need to have added lines for all j < i.
                    
                    # Let's restructure:
                    # Loop i from 0 to M-1:
                    #   If i > 0:
                    #      Add line for j = i-1 to CHT.
                    #   Query CHT for x = f2[i].
                    #   curr_dp[i] = f1[i]*f2[i] + result.
                    
                    # This works.
                    pass
                
                # Add line for j = i-1? No, we add line for j before querying i.
                # So at start of iteration i, we add line for j=i-1.
                if i > 0:
                    m = -f1[i-1]
                    b = prev_dp[i-1]
                    idx = i-1
                    new_line = (m, b, idx)
                    
                    while len(lines) >= 2 and is_bad(lines[-2], lines[-1], new_line):
                        lines.pop()
                    lines.append(new_line)
                
                # Query
                if i == 0:
                    # No lines available?
                    # DP[c][0] is impossible if c > 1 because we need j < 0.
                    # So curr_dp[0] remains -inf.
                    continue
                
                x_query = f2[i]
                # Find best line in `lines`
                # Since x_query is decreasing, and slopes are decreasing,
                # the optimal line is at the "end" or "beginning"?
                # Let's check.
                # High slope (more negative m? No, m = -f1. f1 is small -> m is small (less negative)).
                # f1[0] is small -> m[0] is close to 0.
                # f1[M-1] is large -> m[M-1] is very negative.
                # So slopes are decreasing (0, -0.1, -0.2 ...).
                # Queries x = f2[i]. f2[0]=2, f2[M-1]=1.
                # So queries are decreasing (2, 1.9, ... 1).
                
                # For max(m*x + b):
                # If x is large, lines with higher m (less negative) are better.
                # If x is small, lines with lower m (more negative) might be better if b is high.
                # Since x is decreasing, we start with large x (prefer high m) and move to small x.
                # The optimal line index in the hull should move from "high m" to "low m".
                # In our deque, lines are added in order of decreasing m.
                # So the front of the deque has the highest m, the back has the lowest.
                # As x decreases, we should move from front to back?
                # Let's verify with 2 lines.
                # L1: m=0, b=0. L2: m=-1, b=10.
                # Intersection: 0 = -x + 10 => x=10.
                # For x > 10, L1 is better (0 > -x+10).
                # For x < 10, L2 is better.
                # We start with x=2 (large) -> L1.
                # We end with x=1 (small) -> L2.
                # So we move from front (L1) to back (L2).
                
                # We can maintain a pointer `ptr` to the current best line.
                # Since we only move forward (front to back), we can just increment ptr.
                # But we also add new lines to the back.
                # So the "best" line might be the new one?
                # No, the new line has the lowest slope. It will only become optimal 
                # for very small x.
                # So the pointer moves from front to back.
                
                # We need to store the pointer per i? No, it's monotonic.
                # But we are in a loop for a specific c.
                # The pointer for c can be reset?
                # No, the set of lines grows.
                # However, the "optimal" line for a given x is determined by the hull.
                # We can just do a binary search on the intersection points.
                # Or, since M is small (2000), we can just iterate?
                # 200 * 2000 * 2000 = 800M. Still too slow.
                
                # Let's use the pointer method.
                # We need to store the lines in a list and maintain a pointer.
                # But the pointer depends on x_query.
                # Since x_query is decreasing, the pointer moves forward (towards back).
                
                # We'll keep a global pointer for the current row c.
                # But we are inside the loop for i.
                # Let's restructure to use a deque and a pointer.
                
                # Actually, let's just use a simple binary search on the intersection points 
                # stored in the hull.
                # We need to store the intersection x-coordinates.
                
                # Let's restart the CHT logic with explicit intersections.
                
                pass # Placeholder for actual implementation below

            return curr_dp, parents[c-1] # parents[c-1] is the row for c-1? No, we store parent for c.
        
        # Let's implement the CHT properly in the loop.
        
        # We need to store parents for backtracking.
        # parents[c-1][i] should store the j that maximizes DP[c][i].
        
        # Re-implementing the loop with correct CHT
        
        # Helper to get best line
        # We will maintain a list of lines and a list of intersection x-values.
        
        for c in range(1, K):
            curr_dp = np.full(M, -np.inf)
            # CHT structures
            hull_lines = [] # (m, b, idx)
            hull_intersections = [] # x-coordinate where the new line becomes better than the previous one
            
            # Pointer for queries?
            # Since x_query is decreasing, and we add lines with decreasing slopes,
            # the optimal line index moves from 0 to end.
            
            for i in range(M):
                # 1. Add line for j = i-1
                if i > 0:
                    m = -f1[i-1]
                    b = prev_dp[i-1]
                    idx = i-1
                    new_line = (m, b, idx)
                    
                    # Check if new line is redundant
                    # Intersection of hull_lines[-1] and new_line
                    if len(hull_lines) == 0:
                        hull_lines.append(new_line)
                        hull_intersections.append(-np.inf) # Always best
                    else:
                        # Calculate intersection
                        m1, b1, _ = hull_lines[-1]
                        m2, b2, _ = new_line
                        if m1 == m2:
                            # Keep the one with higher intercept
                            if b1 >= b2:
                                # New line is worse, skip
                                continue
                            else:
                                # New line is better, pop last
                                hull_lines.pop()
                                hull_intersections.pop()
                                # Recurse? Or just loop.
                                # Since we are in a loop, we can just use a while loop.
                                pass # Handled by while loop below
                        else:
                            x_int = (b2 - b1) / (m1 - m2)
                            
                            # Pop while the new intersection is <= previous intersection
                            while len(hull_lines) > 0:
                                if len(hull_lines) == 1:
                                    break
                                m_last, b_last, _ = hull_lines[-1]
                                m_prev, b_prev, _ = hull_lines[-2]
                                x_prev_int = hull_intersections[-1]
                                
                                # If x_int <= x_prev_int, the last line is never optimal
                                if x_int <= x_prev_int:
                                    hull_lines.pop()
                                    hull_intersections.pop()
                                else:
                                    break
                            
                            # Now add new line
                            if len(hull_lines) == 0:
                                hull_lines.append(new_line)
                                hull_intersections.append(-np.inf)
                            else:
                                m_last, b_last, _ = hull_lines[-1]
                                x_int = (b2 - b1) / (m1 - m2)
                                hull_lines.append(new_line)
                                hull_intersections.append(x_int)
                
                # 2. Query
                if i == 0:
                    continue # DP[c][0] is -inf for c>1
                
                x_query = f2[i]
                
                # Find best line
                # hull_intersections contains the x where each line becomes better.
                # hull_intersections[0] is -inf.
                # hull_intersections[k] is the x where hull_lines[k] becomes better than hull_lines[k-1].
                # Since x_query is decreasing, we want the largest k such that hull_intersections[k] >= x_query?
                # No, if x_query is large, we want the line with high m (index 0).
                # If x_query is small, we want the line with low m (high index).
                # The condition for line k to be optimal is x_query <= hull_intersections[k] AND x_query > hull_intersections[k-1]?
                # Let's check.
                # Line 0 is optimal for x > int_1.
                # Line 1 is optimal for int_2 > x > int_1.
                # Line k is optimal for x < int_{k+1} AND x > int_k.
                
                # So we want the largest k such that x_query > hull_intersections[k].
                # Or binary search for the rightmost intersection < x_query.
                
                # Since x_query is decreasing, we can use a pointer that moves forward (increasing index).
                # But we are adding lines, so the hull changes.
                # However, the "optimal" line for a fixed x only moves to higher indices as x decreases.
                # And we are iterating i with decreasing x.
                # So we can maintain a pointer `best_idx` that only increases.
                
                # But we are inside the loop for i.
                # Let's just do a binary search for simplicity and correctness, 
                # as M=2000 and log(2000) ~ 11. 200*2000*11 = 4.4M ops. Fast enough.
                
                # Binary search on hull_intersections
                # We want the largest k such that hull_intersections[k] < x_query.
                # If no such k, use 0.
                
                lo_bs = 0
                hi_bs = len(hull_lines) - 1
                best_k = 0
                
                # hull_intersections is increasing.
                # We want the rightmost value < x_query.
                
                while lo_bs <= hi_bs:
                    mid = (lo_bs + hi_bs) // 2
                    if hull_intersections[mid] < x_query:
                        best_k = mid
                        lo_bs = mid + 1
                    else:
                        hi_bs = mid - 1
                
                m_best, b_best, idx_best = hull_lines[best_k]
                val = m_best * x_query + b_best
                curr_dp[i] = f1[i] * f2[i] + val
                parents[c][i] = idx_best # Store the parent j
                
            prev_dp = curr_dp
        
        # Reconstruct path
        # Start from the best last point
        last_i = np.argmax(prev_dp)
        points = [last_i]
        for c in range(K-1, 0, -1):
            last_i = parents[c][last_i]
            points.append(last_i)
        
        points.reverse()
        
        # Get x0 values
        x0_vals = x0_grid[points]
        
        # Ensure we don't exceed budget
        if len(x0_vals) > self.budget:
            x0_vals = x0_vals[:self.budget]
            
        # Construct X
        X = np.zeros((len(x0_vals), self.dim))
        X[:, 0] = x0_vals
        # X[:, 1:] = 0
        
        # Evaluate
        F = []
        for i in range(len(x0_vals)):
            if time.time() > deadline:
                break
            F.append(problem(X[i]))
            
        return X

    def _fallback(self, problem, bounds, deadline):
        # Simple uniform sampling
        n_points = self.budget
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        x0_lo = lo[0]
        x0_hi = hi[0]
        if x0_lo >= x0_hi:
            x0_vals = np.array([lo[0]])
        else:
            x0_vals = np.linspace(x0_lo, x0_hi, n_points)
            
        X = np.zeros((len(x0_vals), self.dim))
        X[:, 0] = x0_vals
        
        F = []
        for i in range(len(x0_vals)):
            if time.time() > deadline:
                break
            F.append(problem(X[i]))
            
        return X

# EVOLVE-BLOCK-END
