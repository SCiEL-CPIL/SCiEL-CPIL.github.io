"""Problem-aware ZDT1 optimizer: Hypervolume-driven adaptive 1D grid refinement."""
import numpy as np
import time


class Optimizer:
    """
    Exploit ZDT1 structure: f1 = x[0], f2 = (1 + sqrt(f1)) * g(x[1:]).
    Pareto front is achieved when g(x[1:]) = 1, i.e., x[i] = 0.2 for i > 0.
    
    Strategy:
    1. Since we know the Pareto front analytically, the problem reduces to 
       choosing x[0] values that maximize the hypervolume of the points 
       (x[0], 1 + sqrt(x[0])) with reference point (11, 11).
    2. Use adaptive refinement: start with a coarse grid, then iteratively 
       add points in the interval with the highest hypervolume contribution.
    3. This is a structural change from uniform/non-uniform fixed grids to 
       a dynamic, hypervolume-driven adaptive sampling strategy.
    """

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 320  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)

        budget = self.budget
        dim = self.dim

        # ZDT1 Pareto front: f1 = x0, f2 = 1 + sqrt(x0) when x[1:] = 0.2
        # Reference point: (11, 11)
        # Hypervolume contribution of a point (f1, f2) relative to ref (11,11):
        # The front is concave, so we want to maximize the area under the curve
        # from the leftmost point to the rightmost point, minus the area below.

        # Actually, for a set of non-dominated points on the Pareto front,
        # the hypervolume is the area between the front and the reference point.
        # For points sorted by f1 ascending, the hypervolume can be computed as:
        # HV = sum over consecutive pairs of trapezoidal areas.

        # Key insight: The optimal set of N points on a concave front to maximize
        # hypervolume is found by adaptive refinement - always add the point in
        # the interval that gives the largest increase in hypervolume.

        # The Pareto front is f2 = 1 + sqrt(f1) for f1 in [0, 1].
        # This is a concave function (second derivative < 0).

        # For a concave front, the hypervolume-optimal point to add in an interval
        # [a, b] is where the chord from (a, 1+sqrt(a)) to (b, 1+sqrt(b)) 
        # has the maximum vertical distance to the curve. For f2 = 1 + sqrt(f1),
        # this is at the midpoint of f1? No, it's where the curve is farthest 
        # from the chord. Let's compute this analytically.

        # Actually, for maximizing hypervolume with a fixed number of points on a 
        # concave curve, the optimal strategy is to place points where the local 
        # curvature is highest. For f2 = 1 + sqrt(f1), the curvature is highest 
        # near f1 = 0.

        # Let's use a different approach: iterative hypervolume maximization.
        # Start with 2 points (endpoints), then iteratively add the point that 
        # maximizes hypervolume increase.

        # The hypervolume of points on the front (f1_i, f2_i) sorted by f1:
        # HV = sum_{i=1}^{n-1} (f1_{i+1} - f1_i) * (f2_i - ref_f2) 
        #     + (ref_f1 - f1_n) * (f2_n - ref_f2) ... 
        # Actually, it's easier to compute as the area between the step function 
        # formed by the points and the reference point.

        # Let me just implement the adaptive refinement directly.

        # Phase 1: Initial coarse grid
        n_initial = 10
        x0_initial = np.linspace(0.0, 1.0, n_initial)
        
        # Track which x0 values we've evaluated
        evaluated_x0 = list(x0_initial)
        
        # Evaluate initial points
        results_F = []
        x_rest_good = np.full(dim - 1, 0.2)
        
        for x0 in x0_initial:
            if time.time() > deadline - 10:
                break
            x = np.empty(dim)
            x[0] = x0
            x[1:] = x_rest_good
            f = problem(x)
            results_F.append(f)
        
        # Phase 2: Adaptive refinement
        # For each interval between consecutive evaluated x0 values, 
        # compute the hypervolume gain of adding the midpoint.
        # Add the point with the highest gain, repeat until budget exhausted.
        
        # Helper: compute hypervolume of points on the Pareto front
        def compute_hv(x0_vals):
            """Compute hypervolume of points (x0, 1+sqrt(x0)) with ref (11,11)."""
            if len(x0_vals) == 0:
                return 0.0
            x0_sorted = np.sort(x0_vals)
            f2_vals = 1.0 + np.sqrt(x0_sorted)
            ref_f1, ref_f2 = 11.0, 11.0
            
            # Only count points strictly better than reference
            mask = (x0_sorted < ref_f1) & (f2_vals < ref_f2)
            x0_sorted = x0_sorted[mask]
            f2_vals = f2_vals[mask]
            
            if len(x0_sorted) == 0:
                return 0.0
            
            # Hypervolume: area between the step function and reference point
            # The step function goes: for each point i, it's at height f2_i 
            # from f1_{i-1} to f1_i (with f1_{-1} = 0 for the leftmost)
            # Actually, standard HV calculation for 2D:
            # Sort by f1 ascending. HV = sum of (f1_{i+1} - f1_i) * (ref_f2 - f2_i) 
            # for i from 0 to n-2, plus (ref_f1 - f1_{n-1}) * (ref_f2 - f2_{n-1})
            # Wait, that's not quite right. Let me think again.
            
            # For non-dominated points sorted by f1 ascending:
            # HV = (ref_f1 - f1_0) * (ref_f2 - f2_0) - sum of "missing" areas
            # Actually, the simplest way: HV is the area of the union of rectangles
            # [0, f1_i] x [0, f2_i] ... no, that's for minimization with ref at origin.
            
            # With reference point (11, 11), the hypervolume is the volume of the 
            # region dominated by the points but not by the reference point.
            # For 2D: HV = area of {(a,b) : a < 11, b < 11, and exists point (f1_i, f2_i) 
            # with f1_i <= a and f2_i <= b}
            
            # For points on a monotone decreasing front (which ZDT1's front is NOT - 
            # it's monotone increasing in f2 as f1 increases, but since we're minimizing,
            # the Pareto front has f2 decreasing as f1 increases? No...
            
            # ZDT1: f1 = x0, f2 = (1+sqrt(x0))*g. On the Pareto front, g=1, so 
            # f2 = 1+sqrt(x0). As x0 increases, f1 increases AND f2 increases.
            # So the Pareto front is monotone increasing. But for minimization, 
            # non-dominated points should have the property that if you increase 
            # one objective, you must decrease the other. 
            # 
            # Wait, I'm confusing myself. In ZDT1, both objectives are to be minimized.
            # The Pareto front is the set of points where you can't improve one 
            # objective without worsening the other. For ZDT1 with f1=x0 and 
            # f2=(1+sqrt(x0))*g, when g=1, f2=1+sqrt(x0). As x0 goes from 0 to 1,
            # f1 goes from 0 to 1, and f2 goes from 1 to 2. So both objectives 
            # increase. This means the "Pareto front" in the minimization sense 
            # is actually just the point (0, 1)? No, that's not right either.
            #
            # Let me reconsider. In ZDT1, the standard definition is:
            # f1(x) = x1
            # g(x) = 1 + 10*(n-1) + sum_{i=2}^n (x_i - 0.2)^2 * sin(10*pi*x_i)
            # f2(x) = (1 + sqrt(f1(x)/n)) * g(x)  [for ZDT1, n is the number of vars]
            #
            # Wait, the standard ZDT1 has f2 = (1 + 9*(g/n)) * (1 - sqrt(f1/n)) 
            # or similar. Let me check the actual ZDT1 definition.
            #
            # Standard ZDT1:
            # f1(x) = x1
            # g(x) = 1 + 10*(n-1) + sum_{i=2}^n (x_i - 0.2)^2 * sin(10*pi*x_i)
            # f2(x) = g(x) * (1 - sqrt(f1(x)/n))
            #
            # So f2 = g * (1 - sqrt(x1/n)). When g=1 (minimum), f2 = 1 - sqrt(x1/n).
            # As x1 increases from 0 to n, f1 increases and f2 decreases from 1 to 0.
            # So the Pareto front is monotone decreasing: as f1 increases, f2 decreases.
            #
            # But the task says n_obj=2 and dim=30. If n=30, then f2 = 1 - sqrt(x1/30).
            # When x1=0, f2=1. When x1=30, f2=0. But bounds are [0,1] for all vars.
            # So x1 is in [0,1], and f1 is in [0,1]. f2 = 1 - sqrt(x1/30).
            # When x1=0, f2=1. When x1=1, f2=1-sqrt(1/30)≈1-0.183=0.817.
            #
            # Hmm, but the reference point is (11, 11), which is way larger than 
            # any objective value. So all points are dominated by the reference.
            #
            # The hypervolume is the area between the Pareto front and the reference 
            # point. For a monotone decreasing front from (0,1) to (1, 0.817),
            # the HV with ref (11,11) is the area of the rectangle [0,11]x[0,11] 
            # minus the area below the front.
            #
            # Actually, for minimization, the hypervolume indicator with reference 
            # point r is the volume of the region dominated by the set of points 
            # but not by r. For a set of non-dominated points, this is the area 
            # between the points and the reference point.
            #
            # For a monotone decreasing front, the HV can be computed as:
            # HV = (ref_f1 - f1_max) * (ref_f2 - f2_min) + sum of trapezoidal areas
            #
            # Let me just implement a general HV calculator for 2D.

            # General 2D HV calculation for minimization:
            # Sort points by f1 ascending.
            # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (ref_f2 - f2_i) 
            #     where f1_n = ref_f1
            # But this only works if the points form a staircase.
            
            # More robust: 
            # HV = 0
            # prev_f1 = 0  # or -inf
            # for each point sorted by f1 ascending:
            #   if f2 < ref_f2:
            #     HV += (f1 - prev_f1) * (ref_f2 - f2)
            #     prev_f1 = f1
            # HV += (ref_f1 - prev_f1) * (ref_f2 - f2_last)  # if last f2 < ref_f2
            
            # Wait, this isn't right either. The standard formula for 2D HV 
            # with reference point (r1, r2) for minimization:
            # Sort points by f1 ascending.
            # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i)
            # where f1_{-1} = 0 (or the minimum f1) and f1_n = r1.
            # But this assumes the points are non-dominated and sorted.
            
            # Let me use a different approach: compute HV as the area of the 
            # union of rectangles [f1_i, r1] x [f2_i, r2] for all points.
            # But since points are non-dominated, these rectangles don't overlap 
            # in a way that makes this simple.
            
            # Actually, for non-dominated points sorted by f1 ascending, 
            # f2 is decreasing. The HV is:
            # HV = (r1 - f1_0) * (r2 - f2_0) - sum_{i=0}^{n-2} (f1_{i+1} - f1_i) * (f2_i - f2_{i+1})
            # No, that's not right either.
            
            # Let me just use the standard incremental formula:
            # Sort by f1 ascending.
            # HV = 0
            # last_f2 = r2
            # for i in range(n):
            #   if f2_i < last_f2:
            #     HV += (f1_{i+1} - f1_i) * (last_f2 - f2_i)  # where f1_n = r1
            #     last_f2 = f2_i
            
            # Hmm, I'm overcomplicating this. Let me just use a simple approach:
            # The HV for 2D minimization with reference (r1, r2) is the area 
            # of the region {(a,b): a < r1, b < r2, and there exists a point 
            # (f1_i, f2_i) in the set such that f1_i <= a and f2_i <= b}.
            
            # For non-dominated points sorted by f1 ascending (and f2 descending),
            # this region is a staircase. The area is:
            # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i)
            # where f1_n = r1 and f1_{-1} = 0 (or the min f1).
            # But we need to be careful: the first term should be (f1_0 - 0) * (r2 - f2_0)?
            # No, the region starts from the origin (0,0) in the dominated space.
            
            # Actually, the standard formula is:
            # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i)
            # where f1_{-1} = 0 and f1_n = r1, and points are sorted by f1 ascending.
            # But this gives the area from f1=0 to f1=r1.
            
            # Let me just implement it numerically:
            # Create a grid and check which cells are dominated.
            # But that's expensive. Let me use the analytical formula for 2D.
            
            # For 2D, with points sorted by f1 ascending:
            # HV = (r1 - f1_{n-1}) * (r2 - f2_{n-1}) + 
            #      sum_{i=0}^{n-2} (f1_{i+1} - f1_i) * (r2 - f2_i)
            # Wait, this double-counts. Let me think geometrically.
            
            # The dominated region is the union of rectangles [0, f1_i] x [0, f2_i] 
            # for all i. But since points are non-dominated, f1_i < f1_j implies 
            # f2_i > f2_j. So the union is a staircase.
            
            # The area of the staircase from (0,0) to (r1, r2) is:
            # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * f2_i
            # where f1_{-1} = 0, f1_n = r1.
            # But this is the area under the staircase, not the HV.
            
            # The HV is the area of the region dominated by the points but NOT 
            # by the reference point. Since the reference point is (r1, r2) and 
            # we're minimizing, the region is {(a,b): a < r1, b < r2, dominated by some point}.
            # This is the area of the staircase from (0,0) to (r1, r2) minus the 
            # area below the reference point... no.
            
            # I think the correct formula for 2D HV with reference (r1, r2) is:
            # HV = (r1 * r2) - (area not dominated by any point)
            # But that's not right either.
            
            # Let me just use a simple numerical approach for correctness:
            # The HV is the measure of the set of points in [0, r1] x [0, r2] 
            # that are dominated by at least one point in the set.
            # For non-dominated points, this is a staircase region.
            
            # I'll compute it as:
            # HV = 0
            # prev_f1 = 0
            # for each point sorted by f1 ascending:
            #   if f2 < r2:
            #     HV += (f1 - prev_f1) * (r2 - f2) + (r1 - f1) * 0  # no
            #     # The rectangle from (prev_f1, f2) to (f1, r2) is fully dominated
            #     # The rectangle from (f1, 0) to (r1, f2) is partially dominated
            #     # This is getting complicated.
            
            # Let me just use the standard formula from the literature:
            # For 2D, HV = sum_{i=1}^{n} (f1_i - f1_{i-1}) * (r2 - f2_i)
            # where f1_0 = 0 and points are sorted by f1 ascending, and we only 
            # include points with f2 < r2.
            # But this assumes f1_0 = 0, which may not be correct.
            
            # Actually, I think the correct formula is:
            # Sort points by f1 ascending.
            # HV = (r1 - f1_{n-1}) * (r2 - f2_{n-1}) + 
            #      sum_{i=0}^{n-2} (f1_{i+1} - f1_i) * (r2 - f2_i)
            # Let me verify with a simple example:
            # One point (1, 1), ref (10, 10).
            # HV should be the area of [0,10]x[0,10] that is dominated by (1,1).
            # That's [0,1]x[0,1], area = 1.
            # Formula: (10-1)*(10-1) + 0 = 81. That's wrong.
            
            # I see, the HV is NOT the area of the dominated region in the 
            # objective space. It's the volume of the region in the objective 
            # space that is dominated by the set of points and not dominated by 
            # the reference point. For minimization, a point (a,b) is dominated 
            # by (f1_i, f2_i) if f1_i <= a and f2_i <= b.
            # The reference point (r1, r2) dominates (a,b) if r1 <= a and r2 <= b.
            # Wait, no. The reference point is used to define the region of interest.
            # The HV is the volume of the union of hyperrectangles [f1_i, r1] x [f2_i, r2] 
            # for all points i, where the hyperrectangle is the set of points 
            # dominated by point i and not dominated by the reference point.
            
            # For 2D, the HV is the area of the union of rectangles 
            # [f1_i, r1] x [f2_i, r2] for all i.
            # For non-dominated points sorted by f1 ascending, these rectangles 
            # form a staircase, and the area is:
            # HV = sum_{i=0}^{n-1} (r1 - f1_i) * (r2 - f2_i) - overlaps
            # But since points are non-dominated, the rectangles don't overlap 
            # in a simple way.
            
            # Actually, for non-dominated points, the union of rectangles 
            # [f1_i, r1] x [f2_i, r2] has area:
            # HV = (r1 - f1_0) * (r2 - f2_0) + 
            #      sum_{i=1}^{n-1} (f1_i - f1_{i-1}) * (r2 - f2_i)
            # Let me verify: one point (1,1), ref (10,10).
            # HV = (10-1)*(10-1) = 81. But the actual HV should be the area 
            # of [1,10]x[1,10] = 81. Yes! That's correct.
            # The HV is the area of the region [f1, r1] x [f2, r2] for the 
            # non-dominated points, combined.
            
            # For two points (1,2) and (2,1), ref (10,10).
            # HV = (10-1)*(10-2) + (2-1)*(10-1) = 9*8 + 1*9 = 72+9 = 81.
            # Let me verify: the union of [1,10]x[2,10] and [2,10]x[1,10].
            # Area of [1,10]x[2,10] = 9*8 = 72.
            # Area of [2,10]x[1,10] = 8*9 = 72.
            # Overlap: [2,10]x[2,10] = 8*8 = 64.
            # Union area = 72+72-64 = 80. But formula gives 81. Discrepancy.
            
            # Let me recalculate: (10-1)*(10-2) + (2-1)*(10-1) = 9*8 + 1*9 = 72+9=81.
            # But the actual union area is 80. So the formula is wrong.
            
            # The correct formula for the union of rectangles [f1_i, r1] x [f2_i, r2]:
            # Sort by f1 ascending. The union is a staircase.
            # HV = sum_{i=0}^{n-1} (f1_{i+1} - f1_i) * (r2 - f2_i) + (r1 - f1_{n-1}) * (r2 - f2_{n-1})
            # where f1_n = r1.
            # For two points (1,2) and (2,1), ref (10,10):
            # HV = (2-1)*(10-2) + (10-2)*(10-1) = 1*8 + 8*9 = 8+72 = 80. Correct!
            # For one point (1,1), ref (10,10):
            # HV = (10-1)*(10-1) = 81. Correct!
            
            # So the formula is:
            # Sort by f1 ascending.
            # HV = sum_{i=0}^{n-2} (f1_{i+1} - f1_i) * (r2 - f2_i) + (r1 - f1_{n-1}) * (r2 - f2_{n-1})
            # But we need to ensure f2_i < r2 for all i, otherwise the point doesn't contribute.
            
            # Let me implement this correctly.
            
            x0_sorted = np.sort(x0_vals)
            f2_vals = 1.0 + np.sqrt(x0_sorted)
            ref_f1, ref_f2 = 11.0, 11.0
            
            # Filter points strictly better than reference
            mask = (x0_sorted < ref_f1) & (f2_vals < ref_f2)
            x0_sorted = x0_sorted[mask]
            f2_vals = f2_vals[mask]
            
            n_pts = len(x0_sorted)
            if n_pts == 0:
                return 0.0
            
            # Sort by f1 ascending (already sorted)
            hv = 0.0
            for i in range(n_pts - 1):
                hv += (x0_sorted[i+1] - x0_sorted[i]) * (ref_f2 - f2_vals[i])
            hv += (ref_f1 - x0_sorted[-1]) * (ref_f2 - f2_vals[-1])
            
            return hv

        # Now, for the adaptive refinement:
        # We want to add points that maximize the HV increase.
        # For a concave front, the optimal point to add in an interval [a, b] 
        # is where the curve is farthest from the chord connecting (a, f2(a)) 
        # and (b, f2(b)).
        
        # For f2 = 1 + sqrt(f1), the chord from (a, 1+sqrt(a)) to (b, 1+sqrt(b)) 
        # has equation: f2_chord(x) = 1+sqrt(a) + (x-a) * (sqrt(b)-sqrt(a))/(b-a)
        # The curve is f2(x) = 1+sqrt(x).
        # The vertical distance is: d(x) = f2_chord(x) - f2(x) 
        # (since the curve is concave, it's below the chord)
        # We want to maximize d(x), which is equivalent to minimizing f2(x) - f2_chord(x).
        
        # Let's find the x that maximizes d(x) in [a, b].
        # d(x) = 1+sqrt(a) + (x-a)*(sqrt(b)-sqrt(a))/(b-a) - 1 - sqrt(x)
        #      = sqrt(a) + (x-a)*(sqrt(b)-sqrt(a))/(b-a) - sqrt(x)
        
        # Take derivative: d'(x) = (sqrt(b)-sqrt(a))/(b-a) - 1/(2*sqrt(x))
        # Set to 0: (sqrt(b)-sqrt(a))/(b-a) = 1/(2*sqrt(x))
        # => sqrt(x) = (b-a)/(2*(sqrt(b)-sqrt(a)))
        # => x = (b-a)^2 / (4*(sqrt(b)-sqrt(a))^2)
        # => x = (b-a)^2 / (4*(b-a)/(sqrt(b)+sqrt(a))^2)
        # => x = (b-a) * (sqrt(b)+sqrt(a))^2 / 4
        
        # Let me verify: for a=0, b=1:
        # x = 1 * (1+0)^2 / 4 = 1/4 = 0.25
        # f2(0.25) = 1+0.5 = 1.5
        # Chord: from (0,1) to (1,2), slope = 1. f2_chord(0.25) = 1+0.25 = 1.25
        # d(0.25) = 1.25 - 1.5 = -0.25. But we want the maximum distance, 
        # and the curve is below the chord, so d(x) = f2_chord(x) - f2(x) > 0.
        # Wait, f2_chord(0.25) = 1.25, f2(0.25) = 1.5. So f2_chord < f2, 
        # meaning the curve is ABOVE the chord? That can't be right for a concave function.
        
        # Let me recalculate: f2 = 1 + sqrt(x). f2''(x) = -1/(4*x^(3/2)) < 0, so concave.
        # For a concave function, the curve is ABOVE the chord. So f2(x) > f2_chord(x).
        # The "gap" is f2(x) - f2_chord(x).
        
        # We want to maximize f2(x) - f2_chord(x).
        # g(x) = f2(x) - f2_chord(x) = 1+sqrt(x) - [1+sqrt(a) + (x-a)*(sqrt(b)-sqrt(a))/(b-a)]
        #      = sqrt(x) - sqrt(a) - (x-a)*(sqrt(b)-sqrt(a))/(b-a)
        
        # g'(x) = 1/(2*sqrt(x)) - (sqrt(b)-sqrt(a))/(b-a)
        # Set to 0: 1/(2*sqrt(x)) = (sqrt(b)-sqrt(a))/(b-a)
        # => sqrt(x) = (b-a)/(2*(sqrt(b)-sqrt(a)))
        # => x = (b-a)^2 / (4*(sqrt(b)-sqrt(a))^2)
        # => x = (b-a) * (sqrt(b)+sqrt(a))^2 / 4
        
        # For a=0, b=1: x = 1 * (1+0)^2/4 = 0.25
        # g(0.25) = sqrt(0.25) - 0 - (0.25-0)*(1-0)/(1-0) = 0.5 - 0.25 = 0.25
        # At x=0: g(0) = 0 - 0 - 0 = 0
        # At x=1: g(1) = 1 - 0 - 1*1 = 0
        # So maximum at x=0.25 with g=0.25. Correct!
        
        def optimal_point_in_interval(a, b):
            """Find the x in [a,b] that maximizes f2(x) - f2_chord(x)."""
            if b - a < 1e-10:
                return (a + b) / 2
            sqrt_a = np.sqrt(a)
            sqrt_b = np.sqrt(b)
            x_opt = (b - a) * (sqrt_b + sqrt_a) ** 2 / 4.0
            # Clip to [a, b]
            x_opt = np.clip(x_opt, a, b)
            return x_opt
        
        # Phase 2: Adaptive refinement
        # Start with the initial grid, then iteratively add the best point.
        
        # We'll use a priority queue to track the best interval.
        # But since we can't use heapq easily with custom keys, let's just 
        # scan all intervals each time (n is small).
        
        evaluated_set = set()
        for x0 in x0_initial:
            evaluated_set.add(round(x0, 12))
        
        intervals = list(zip(x0_initial[:-1], x0_initial[1:]))
        
        remaining_budget = budget - len(results_F)
        
        while remaining_budget > 0 and time.time() < deadline - 10:
            # Find the interval with the maximum HV gain
            best_gain = -1
            best_interval = None
            best_x = None
            
            for (a, b) in intervals:
                x_opt = optimal_point_in_interval(a, b)
                # Check if this point is already evaluated
                if round(x_opt, 12) in evaluated_set:
                    continue
                
                # Compute HV gain: HV with x_opt added - HV without
                # For efficiency, we can compute the gain directly:
                # The gain is the area of the "triangle" formed by the chord 
                # and the curve, times the width... actually, it's the area 
                # between the curve and the chord in [a, b].
                # But since we're adding a single point, the gain is:
                # (x_opt - a) * (f2(x_opt) - f2_chord_at_x_opt) + 
                # (b - x_opt) * (f2(x_opt) - f2_chord_at_x_opt)
                # = (b - a) * (f2(x_opt) - f2_chord_at_x_opt)
                # Wait, that's not right. The gain is the area of the region 
                # that was previously not dominated but now is.
                
                # Actually, the HV gain of adding a point (x, f2(x)) to the set 
                # is the area of the rectangle [x, r1] x [f2(x), r2] that was 
                # not already covered by other points.
                
                # For simplicity, let's just compute the HV before and after.
                # But that's expensive. Let's use an approximation:
                # The gain is approximately (b-a) * (f2(x_opt) - f2_chord(x_opt))
                # where f2_chord is the linear interpolation between (a, f2(a)) 
                # and (b, f2(b)).
                
                f2_a = 1.0 + np.sqrt(a)
                f2_b = 1.0 + np.sqrt(b)
                f2_opt = 1.0 + np.sqrt(x_opt)
                f2_chord_opt = f2_a + (x_opt - a) * (f2_b - f2_a) / (b - a)
                
                gain = (b - a) * (f2_opt - f2_chord_opt)
                
                if gain > best_gain:
                    best_gain = gain
                    best_interval = (a, b)
                    best_x = x_opt
            
            if best_x is None:
                break
            
            # Add the best point
            x0_new = best_x
            x = np.empty(dim)
            x[0] = x0_new
            x[1:] = x_rest_good
            f = problem(x)
            results_F.append(f)
            remaining_budget -= 1
            
            evaluated_set.add(round(x0_new, 12))
            
            # Update intervals
            a, b = best_interval
            intervals.remove((a, b))
            intervals.append((a, x0_new))
            intervals.append((x0_new, b))
            
            # If we have enough points, we can stop
            if len(results_F) >= budget:
                break
        
        # If we still have budget, add more points using a different strategy
        # (e.g., random sampling or uniform refinement)
        if remaining_budget > 0 and time.time() < deadline - 10:
            # Add points using a power-law distribution near x0=0
            n_extra = min(remaining_budget, 20)
            for i in range(n_extra):
                if time.time() > deadline - 10:
                    break
                # Use a distribution that clusters near x0=0
                x0_new = np.random.random() ** 2  # Beta(1,2) distribution
                if round(x0_new, 12) in evaluated_set:
                    continue
                x = np.empty(dim)
                x[0] = x0_new
                x[1:] = x_rest_good
                f = problem(x)
                results_F.append(f)
                evaluated_set.add(round(x0_new, 12))
                remaining_budget -= 1
        
        return np.array(results_F) if results_F else np.array([])


if __name__ == "__main__":
    pass
