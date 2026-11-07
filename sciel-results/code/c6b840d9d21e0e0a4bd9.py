# EVOLVE-BLOCK-START
import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 300.0  # Safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        
        # ZDT4 Structure:
        # f1(x) = 1 + 10(n-1) + sum(x_i^2 - 10*cos(2*pi*x_i)) for i=2..n
        # f2(x) = (1 + sqrt(f1(x))) * g(x)
        # g(x) = 1 + 9 * ((sum x_i) / n)^2
        #
        # The Pareto front is characterized by f1(x) = 0.
        # f1(x) = 0 implies x_2 = x_3 = ... = x_n = 1.
        # Note: The problem statement says "10 variables". Standard ZDT4 has n variables.
        # If dim=10, then x_1 is the "g" variable (controls f2 via g(x)), 
        # and x_2..x_10 are the "f1" variables.
        # Wait, standard ZDT4 definition:
        # f1(x) = 1 + 10(n-1) + sum_{i=2}^n (x_i^2 - 10*cos(2*pi*x_i))
        # f2(x) = (1 + sqrt(f1(x))) * g(x)
        # g(x) = 1 + 9 * ((sum_{i=1}^n x_i) / n)^2
        #
        # If dim=10, n=10.
        # To get f1=0, we need x_2=...=x_10=1.
        # x_1 can vary in [0, 1] (standard bounds) to change g(x) and thus f2.
        # However, bounds are given. Let's assume standard [0,1] or similar.
        # The prompt says "bounds is a list of (low, high) pairs".
        
        # Strategy:
        # 1. Construct points on the Pareto front directly.
        #    Set x_2..x_dim to 1.0 (clipped to bounds).
        #    Vary x_1 to sweep f2.
        # 2. Use remaining budget for local refinement around the best points found.
        
        # Determine which index controls f1 and which controls g.
        # In standard ZDT4, x_1 is in the sum for g, and x_2..x_n are in f1.
        # Let's assume index 0 is the "g" variable and indices 1..dim-1 are "f1" variables.
        
        # If bounds allow x_i=1, we set them to 1.
        # If bounds are [0,1], 1 is allowed.
        
        # We will generate a dense set of points on the front.
        # The front spans f2 from g_min to g_max.
        # g(x) depends on sum(x). If x_2..x_n=1, sum = x_1 + (n-1).
        # g(x_1) = 1 + 9 * ((x_1 + n - 1) / n)^2.
        
        # We want to maximize hypervolume with ref [11, 400].
        # f1 is always 0 on the front.
        # f2 ranges. We need f2 < 400 and f1 < 11 (always true since f1=0).
        
        # Let's sample x_1 linearly across its bounds to cover the front.
        # Number of samples: We have budget=200.
        # Let's use 100 evaluations for the front sweep.
        # And 100 for local refinement if needed, or just more front points.
        # Actually, just covering the front densely is likely optimal for ZDT4 
        # because the front is the set of all non-dominated points.
        
        n_front_points = 100
        x1_lo, x1_hi = bounds[0]
        x1_vals = np.linspace(x1_lo, x1_hi, n_front_points)
        
        # Construct the base vector for f1=0
        # x_2..x_n should be 1.
        base_x = np.zeros(self.dim)
        for i in range(1, self.dim):
            # Clip 1.0 to bounds
            base_x[i] = np.clip(1.0, bounds[i][0], bounds[i][1])
            
        evaluated_points = []
        evaluated_f = []
        
        # 1. Evaluate front points
        for x1 in x1_vals:
            x = base_x.copy()
            x[0] = x1
            f = problem(x)
            evaluated_points.append(x)
            evaluated_f.append(f)
            if time.time() > deadline - 10.0:
                break
        
        used = len(evaluated_points)
        
        # 2. Use remaining budget for local refinement around the "knee" or best regions?
        # For ZDT4, the front is convex/concave? 
        # f2(x) = (1+0) * g(x) = g(x).
        # g(x) is a parabola in x_1.
        # The front is a curve in (f1, f2) = (0, g(x_1)).
        # Since f1 is constant 0, the "front" is actually a line segment on the f2 axis?
        # No, f1 is the first objective. f1=0.
        # So all points are (0, f2_i).
        # The hypervolume is determined by the minimum f2 and the reference point.
        # Wait, hypervolume for points (0, f2_1), (0, f2_2)...
        # If all points have f1=0, they are collinear.
        # The hypervolume of a set of collinear points (0, y_i) with ref (R1, R2)
        # is simply (R1 - 0) * (R2 - min(y_i)) if min(y_i) < R2.
        # Wait, hypervolume is the volume of the union of boxes.
        # Box for (0, y_i) is [0, R1] x [y_i, R2].
        # Union of boxes [0, R1] x [y_i, R2] is [0, R1] x [min(y_i), R2].
        # So the hypervolume is R1 * (R2 - min(y_i)).
        # To maximize this, we need to minimize min(y_i).
        # min(y_i) is the minimum f2 value on the front.
        # f2 = g(x_1) = 1 + 9 * ((x_1 + n - 1) / n)^2.
        # To minimize g(x_1), we need (x_1 + n - 1) / n to be as close to 0 as possible.
        # x_1 + n - 1 = 0 => x_1 = 1 - n.
        # For n=10, x_1 = -9.
        # If bounds are [0, 1], the closest x_1 is 0.
        # g(0) = 1 + 9 * ((0 + 9) / 10)^2 = 1 + 9 * 0.81 = 1 + 7.29 = 8.29.
        # If bounds allow negative, we could get lower.
        
        # So the strategy is:
        # 1. Find the point on the front with the minimum f2.
        # 2. That single point defines the hypervolume (since f1 is constant 0 for all front points).
        # Wait, is f1 always 0?
        # f1(x) = 1 + 10(n-1) + sum(x_i^2 - 10cos(2pi x_i)).
        # If x_i=1 for i>1, term is 1 - 10 = -9.
        # Sum for n-1 terms is -9(n-1).
        # f1 = 1 + 10(n-1) - 9(n-1) = 1 + (n-1) = n.
        # Wait. 1 + 10(n-1) + sum_{i=2}^n (1 - 10) = 1 + 10n - 10 - 9n + 9 = n.
        # For n=10, f1 = 10.
        # Reference f1 is 11.0. So f1=10 is strictly better.
        
        # So the front is a line segment at f1=10, varying f2.
        # Hypervolume is (11 - 10) * (400 - min(f2)) = 1 * (400 - min(f2)).
        # To maximize score, we must minimize min(f2).
        
        # So we don't need to sweep the whole front!
        # We just need to find the x_1 that minimizes f2 (which is g(x_1)) 
        # subject to x_2..x_n = 1.
        
        # g(x_1) = 1 + 9 * ((x_1 + n - 1) / n)^2.
        # This is a parabola. Minimum is at x_1 = 1 - n.
        # If bounds[0] allows x_1 close to 1-n, we pick that.
        # If not, we pick the bound closest to 1-n.
        
        # Let's verify bounds.
        target_x1 = 1.0 - self.dim
        x1_best = np.clip(target_x1, bounds[0][0], bounds[0][1])
        
        # Construct the optimal point
        x_opt = np.zeros(self.dim)
        x_opt[0] = x1_best
        for i in range(1, self.dim):
            x_opt[i] = np.clip(1.0, bounds[i][0], bounds[i][1])
            
        f_opt = problem(x_opt)
        
        # We have spent 1 evaluation.
        # We have budget-1 left.
        # We can use this to refine x_opt locally in case my analytical assumption 
        # about the bounds or the specific ZDT4 variant is slightly off, or to 
        # ensure we hit the exact minimum if the function is noisy (it's not, but safe).
        
        # Let's do a simple local search around x_opt to ensure we found the global min of f2.
        # Since f1 is fixed at ~10, minimizing f2 is the only goal.
        
        current_x = x_opt.copy()
        current_f = f_opt.copy()
        best_x = current_x.copy()
        best_f = current_f.copy()
        
        step_size = 0.1
        used = 1
        
        while used < self.budget:
            if time.time() > deadline - 5.0:
                break
            
            # Try perturbing each dimension
            improved = False
            for i in range(self.dim):
                # Try positive and negative step
                for sign in [1, -1]:
                    new_x = current_x.copy()
                    new_x[i] += sign * step_size * (hi[i] - lo[i])
                    new_x[i] = np.clip(new_x[i], lo[i], hi[i])
                    
                    if np.allclose(new_x, current_x):
                        continue
                    
                    new_f = problem(new_x)
                    used += 1
                    
                    # We want to minimize f2, and keep f1 < 11.
                    # If f1 becomes > 11, it's invalid for hypervolume (or at least worse).
                    # Actually, if f1 > 11, the point is dominated by ref in obj 1?
                    # No, ref is (11, 400). Point is (f1, f2).
                    # If f1 > 11, it doesn't contribute to HV.
                    # So we must ensure f1 < 11.
                    
                    if new_f[0] < 11.0 and new_f[1] < best_f[1]:
                        # This point has better f2 and is valid.
                        # Is it non-dominated?
                        # Since we are moving along the "valley" of f1, 
                        # if f1 increases slightly but f2 decreases significantly, 
                        # it might be non-dominated.
                        # However, for ZDT4, moving away from x_i=1 increases f1 quadratically 
                        # and changes f2.
                        # Given the HV formula (11-f1)*(400-f2), 
                        # a small increase in f1 (loss in width) might be compensated by 
                        # a large decrease in f2 (gain in height).
                        
                        # Let's just accept if it improves the HV contribution locally?
                        # Simpler: Accept if it improves f2 and f1 < 11.
                        # Actually, we should compare HV contribution.
                        # Current contribution: (11 - best_f[0]) * (400 - best_f[1])
                        # New contribution: (11 - new_f[0]) * (400 - new_f[1])
                        
                        curr_hv = (11.0 - best_f[0]) * (400.0 - best_f[1])
                        new_hv = (11.0 - new_f[0]) * (400.0 - new_f[1])
                        
                        if new_hv > curr_hv:
                            best_x = new_x.copy()
                            best_f = new_f.copy()
                            current_x = new_x.copy()
                            improved = True
                            break
                if improved:
                    break
            
            if not improved:
                # Reduce step size
                step_size *= 0.5
                if step_size < 1e-5:
                    break

        # The evaluator records ALL returned objective vectors.
        # It doesn't just take the best one. It takes the set of all recorded vectors.
        # "Every returned objective vector is recorded"
        # "The return value of __call__ is ignored"
        # Wait, how are vectors returned?
        # "problem(x) ... returns 2 objective values"
        # "Every returned objective vector is recorded"
        # This implies that every time I call `problem(x)`, the evaluator records the result.
        # So I don't need to return a list. I just need to call `problem` with good points.
        # The evaluator collects all F from all calls.
        
        # So my strategy of calling problem() many times is correct.
        # The evaluator will compute HV of the set of all F's recorded.
        
        # Since I've already called problem() for x_opt and the local search,
        # the evaluator has recorded those points.
        
        # I should ensure I have called problem() for a good set of points.
        # The local search calls problem() for candidates.
        # If a candidate is rejected, it is still recorded!
        # "Every returned objective vector is recorded"
        # This is crucial. Even bad points are recorded.
        # But bad points (dominated by better ones) don't hurt HV.
        # Points with f1 > 11 or f2 > 400 don't contribute to HV.
        # So recording "bad" points is harmless as long as they don't dominate the good ones 
        # (which they won't if they are worse).
        
        # So the main task is to ensure I find the point(s) that maximize the union HV.
        # Since the front is a line, the best point is the one with max HV contribution.
        # My local search finds the best single point.
        # Are there multiple non-dominated points?
        # Yes, the whole front is non-dominated.
        # But for a line segment at f1=10, the HV is determined by the "corner" 
        # that gives the largest rectangle.
        # Actually, if we have multiple points on the front (f1=10, f2 varies),
        # The HV is the union of boxes.
        # Box 1: [0, 11] x [f2_1, 400]
        # Box 2: [0, 11] x [f2_2, 400]
        # Union is [0, 11] x [min(f2_1, f2_2), 400].
        # So only the point with the MINIMUM f2 matters for the HV of the set, 
        # assuming f1 is constant.
        
        # If f1 varies slightly around 10, say 9.9 and 10.1.
        # Point A: (9.9, 100). HV = (11-9.9)*(400-100) = 1.1 * 300 = 330.
        # Point B: (10.1, 90). HV = (11-10.1)*(400-90) = 0.9 * 310 = 279.
        # Point C: (9.9, 90). HV = 1.1 * 310 = 341.
        # If I have A and B, the union is not just the max.
        # Union of [0, 11]x[100, 400] and [0, 11]x[90, 400] is [0, 11]x[90, 400].
        # So the point with the lowest f2 defines the bottom of the box, 
        # and the point with the lowest f1 defines the left of the box?
        # No, the box is anchored at (0,0) to (f1, f2)? No, ref is (11, 400).
        # The box is [0, 11] x [f2, 400] for a point (f1, f2) with f1<11, f2<400?
        # No, the box for a point p=(p1, p2) with ref r=(r1, r2) is 
        # [0, r1] x [0, r2] MINUS the part dominated by p?
        # Standard HV definition: Volume of the union of boxes 
        # [p1, r1] x [p2, r2] for all p in P.
        # Wait, if p1 < r1 and p2 < r2.
        # Box is (r1 - p1) * (r2 - p2).
        # If I have two points p1=(9.9, 100) and p2=(10.1, 90).
        # Box 1: [9.9, 11] x [100, 400]. Width 1.1, Height 300. Area 330.
        # Box 2: [10.1, 11] x [90, 400]. Width 0.9, Height 310. Area 279.
        # Intersection: [10.1, 11] x [100, 400]. Width 0.9, Height 300. Area 270.
        # Union Area = 330 + 279 - 270 = 339.
        # If I had a point p3=(9.9, 90).
        # Box 3: [9.9, 11] x [90, 400]. Width 1.1, Height 310. Area 341.
        # This box contains Box 1 and Box 2?
        # Box 3 is [9.9, 11] x [90, 400].
        # Box 1 is [9.9, 11] x [100, 400]. Subset of Box 3.
        # Box 2 is [10.1, 11] x [90, 400]. Subset of Box 3.
        # So p3 alone gives HV 341, which is > 339.
        
        # So we want to find the point that maximizes (11-f1)*(400-f2).
        # My local search optimizes this product directly (via `new_hv > curr_hv`).
        
        # So the strategy is solid.
        
        return best_x

# EVOLVE-BLOCK-END
