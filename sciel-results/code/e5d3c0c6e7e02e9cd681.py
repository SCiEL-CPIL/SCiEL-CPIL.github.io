import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 310  # 20s safety margin before 330s kill

        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget

        # Reference point for scoring
        ref = np.array([3000.0, 0.06])

        # 1. Initialization: Stratified Sampling (Latin Hypercube)
        # Use 50 points to get a good initial spread.
        n_init = 50
        # Simple LHS: for each dimension, split range into n_init bins, pick one random point in each bin for that dimension, then shuffle.
        # Since dim=4 and n_init=50, this is a good approximation of a space-filling design.
        X_init = np.zeros((n_init, dim))
        for d in range(dim):
            # Generate n_init uniform random numbers in [0, 1]
            u = np.random.rand(n_init)
            # Map to bins: bin i is [i/n_init, (i+1)/n_init)
            # Pick a random point within the bin
            # Actually, simpler: sort u, then add random offset within the bin width?
            # Standard LHS:
            # 1. Create n_init random numbers in [0,1]
            # 2. Sort them
            # 3. Add uniform random offset in [0, 1/n_init]
            u_sorted = np.sort(u)
            offsets = np.random.rand(n_init) / n_init
            u_lhs = u_sorted + offsets
            X_init[:, d] = lo[d] + (hi[d] - lo[d]) * u_lhs

        F_init = np.array([problem(x) for x in X_init])
        
        X_all = list(X_init)
        F_all = list(F_init)
        
        evals_used = n_init

        # Helper: Compute exact hypervolume contribution for each point in the non-dominated set (2D)
        def compute_hv_contributions(F_nd):
            """
            F_nd: list of (f1, f2) tuples, all non-dominated.
            Returns list of HV contributions corresponding to each point in F_nd.
            """
            if not F_nd:
                return []
            
            # Sort by f1 ascending
            indices = sorted(range(len(F_nd)), key=lambda i: F_nd[i][0])
            sorted_F = [F_nd[i] for i in indices]
            
            n = len(sorted_F)
            contribs_sorted = [0.0] * n
            
            # The contribution of a point (f1_i, f2_i) in a 2D non-dominated set sorted by f1:
            # It is the area of the rectangle it "owns".
            # The right boundary is f1_{i+1} (or ref[0] if last).
            # The bottom boundary is f2_{i-1} (or -inf/0? No, in minimization, lower is better).
            # Wait, the region dominated by point i but not by any other point.
            # For point i (sorted by f1 asc):
            # Left bound: f1_{i-1} (or 0/-inf)
            # Right bound: f1_{i+1} (or ref[0])
            # Bottom bound: f2_{i+1}? No.
            
            # Let's visualize.
            # Points: (1, 9), (2, 8), (3, 7). Ref: (10, 10).
            # HV of (1,9): width = 2-1=1? No, next is 2. So width is 2-1=1?
            # Actually, the region dominated by (1,9) is x>1, y>9.
            # The region dominated by (2,8) is x>2, y>8.
            # The unique contribution of (1,9) is the part of its box not covered by (2,8).
            # Box of (1,9): [1, 10] x [9, 10]. Area = 9.
            # Box of (2,8): [2, 10] x [8, 10]. Area = 16.
            # Intersection: [2, 10] x [9, 10]. Area = 8.
            # Unique (1,9): 9 - 8 = 1.
            # Unique (2,8): Box [2,10]x[8,10] minus intersection with (3,7) [3,10]x[7,10].
            # Intersect: [3,10]x[8,10]. Area 7.
            # Unique (2,8): 16 - 7 = 9.
            # Unique (3,7): Box [3,10]x[7,10]. Area 7.
            # Total HV = 1 + 9 + 7 = 17.
            # Check total HV directly:
            # Union of boxes.
            # y from 9 to 10: width is 10-1 = 9. Area 9.
            # y from 8 to 9: width is 10-2 = 8. Area 8.
            # y from 7 to 8: width is 10-3 = 7. Area 7.
            # Total 24?
            
            # Let's re-calculate unique contribution formula.
            # For point i sorted by f1:
            # Left edge of its unique region: f1_{i-1} (or 0)
            # Right edge: f1_{i+1} (or ref[0])
            # Bottom edge: f2_i
            # Top edge: f2_{i-1} (or ref[1])?
            
            # Let's look at the "staircase".
            # The hypervolume is the area under the staircase formed by the points and the reference.
            # The contribution of point i is the area of the rectangle:
            # Width: f1_{i+1} - f1_i (where f1_{n} is ref[0])
            # Height: ref[1] - f2_i ? No.
            
            # Let's use the standard 2D HV contribution formula.
            # Sort points by f1 ascending: p0, p1, ..., p_{n-1}.
            # Add a virtual point p_n = ref.
            # The contribution of p_i is:
            # (f1_{i+1} - f1_i) * (ref[1] - f2_i)  -- Wait, this assumes f2 is decreasing?
            # In non-dominated set sorted by f1 ascending, f2 must be descending.
            # So f2_0 > f2_1 > ... > f2_{n-1}.
            
            # The area contributed by p_i is the rectangle defined by:
            # x from f1_i to f1_{i+1}
            # y from f2_i to ref[1]? No.
            # The region dominated by p_i is x > f1_i, y > f2_i.
            # The region dominated by p_{i+1} is x > f1_{i+1}, y > f2_{i+1}.
            # Since f1_{i+1} > f1_i and f2_{i+1} < f2_i:
            # The part of p_i's box that is NOT in p_{i+1}'s box is:
            # x in [f1_i, f1_{i+1}] AND y > f2_i.
            # PLUS
            # x > f1_{i+1} AND y in [f2_{i+1}, f2_i].
            
            # This is getting complex. Let's use the "area under curve" interpretation.
            # Total HV = sum over intervals.
            # For 2D, if we sort by f1, the HV is the sum of rectangles:
            # For i from 0 to n-1:
            # Width = f1_{i+1} - f1_i (with f1_n = ref[0])
            # Height = ref[1] - f2_i
            # Sum = sum( (f1_{i+1} - f1_i) * (ref[1] - f2_i) )
            
            # Let's test with (1,9), (2,8), (3,7), ref(10,10).
            # i=0: (2-1)*(10-9) = 1
            # i=1: (3-2)*(10-8) = 2
            # i=2: (10-3)*(10-7) = 21
            # Sum = 24. Matches previous calculation.
            
            # So the contribution of point i is NOT simply that term.
            # The term (f1_{i+1} - f1_i) * (ref[1] - f2_i) is the area of the rectangle
            # with corners (f1_i, ref[1]) and (f1_{i+1}, f2_i).
            # This rectangle is part of the HV.
            # Is it the *unique* contribution of point i?
            # If we remove point i, the HV changes.
            # The unique contribution of point i is the area that is covered by i but not by any other point.
            # In the sorted list, the unique contribution of p_i is:
            # (f1_{i+1} - f1_i) * (f2_{i-1} - f2_i) ? No.
            
            # Let's look at the "marginal" hypervolume contribution.
            # HV(S) - HV(S \ {p_i}).
            # For 2D, this is known to be:
            # (f1_{i+1} - f1_i) * (ref[1] - f2_i) + (f1_i - f1_{i-1}) * (f2_i - f2_{i-1}) ?
            # Actually, there is a simpler formula for the contribution of a point in a 2D non-dominated set.
            # Contribution of p_i = (f1_{i+1} - f1_i) * (ref[1] - f2_i) + (f1_i - f1_{i-1}) * (f2_{i-1} - f2_i)
            # Where f1_{-1} = -inf (or 0) and f1_n = ref[0], f2_{-1} = ref[1].
            # Wait, if i=0:
            # Term 1: (f1_1 - f1_0) * (ref[1] - f2_0)
            # Term 2: (f1_0 - f1_{-1}) * (f2_{-1} - f2_0) -> f1_{-1} is -inf? No, the domain starts at 0?
            # The reference point defines the upper bound. The lower bound is effectively 0 (or -inf, but f values are positive).
            # Let's assume the "ideal" point is (0,0) or similar, but the HV is calculated relative to ref.
            # The region is [0, ref[0]] x [0, ref[1]]? No, [f_min, ref[0]] x [f_min, ref[1]].
            # Usually HV is calculated in the box defined by the best point and ref.
            # But the scoring says "reference point [3000.0, 0.06]".
            # And "points strictly better than the reference point".
            # So the box is [0, 3000] x [0, 0.06]? Or [min_f, 3000] x [min_f, 0.06]?
            # Standard HV definition: The volume of the region dominated by the set and bounded by the reference point.
            # If we have point (1, 0.01), ref (3000, 0.06).
            # Region: x in [1, 3000], y in [0.01, 0.06].
            # Area: (3000-1) * (0.06-0.01).
            
            # Let's stick to the "remove and recalculate" logic for correctness, but optimize it.
            # With n <= 200, O(N^2) is fine.
            
            def calc_hv(points):
                # points: list of (f1, f2)
                valid = [p for p in points if p[0] < ref[0] and p[1] < ref[1]]
                if not valid:
                    return 0.0
                valid.sort(key=lambda x: x[0])
                hv = 0.0
                for i in range(len(valid)):
                    f1, f2 = valid[i]
                    f1_next = valid[i+1][0] if i+1 < len(valid) else ref[0]
                    width = f1_next - f1
                    height = ref[1] - f2
                    hv += width * height
                return hv

            # Calculate total HV
            total_hv = calc_hv(F_nd)
            
            # Calculate contribution of each point
            contribs = [0.0] * len(F_nd)
            for i in range(len(F_nd)):
                # Create a new list without point i
                # Since N is small, this is cheap
                F_nd_minus_i = [F_nd[j] for j in range(len(F_nd)) if j != i]
                hv_minus_i = calc_hv(F_nd_minus_i)
                contribs[i] = total_hv - hv_minus_i
                
            return contribs

        # Main Loop
        while evals_used < budget:
            if time.time() > deadline:
                break
            
            # 1. Get non-dominated indices
            # F_all is a list of lists/arrays
            nd_indices = []
            for i in range(len(F_all)):
                dominated = False
                for j in range(len(F_all)):
                    if i == j:
                        continue
                    # Check if j dominates i
                    if F_all[j][0] <= F_all[i][0] and F_all[j][1] <= F_all[i][1]:
                        if F_all[j][0] < F_all[i][0] or F_all[j][1] < F_all[i][1]:
                            dominated = True
                            break
                if not dominated:
                    nd_indices.append(i)
            
            if not nd_indices:
                break
                
            F_nd = [F_all[i] for i in nd_indices]
            
            # 2. Compute HV contributions
            contribs = compute_hv_contributions(F_nd)
            
            # 3. Select the point with the highest contribution
            max_contrib_idx = np.argmax(contribs)
            parent_idx_in_nd = max_contrib_idx
            global_parent_idx = nd_indices[parent_idx_in_nd]
            x_parent = np.array(X_all[global_parent_idx])
            
            # 4. Coordinate Descent
            # We have 4 dimensions.
            # We will try to improve the HV by moving in each coordinate direction.
            # Step size: Start with a fraction of the range.
            # Adaptive step size?
            # Let's use a fixed step size relative to the range, e.g., 5% of the range.
            # Or, we can try a few step sizes? No, budget is tight.
            # Let's try 1 step per dimension per iteration?
            # That would be 4 evals per iteration.
            # Budget 200 - 50 init = 150 evals left.
            # 150 / 4 = 37 iterations.
            # In each iteration, we pick the best HV contributor, perturb it in 4 directions,
            # evaluate all 4, and keep the one that results in the best HV (or just the best objective vector?).
            # Actually, we should evaluate the 4 candidates, and if any of them are non-dominated and improve the HV,
            # we add them to the population.
            
            # Strategy:
            # Generate 4 candidates: x_parent +/- step in each dimension.
            # Evaluate them.
            # If a candidate is non-dominated, add it to X_all, F_all.
            # This uses 4 evals per loop iteration.
            
            if budget - evals_used < 4:
                # If we have less than 4 evals left, just do random or stop?
                # Let's do 1 eval: random perturbation.
                noise = np.random.normal(0, 0.05 * (hi - lo))
                x_new = np.clip(x_parent + noise, lo, hi)
                f_new = problem(x_new)
                X_all.append(x_new)
                F_all.append(f_new)
                evals_used += 1
                continue

            step_size = 0.05 * (hi - lo)
            candidates = []
            for d in range(dim):
                # Positive direction
                x_pos = x_parent.copy()
                x_pos[d] += step_size[d]
                if x_pos[d] <= hi[d]:
                    candidates.append(x_pos)
                # Negative direction
                x_neg = x_parent.copy()
                x_neg[d] -= step_size[d]
                if x_neg[d] >= lo[d]:
                    candidates.append(x_neg)
            
            # We might have up to 8 candidates.
            # But we only have 4 evals left in this "batch" logic?
            # No, the loop condition is `while evals_used < budget`.
            # We should only evaluate as many as we have budget for.
            
            n_evals_this_step = min(len(candidates), budget - evals_used)
            
            # Shuffle candidates to ensure fairness if we can't evaluate all?
            # Or just take the first n_evals_this_step?
            # Since we are doing coordinate descent, maybe prioritize dimensions?
            # Let's just evaluate the first n_evals_this_step.
            
            for i in range(n_evals_this_step):
                x_new = candidates[i]
                f_new = problem(x_new)
                X_all.append(x_new)
                F_all.append(f_new)
                evals_used += 1
                
                if time.time() > deadline:
                    return None

        return None
