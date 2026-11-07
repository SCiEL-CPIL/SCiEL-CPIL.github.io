import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t_start = time.time()
        deadline = t_start + 295.0  # Safety margin before 300s limit
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        n_dim = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # Population size 20 allows 10 full generations (20 offspring + 20 parents = 40, select 20)
        # 200 budget / 20 per gen = 10 generations.
        pop_size = 20
        
        # 1. Initialization
        X = np.random.uniform(lo, hi, size=(pop_size, n_dim))
        F = np.array([problem(x) for x in X])
        used = pop_size
        
        def hypervolume_2d(points, ref):
            """
            Calculate 2D hypervolume.
            points: Nx2 array
            ref: 2-element array [ref_x, ref_y]
            Returns hypervolume.
            """
            if len(points) == 0:
                return 0.0
            # Filter points dominated by reference? No, reference is the anchor.
            # Points must be better than ref (lower values).
            # Sort by x
            sorted_points = sorted(points, key=lambda p: p[0])
            hv = 0.0
            # For each point, calculate area of rectangle where it is the "max" in x and "min" in y?
            # Actually, standard sweep line:
            # Sort by x ascending.
            # For each point, the contribution is (next_x - curr_x) * (ref_y - curr_y) if curr_y < ref_y and next_x < ref_x
            # Wait, the reference point defines the corner.
            # If we minimize, the front is in the bottom-left.
            # Ref is top-right.
            # Area = sum over sorted points (by x) of (x_{i+1} - x_i) * (ref_y - y_i) ?
            # Let's verify.
            # Points: (1, 1), (2, 2). Ref: (3, 3).
            # Sorted: (1, 1), (2, 2).
            # i=0: x_next=2, x_curr=1. width=1. height = ref_y - y_curr = 3-1=2. Area = 2.
            # i=1: x_next=3 (ref_x), x_curr=2. width=1. height = ref_y - y_curr = 3-2=1. Area = 1.
            # Total = 3.
            # Geometric check:
            # Rect 1 (1,1) to (2,3) ? No.
            # The union of boxes [1,3]x[1,3] and [2,3]x[2,3]?
            # Box 1: width 2, height 2. Area 4.
            # Box 2: width 1, height 1. Area 1.
            # Union: (3x3 square from 0? No, from point to ref).
            # Let's use the standard formula:
            # HV = sum_{i} (x_{i+1} - x_i) * (ref_y - y_i) where x_N = ref_x.
            # Yes, this works if points are sorted by x.
            
            # Ensure points are strictly better than ref?
            # The problem says "Only points strictly better than the reference point in every objective count."
            # So we should filter points first.
            valid_points = [p for p in points if np.all(p < ref)]
            if not valid_points:
                return 0.0
            
            # Sort by first objective
            valid_points.sort(key=lambda p: p[0])
            
            total_hv = 0.0
            current_x = valid_points[0][0]
            for i in range(len(valid_points)):
                p_curr = valid_points[i]
                if i < len(valid_points) - 1:
                    x_next = valid_points[i+1][0]
                else:
                    x_next = ref[0]
                
                # Width
                w = x_next - current_x
                # Height
                h = ref[1] - p_curr[1]
                
                if w > 0 and h > 0:
                    total_hv += w * h
                
                # Update current_x to the next point's x?
                # No, the "step" is from x_i to x_{i+1}.
                # The y value used is y_i because y_i is lower than y_{i+1} (since they are non-dominated and sorted by x, y must decrease).
                # Wait, if they are non-dominated, sorting by x ascending implies y descending.
                # So y_i > y_{i+1}?
                # Minimizing: (1, 10) and (10, 1).
                # Sorted by x: (1, 10), (10, 1).
                # i=0: x_curr=1, x_next=10. w=9. h = ref[1] - 10.
                # i=1: x_curr=10, x_next=ref[0]. w=ref[0]-10. h = ref[1] - 1.
                # This seems correct.
                
                current_x = p_curr[0] # Not strictly needed if we use index, but good for clarity.
                # Actually, the loop logic above uses `current_x` which starts at first point.
                # In next iteration, we need x_next from prev step?
                # Let's re-structure.
            
            # Re-implementation for clarity
            total_hv = 0.0
            for i in range(len(valid_points)):
                p_curr = valid_points[i]
                # Next x is either next point's x or ref_x
                if i + 1 < len(valid_points):
                    x_next = valid_points[i+1][0]
                else:
                    x_next = ref[0]
                
                w = x_next - p_curr[0]
                h = ref[1] - p_curr[1]
                
                if w > 0 and h > 0:
                    total_hv += w * h
            return total_hv

        def hypervolume_3d_contribution(points, ref):
            """
            Approximate or exact 3D HV?
            With N=20, exact 3D HV is O(N^3) or O(N^2 log N).
            For selection in the last front, we only need to compare marginal contributions.
            A simple approximation for marginal contribution in 3D:
            For a point p, its contribution is the volume of the box it "owns".
            This is complex. 
            Alternative: Use a simple scalarization or projected HV for selection?
            Given the low N, we can compute exact HV of the set.
            But doing this greedily k times is O(k * N * HV_cost).
            HV_cost for N=20 in 3D is very small.
            """
            pass

        def calculate_hv_3d(points, ref):
            """
            Calculate exact 3D hypervolume for a small set of points.
            points: Nx3
            ref: 3
            """
            if len(points) == 0:
                return 0.0
            
            # Filter valid points
            valid = []
            for p in points:
                if np.all(p < ref):
                    valid.append(p)
            
            if not valid:
                return 0.0
            
            # Use a recursive divide and conquer or simple inclusion-exclusion?
            # For N<=20, a simple algorithm:
            # Project to 2D?
            # Standard algorithm:
            # 1. Sort by x.
            # 2. For each interval between x_i and x_{i+1} (and x_N to ref_x),
            #    the active points are those with x <= x_mid.
            #    The volume in this slice is (x_{i+1} - x_i) * HV_2D(active_points_yz, ref_yz).
            
            # Sort by x
            valid.sort(key=lambda p: p[0])
            
            total_vol = 0.0
            
            # We need to consider the "steps" in x.
            # x coordinates: valid[0].x, valid[1].x, ..., ref.x
            # For each interval, the set of points "active" (with x < interval_end) contributes to the 2D HV of their y,z projections.
            
            # To be efficient:
            # As we move x from left to right, points enter the active set.
            # We maintain the 2D HV of the active set.
            
            # However, calculating 2D HV from scratch is O(N^2).
            # Total O(N^3) for N=20 is 8000 ops, which is trivial.
            
            # Let's just iterate through unique x coordinates
            x_coords = [p[0] for p in valid] + [ref[0]]
            
            # We need to know which points are "active" for each x-slice.
            # A point is active in slice [x_i, x_{i+1}] if x_point < x_{i+1}.
            # Actually, the "step" function approach:
            # The hypervolume is the integral of the 2D area of the projection of the "dominated region" up to x.
            # Wait, standard formulation:
            # HV = sum over disjoint boxes.
            # Let's use the property:
            # HV_3D(P) = sum_{i} (x_{i+1} - x_i) * HV_2D(Proj_yz(P_{<=i}), Ref_yz)
            # Where P_{<=i} are points with x <= x_{i+1} (effectively points in the "left" part).
            # More precisely, for the interval [x_i, x_{i+1}], the cross-section at any x in this interval is the set of points with x_point <= x.
            # Since x_point values are discrete, the set is constant for x in [x_i, x_{i+1}) if we define x_i as the i-th point's x.
            # Let x_0 = -inf, x_1 = p_1.x, ..., x_k = p_k.x, x_{k+1} = ref.x.
            # For x in [x_i, x_{i+1}), the active points are {p_j | p_j.x < x_{i+1}}.
            # Actually, if x_i is the x of point i, then for x in [x_i, x_{i+1}), the points with x_j <= x are p_1...p_i.
            # So the area is HV_2D({p_1.y, p_1.z} ... {p_i.y, p_i.z}, ref_yz).
            
            # Let's implement this.
            # Sort points by x.
            active_2d_points = []
            prev_x = valid[0][0]
            
            # We need to calculate HV_2D for the set of points with x <= current_x.
            # Let's iterate through the points.
            
            # Optimization: Just compute HV_2D for the prefix sets.
            # N=20, so 20 calls to HV_2D (which is O(N^2) sort + O(N) sum) is fine.
            
            # However, we can be smarter.
            # Let's just do the simple loop.
            
            # x_bounds for slices:
            # Slice 0: [valid[0].x, valid[1].x] (if len > 1) else [valid[0].x, ref.x]
            # Active points for slice i: valid[0..i-1]?
            # Let's define slices by the x-coordinates of the points and the ref.
            
            xs = [p[0] for p in valid] + [ref[0]]
            
            # For each interval [xs[i], xs[i+1]], the points contributing are valid[0..i] ?
            # If x is in [xs[i], xs[i+1]), the points with x_p < x are those with x_p <= xs[i] (since xs[i] is a point's x).
            # Wait, if point i has x = xs[i], it is active for x > xs[i].
            # So for interval [xs[i], xs[i+1]), the active points are valid[0..i].
            
            for i in range(len(xs) - 1):
                width = xs[i+1] - xs[i]
                if width <= 0:
                    continue
                
                # Active points: valid[0..i]
                # Their y,z projections
                pts_yz = np.array([valid[j][1:] for j in range(i+1)])
                
                # Calculate 2D HV of these projections with respect to ref[1:]
                hv_2d = hypervolume_2d(pts_yz, ref[1:])
                
                total_vol += width * hv_2d
            
            return total_vol

        def non_dominated_sort(F):
            n = len(F)
            S = [[] for _ in range(n)]
            n_dom = [0] * n
            rank = [0] * n
            
            for i in range(n):
                for j in range(i + 1, n):
                    if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                        S[i].append(j)
                        n_dom[j] += 1
                    elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                        S[j].append(i)
                        n_dom[i] += 1
            
            fronts = []
            current_front = [i for i in range(n) if n_dom[i] == 0]
            for i in current_front:
                rank[i] = 0
            
            k = 0
            while current_front:
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in S[i]:
                        n_dom[j] -= 1
                        if n_dom[j] == 0:
                            rank[j] = k + 1
                            next_front.append(j)
                current_front = next_front
                k += 1
            return fronts, rank

        ref_point = np.array([3.0, 5.0, 7.0])

        # Main Loop
        while used < budget:
            if time.time() > deadline:
                break
            
            # Generate pop_size offspring
            offspring_X = []
            offspring_F = []
            
            for _ in range(pop_size):
                if used >= budget:
                    break
                if time.time() > deadline:
                    break
                    
                # DE/rand/1
                r1, r2, r3 = np.random.choice(pop_size, 3, replace=False)
                
                # Weight F
                F_weight = 0.5 + 0.5 * np.random.rand() # 0.5 to 1.0
                C_rate = 0.5
                
                v = X[r1] + F_weight * (X[r2] - X[r3])
                
                # Crossover (Binomial)
                # Random crossover points
                j_rand = np.random.randint(n_dim)
                mask = np.random.rand(n_dim) < C_rate
                mask[j_rand] = True # Ensure at least one gene is from v
                
                child = np.where(mask, v, X[r1])
                child = np.clip(child, lo, hi)
                
                fv = problem(child)
                used += 1
                offspring_X.append(child)
                offspring_F.append(fv)
            
            if not offspring_X:
                break
            
            offspring_X = np.array(offspring_X)
            offspring_F = np.array(offspring_F)
            
            # Combine
            X_comb = np.vstack([X, offspring_X])
            F_comb = np.vstack([F, offspring_F])
            
            # Non-dominated Sort
            fronts, ranks = non_dominated_sort(F_comb)
            
            # Selection
            selected_idx = []
            for front in fronts:
                if len(selected_idx) + len(front) <= pop_size:
                    selected_idx.extend(front)
                else:
                    remaining = pop_size - len(selected_idx)
                    if remaining <= 0:
                        break
                    
                    # Hypervolume-based selection for the last front
                    # We have 'front' indices in F_comb.
                    # We want to select 'remaining' individuals from this front
                    # to maximize HV(Selected_So_Far U Selected_From_Front).
                    
                    # Base set: all individuals in selected_idx
                    base_F = F_comb[selected_idx] if selected_idx else np.empty((0, n_obj))
                    
                    # Candidate set: individuals in 'front'
                    cand_indices = list(front)
                    cand_F = F_comb[cand_indices]
                    
                    # Greedy selection
                    local_selected_idx = []
                    local_selected_F = []
                    
                    for _ in range(remaining):
                        best_idx_in_front = -1
                        best_hv = -1
                        
                        # Current base for HV calculation includes previously selected from this front
                        current_base_F = np.vstack([base_F, local_selected_F]) if local_selected_F else base_F
                        
                        for j, idx in enumerate(cand_indices):
                            if idx in local_selected_idx:
                                continue
                            
                            # Trial set
                            trial_F = np.vstack([current_base_F, cand_F[j]])
                            
                            # Calculate HV
                            hv = calculate_hv_3d(trial_F, ref_point)
                            
                            if hv > best_hv:
                                best_hv = hv
                                best_idx_in_front = idx
                                best_cand_F_row = cand_F[j]
                        
                        if best_idx_in_front != -1:
                            local_selected_idx.append(best_idx_in_front)
                            local_selected_F.append(best_cand_F_row)
                        else:
                            break
                    
                    selected_idx.extend(local_selected_idx)
                    break
            
            if not selected_idx:
                break
                
            X = X_comb[selected_idx]
            F = F_comb[selected_idx]
            
        return X
