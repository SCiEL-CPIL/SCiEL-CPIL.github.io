def hv_3d(points, ref):
    # points: list of tuples (x, y, z)
    # ref: tuple (rx, ry, rz)
    if not points:
        return 0.0
    # Filter points dominated by ref (all must be < ref)
    # The problem states points strictly better than ref count.
    # We assume points in memory are already filtered or we filter here.
    
    # Sort by x
    points = sorted(points, key=lambda p: p[0])
    
    # We can use the "slice" method.
    # Take the point with smallest x (p0).
    # It contributes a box from p0 to the "envelope" of the rest.
    # This is tricky.
    
    # Alternative: Inclusion-Exclusion or simple recursive splitting.
    # Let's use a standard recursive algorithm for 3D HV.
    
    def _hv(pts, ref):
        if not pts:
            return 0.0
        # Find the point that minimizes x, then y, then z?
        # Actually, standard algorithm:
        # 1. Sort by x.
        # 2. Let p0 be the point with min x.
        # 3. The volume is (ref[0] - p0[0]) * HV_2D(pts_with_x > p0[0], ref_yz) + ...?
        # No, that's for 2D.
        
        # Let's use the algorithm from "Fast Hypervolume Calculation".
        # Or simply, since N is small (<= 40 during pruning), we can use a simpler O(N^3) approach?
        # Actually, let's implement a robust 3D HV.
        
        # Sort by x
        pts = sorted(pts, key=lambda p: p[0])
        
        # We iterate through points, maintaining a "skyline" in y-z.
        # This is effectively a 2D HV calculation for each x-slice.
        
        total_hv = 0.0
        # We need to handle the "active" set of points for each x-interval.
        
        # A simpler approach for N<50:
        # Recursive:
        # Find point p with min x.
        # Split pts into P1 (x <= p.x) and P2 (x > p.x)? No, p is unique min x.
        # The box defined by p and ref is (ref.x - p.x) * (ref.y - p.y) * (ref.z - p.z).
        # But this box overlaps with others.
        
        # Let's use the "Inclusion-Exclusion" on the sorted list?
        # Or better, use the algorithm:
        # HV(S) = sum over i in S of (ref.x - x_i) * HV_2D(S_i, ref_y, ref_z) ... where S_i are points with x > x_i?
        # No, that's not quite right because of y-z interactions.
        
        # Correct Recursive Step for 3D:
        # 1. Sort by x.
        # 2. Let p0 be first point (min x).
        # 3. Let S' be all points with x > p0.x.
        # 4. The contribution of p0 is the volume of the box [p0.x, ref.x] x [p0.y, ref.y] x [p0.z, ref.z] MINUS the volume occupied by S' in that box.
        # This is getting complex to implement correctly from scratch in a short time.
        
        # Alternative: Use the fact that N is small.
        # We can compute HV by summing the volume of the "cells" in a grid? No, exact is needed.
        
        # Let's use a known simple recursive implementation for 3D.
        def hv_recursive(pts, ref):
            if not pts:
                return 0.0
            # Sort by x
            pts = sorted(pts, key=lambda p: p[0])
            # The point with min x is pts[0]
            p0 = pts[0]
            rest = pts[1:]
            
            # The volume is the box from p0 to ref, minus the part covered by rest.
            # This is equivalent to:
            # (ref[0] - p0[0]) * HV_2D(rest, ref[1], ref[2])  ??? No.
            
            # Let's look at the 2D case first.
            # HV_2D(S, ref):
            # Sort by x.
            # Sum (ref.x - x_i) * (ref.y - y_i) for points on the skyline?
            # No, for 2D, if we sort by x, and keep only non-dominated (skyline),
            # the HV is sum of (x_{i+1} - x_i) * (ref.y - y_i) ...
            
            # For 3D, we can reduce to 2D.
            # Sort by x.
            # Iterate through points.
            # Maintain a list of "active" 2D points (y, z) for the current x-slice.
            # When we move to the next x, we add the new point's (y,z) to the active set.
            # We need to calculate the HV of the active set in 2D, multiplied by the x-width.
            
            # Algorithm:
            # 1. Sort points by x.
            # 2. Initialize active_2d = []
            # 3. total_hv = 0
            # 4. prev_x = ref[0]
            # 5. Iterate points in REVERSE order of x (from large x to small x)?
            #    If we go from small x to large x:
            #    The "box" added by point i is from x_i to x_{i+1} (or ref.x).
            #    The height of this box is determined by the 2D HV of the points with x >= x_i?
            
            # Let's try this:
            # Sort by x ascending.
            # Let x_coords = [p[0] for p in pts] + [ref[0]]
            # We sweep x from ref[0] down to min_x.
            # At each interval [x_i, x_{i-1}], the cross-section is the 2D HV of points with x <= x_i?
            
            # Actually, standard algorithm:
            # Sort by x.
            # HV = sum_{i} (x_{i+1} - x_i) * HV_2D(S_i, ref_y, ref_z)
            # Where S_i is the set of points with x <= x_i?
            # No, S_i should be the points that are "active" in that slice.
            # If we sort by x ascending: p1, p2, ... pk.
            # Interval 1: [p1.x, p2.x]. Active points: p1.
            # Interval 2: [p2.x, p3.x]. Active points: p1, p2.
            # ...
            # Interval k: [pk.x, ref.x]. Active points: p1...pk.
            
            # So:
            # 1. Sort pts by x.
            # 2. Create list of x breakpoints: [p1.x, p2.x, ..., pk.x, ref.x]
            # 3. For each interval j, the active set is pts[0:j+1].
            # 4. Calculate HV_2D(active_set, ref.y, ref.z).
            # 5. Multiply by width (x_{j+1} - x_j).
            # 6. Sum.
            
            # This requires calculating HV_2D many times.
            # HV_2D is O(N log N) or O(N^2).
            # With N=20, this is very fast.
            
            return _hv_3d_impl(pts, ref)
            
        def _hv_2d(pts, ref_y, ref_z):
            # pts: list of (y, z)
            if not pts:
                return 0.0
            # Sort by y
            pts = sorted(pts, key=lambda p: p[0])
            # Skyline: keep points that are not dominated in 2D
            # Dominance: y1 <= y2 and z1 <= z2
            skyline = []
            max_z = -np.inf
            # Iterate from high y to low y?
            # If we sort by y ascending, a point is non-dominated if its z is greater than all previous z's?
            # No, minimization.
            # Point A dominates B if A.y <= B.y and A.z <= B.z.
            # Sort by y ascending.
            # Keep a point if its z is smaller than the min z of all points with larger y?
            # Actually, easier:
            # Sort by y ascending.
            # Iterate, keep track of min_z_so_far? No.
            # If we have points sorted by y, the skyline is the sequence of points where z is decreasing.
            # p1 (y1, z1), p2 (y2, z2)...
            # If z1 < z2, p1 dominates p2? No, p1.y <= p2.y. If z1 <= z2, p1 dominates p2.
            # So we want to keep points with *decreasing* z as y increases?
            # No, if y increases, z must decrease to be non-dominated.
            
            # Let's just filter:
            # A point is non-dominated if no other point has y <= y_i and z <= z_i.
            # For N=20, O(N^2) filter is fine.
            non_dominated = []
            for p in pts:
                dominated = False
                for q in pts:
                    if p is q: continue
                    if q[0] <= p[0] and q[1] <= p[1] and (q[0] < p[0] or q[1] < p[1]):
                        dominated = True
                        break
                if not dominated:
                    non_dominated.append(p)
                    
            if not non_dominated:
                return 0.0
                
            # Sort non_dominated by y ascending
            non_dominated = sorted(non_dominated, key=lambda p: p[0])
            
            # Calculate HV
            # The region is bounded by ref_y, ref_z.
            # We sum rectangles.
            # Breakpoints in y: [p1.y, p2.y, ..., pk.y, ref_y]
            # For interval [y_i, y_{i+1}], the height is (ref_z - z_i) where p_i is the point with min z in the active set?
            # In the skyline (sorted by y asc), z is strictly decreasing.
            # So for interval [y_i, y_{i+1}], the "bottom" of the box is z_i.
            # Wait, if z is decreasing, z_1 > z_2 > ...
            # The point p_i has y_i. The next point p_{i+1} has y_{i+1} > y_i and z_{i+1} < z_i.
            # The area covered by p_i in the slice [y_i, y_{i+1}] is (y_{i+1} - y_i) * (ref_z - z_i).
            # The last point p_k covers [y_k, ref_y] with height (ref_z - z_k).
            
            total = 0.0
            for i in range(len(non_dominated)):
                y_curr = non_dominated[i][0]
                z_curr = non_dominated[i][1]
                if i < len(non_dominated) - 1:
                    y_next = non_dominated[i+1][0]
                else:
                    y_next = ref_y
                
                width = y_next - y_curr
                height = ref_z - z_curr
                if width > 0 and height > 0:
                    total += width * height
                    
            return total

        def _hv_3d_impl(pts, ref):
            if not pts:
                return 0.0
            # Sort by x
            pts = sorted(pts, key=lambda p: p[0])
            
            total_hv = 0.0
            # We need to be careful with points that have the same x.
            # Group by x?
            # Or just iterate.
            
            # Breakpoints
            # We iterate through unique x values.
            
            # Let's use the slicing method described above.
            # We need to maintain the set of "active" (y,z) points.
            # As we increase x, we add points.
            # But the HV formula sums over intervals.
            
            # Let's refine the slicing:
            # Sort pts by x.
            # Let xs = sorted unique x values from pts.
            # Add ref.x to the end.
            
            # Actually, simpler:
            # Iterate through points in order of x.
            # Keep a list `active_yz` of (y,z) for all points with x <= current_x.
            # When we move from x_i to x_{i+1}, the volume added is (x_{i+1} - x_i) * HV_2D(active_yz, ref.y, ref.z).
            
            # To do this correctly:
            # 1. Sort pts by x.
            # 2. Create a list of x-coordinates: [p1.x, p2.x, ..., pk.x, ref.x]
            # 3. We need to know which points are active in each interval.
            #    Interval 0: [x_1, x_2]. Active: points with x <= x_1.
            #    Interval 1: [x_2, x_3]. Active: points with x <= x_2.
            
            # So:
            active_yz = []
            total_hv = 0.0
            prev_x = pts[0][0] # Start with the first point's x?
            
            # Wait, the first interval starts at min_x.
            # Let's group points by x.
            # groups = defaultdict(list)
            # for p in pts: groups[p[0]].append((p[1], p[2]))
            
            # sorted_xs = sorted(groups.keys())
            # sorted_xs.append(ref[0])
            
            # for i in range(len(sorted_xs) - 1):
            #     x_curr = sorted_xs[i]
            #     x_next = sorted_xs[i+1]
            #     # Add points with x == x_curr to active_yz
            #     for yz in groups[x_curr]:
            #         active_yz.append(yz)
            #     # Calculate HV_2D of active_yz
            #     hv_2d_val = _hv_2d(active_yz, ref[1], ref[2])
            #     total_hv += (x_next - x_curr) * hv_2d_val
            
            # This looks correct and efficient enough for N=20.
            
            groups = {}
            for p in pts:
                x_val = p[0]
                if x_val not in groups:
                    groups[x_val] = []
                groups[x_val].append((p[1], p[2]))
                
            sorted_xs = sorted(groups.keys())
            sorted_xs.append(ref[0])
            
            for i in range(len(sorted_xs) - 1):
                x_curr = sorted_xs[i]
                x_next = sorted_xs[i+1]
                
                # Add points at x_curr
                for yz in groups.get(x_curr, []):
                    active_yz.append(yz)
                    
                hv_2d_val = _hv_2d(active_yz, ref[1], ref[2])
                total_hv += (x_next - x_curr) * hv_2d_val
                
            return total_hv

    # End of HV helpers

    # Main Loop
    # Memory: list of (x, y, z) or just F values?
    # We need X to generate offspring.
    # So Memory is list of (x_vector, f_vector).
    
    memory = []
    # Init
    for i in range(pop_size):
        x = lo + (hi - lo) * np.random.rand(dim)
        f = problem(x)
        memory.append((x, f))
    evals = pop_size
    
    # Prune initial memory?
    # If some are dominated, remove them.
    # And if > pop_size, prune by HV.
    
    def prune_memory(mem, ref, max_size):
        # 1. Remove dominated
        # mem is list of (x, f)
        # Check dominance in F
        n = len(mem)
        keep = [True] * n
        for i in range(n):
            if not keep[i]: continue
            for j in range(n):
                if i == j: continue
                if not keep[j]: continue
                # j dominates i?
                if np.all(mem[j][1] <= mem[i][1]) and np.any(mem[j][1] < mem[i][1]):
                    keep[i] = False
                    break
        
        non_dominated = [m for i, m in enumerate(mem) if keep[i]]
        
        # 2. If size > max_size, prune by HV
        if len(non_dominated) > max_size:
            # Calculate HV contribution for each
            # We need to calculate HV of the set, and HV of set minus point i.
            # This is O(N^2 * HV_cost).
            # N is small.
            
            # Convert to list of tuples for HV function
            pts = [m[1] for m in non_dominated]
            
            # Total HV
            total_hv = hv_3d(pts, ref)
            
            # Contributions
            contribs = []
            for i in range(len(pts)):
                # Remove i
                subset = pts[:i] + pts[i+1:]
                hv_subset = hv_3d(subset, ref)
                contribs.append(total_hv - hv_subset)
                
            # Find indices of lowest contributions
            # We want to remove the one with lowest contribution
            # Sort indices by contribution
            order = np.argsort(contribs)
            to_remove = order[0]
            
            # Remove one
            non_dominated.pop(to_remove)
            
            # Recurse or loop?
            # Loop is safer.
            # Actually, we should remove until size <= max_size.
            # But removing one at a time is slow if N is large.
            # For N=20, it's fine.
            
            # Better: Select the best max_size points based on contribution?
            # No, greedy removal is standard.
            
            while len(non_dominated) > max_size:
                pts = [m[1] for m in non_dominated]
                total_hv = hv_3d(pts, ref)
                contribs = []
                for i in range(len(pts)):
                    subset = pts[:i] + pts[i+1:]
                    hv_subset = hv_3d(subset, ref)
                    contribs.append(total_hv - hv_subset)
                
                order = np.argsort(contribs)
                to_remove = order[0]
                non_dominated.pop(to_remove)
                
        return non_dominated

    memory = prune_memory(memory, ref, pop_size)
    
    # Loop
    while evals < budget:
        # Selection
        # Tournament on HV contribution?
        # Calculate contributions for current memory.
        pts = [m[1] for m in memory]
        total_hv = hv_3d(pts, ref)
        contribs = np.zeros(len(pts))
        for i in range(len(pts)):
            subset = pts[:i] + pts[i+1:]
            hv_subset = hv_3d(subset, ref)
            contribs[i] = total_hv - hv_subset
        
        # Normalize contributions?
        # Select parents
        # Tournament size 2
        p1_idx = np.random.randint(len(memory))
        p2_idx = np.random.randint(len(memory))
        if contribs[p1_idx] < contribs[p2_idx]:
            p1_idx, p2_idx = p2_idx, p1_idx
            
        p1 = memory[p1_idx]
        p2 = memory[p2_idx]
        
        # Crossover
        # Polynomial or Gaussian?
        # Let's use Gaussian with adaptive sigma.
        # Sigma starts at 0.1 * range and decays?
        # Or use Polynomial which is self-contained.
        
        # Let's use a simple blend + Gaussian noise.
        # Or SBX.
        
        # Given the short budget, simple Gaussian might be enough if sigma is tuned.
        # Let's use Polynomial Mutation on a copy.
        
        # Actually, let's use a mix.
        # Crossover: Average
        child_x = 0.5 * p1[0] + 0.5 * p2[0]
        
        # Mutation: Gaussian
        # Adaptive sigma: start 0.5, decay to 0.05
        progress = evals / budget
        sigma = 0.5 * (1 - progress) + 0.05 * progress
        
        child_x = child_x + np.random.randn(dim) * sigma * (hi - lo)
        child_x = np.clip(child_x, lo, hi)
        
        child_f = problem(child_x)
        evals += 1
        
        memory.append((child_x, child_f))
        
        # Prune
        memory = prune_memory(memory, ref, pop_size)
        
    return None

# Reference point
ref = np.array([3.0, 5.0, 7.0])

# One issue: The HV function expects points strictly better than ref.
# We should filter points in prune_memory or HV function.
# In HV function, if a point is not better than ref in all dims, it contributes 0.
# My _hv_2d and _hv_3d assume points are within bounds?
# I should filter points in `hv_3d` to only include those where p < ref for all dims.

# Let's refine hv_3d to filter.

# Also, the "Memory" should probably be larger than pop_size to allow more diversity before pruning?
# No, pruning keeps it at pop_size.

# Let's double check the HV logic.
# The scoring uses "exact hypervolume of all recorded vectors".
# My `prune_memory` keeps a subset.
# But the *recorded* vectors are ALL evaluated points.
# So my `memory` is just for search guidance.
# The final score is based on the union of ALL `problem(x)` calls.
# This is a crucial distinction.
# My strategy of pruning `memory` is correct for *search*, but I must ensure I don't *lose* points from the *record*.
# Since the evaluator records every call to `problem`, I don't need to "return" the points.
# I just need to make sure my *search* finds good points.
# The pruning of `memory` only affects which points are used as *parents* for future generations.
# It does not delete points from the evaluator's log.
# So the strategy is sound.

# One optimization: If I evaluate a point that is already in `memory` (or dominated by it), it still counts for HV if it's non-dominated in the *global* set.
# But if it's dominated by a point in `memory`, it won't improve the HV of the *final set* if the dominating point is also in the final set.
# However, if the dominating point is later removed from `memory` (but still in the log), it still counts.
# So, evaluating dominated points is a waste of budget, unless they are non-dominated in the *global* log but dominated in the *local* memory (impossible if memory contains the dominator).
# So, we should avoid evaluating points that are dominated by points in `memory`.
# How? Check before evaluating.
# If `problem(x)` is called, it's recorded.
# So we should check dominance against `memory` before calling `problem`.
# If dominated, skip evaluation?
# But we need to use the budget.
# If we skip, we don't use the budget.
# We should generate a new random point or mutate differently.
# Or just accept that some evaluations might be "wasted" if the mutation is poor.
# Given small budget, checking dominance is cheap.

# Let's add a dominance check.

# Final check on HV implementation.
# The reference point is [3, 5, 7].
# WFG7 objectives are typically in [0, 10] or similar.
# Points must be < ref.

# Code below implements this.

