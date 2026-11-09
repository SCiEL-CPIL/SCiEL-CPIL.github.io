"""Multi-objective optimizer for RE34 (3 objectives, 5 variables)."""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """LHS-initialized DE with archive-guided mutation and HV-driven acceptance."""

    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 318.0  # safety margin before 330s kill

        dim = self.dim
        n_obj = self.n_obj
        budget = self.budget

        lo = np.array([b[0] for b in bounds], dtype=np.float64)
        hi = np.array([b[1] for b in bounds], dtype=np.float64)
        span = hi - lo

        rng = np.random

        # --- Latin Hypercube Sampling for initial population ---
        pop_size = 40
        lhs_u = np.empty((pop_size, dim))
        for d in range(dim):
            intervals = np.arange(pop_size) / pop_size
            offsets = rng.random(pop_size) / pop_size
            lhs_u[:, d] = intervals + offsets
            perm = rng.permutation(pop_size)
            lhs_u[:, d] = lhs_u[perm, d]
        X_init = lo + span * lhs_u

        # Add structured points
        structured = np.array([
            (lo + hi) / 2,
            lo,
            hi,
        ])
        X = np.vstack([X_init, structured])[:pop_size]

        # Evaluate initial population
        F = np.array([problem(x) for x in X], dtype=np.float64)
        used = pop_size

        # --- Archive management ---
        archive_X = []
        archive_F = []

        def update_archive(x, f):
            nonlocal archive_X, archive_F
            for af in archive_F:
                if np.all(af <= f) and np.any(af < f):
                    return
            new_ax = []
            new_af = []
            for i in range(len(archive_F)):
                af = archive_F[i]
                if not (np.all(f <= af) and np.any(f < af)):
                    new_ax.append(archive_X[i])
                    new_af.append(af)
            new_ax.append(x.copy())
            new_af.append(f.copy())
            archive_X = new_ax
            archive_F = new_af

        for i in range(pop_size):
            update_archive(X[i], F[i])

        # --- Hypervolume (3D exact) ---
        ref = np.array([1705.0, 12.0, 0.3])

        def hypervolume_3d(pts_list):
            """Exact 3D hypervolume for non-dominated set vs reference."""
            if len(pts_list) == 0:
                return 0.0
            pts = []
            for p in pts_list:
                if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]:
                    pts.append(p)
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts, dtype=np.float64)
            # Sort by obj1 ascending
            order = np.argsort(pts[:, 0])
            pts = pts[order]
            n = len(pts)
            # For each point i, its contribution is the box it uniquely owns.
            # With points sorted by obj1, point i owns x-range [pts[i,0], min(pts[i+1,0], ref[0])]
            # In that x-slice, the 2D HV in (obj2, obj3) for points with obj1 in that range
            # is just the max coverage. For simplicity with small n, use inclusion-exclusion.
            
            # Alternative simpler exact method for 3D:
            # HV = sum over all subsets... too expensive.
            # Use the standard O(n^2) for 3D:
            # Sort by x. For each point i, the volume it contributes is:
            #   dx_i * (2D HV of points j>=i in the yz-plane, clipped to [y_i,ref_y]x[z_i,ref_z])
            # This is still complex. For small archives (<100), use a direct box decomposition.
            
            # Direct approach: the HV is the volume of the union of boxes
            # [p_i, ref] for all i. Use the fact that in 3D, we can do:
            # Sort by x. Sweep and maintain the 2D envelope in yz.
            
            # For very small n (which is our case), just do the O(n^3) inclusion-exclusion
            # or even simpler: since n is small, compute by summing boxes and subtracting overlaps.
            
            # Most robust simple method for n < 200: 
            # HV = integral. Use the "layer cake" representation.
            # For 3D with points sorted by x:
            total = 0.0
            for i in range(n):
                xi = pts[i, 0]
                xi_next = pts[i + 1, 0] if i + 1 < n else ref[0]
                if xi_next <= xi:
                    continue
                dx = xi_next - xi
                # In the slab [xi, xi_next] x [0, ref_y] x [0, ref_z],
                # the points that are "active" are those with p[0] <= xi (they extend into this slab)
                # The cross-section at any x in [xi, xi_next] is the 2D region covered by
                # boxes [p_j[1], ref[1]] x [p_j[2], ref[2]] for all j with p_j[0] <= x.
                # Since x >= xi and points are sorted, all j <= i have p_j[0] <= xi <= x.
                # So the cross-section is the union of 2D boxes for j=0..i.
                # The area of this 2D union is what we need.
                
                # Compute 2D HV of points[0..i] in yz-plane
                # Points: (p_j[1], p_j[2]) for j=0..i, reference (ref[1], ref[2])
                # 2D HV: sort by y, sweep
                yz = pts[:i+1, 1:3]
                # Filter to those with y < ref[1] and z < ref[2]
                mask = (yz[:, 0] < ref[1]) & (yz[:, 1] < ref[2])
                yz = yz[mask]
                if len(yz) == 0:
                    continue
                # Sort by y
                yz_order = np.argsort(yz[:, 0])
                yz = yz[yz_order]
                # 2D HV: for each point, dx * max_z_remaining
                # Actually: sort by y ascending. For point i, the x-extent it uniquely covers
                # is [y_i, y_{i+1}], and the z-extent is from min z of active points to ref_z.
                # Simpler: the 2D HV is sum of rectangles.
                # Standard 2D HV: sort by y. Maintain the minimum z seen so far (from right).
                # Actually: sort by y ascending. For each point, the rectangle it contributes
                # is [y_i, y_next] x [min_z_active, ref_z] where min_z_active is the min z
                # of all points with y <= current x... 
                
                # Easiest for small n: O(m^2) 2D HV
                m = len(yz)
                if m == 0:
                    continue
                # Sort by y
                yz = yz[np.argsort(yz[:, 0])]
                area = 0.0
                # For each point, compute the area of the box it uniquely contributes
                # Point i contributes [yz[i,0], yz[i+1,0] or ref[1]] x [yz[i,1], ref[2]]
                # minus overlaps. For 2D, the standard method:
                # Sort by y. Sweep from high y to low y, tracking the max z seen.
                # Actually, let's just do the direct box sum with overlap correction for small m.
                
                # For m <= 50, O(m^2) is fine
                # The 2D HV is the area of union of rectangles [y_j, ref[1]] x [z_j, ref[2]]
                # = sum_j (ref[1]-y_j)*(ref[2]-z_j) - sum_{i<j} overlap(i,j) + ...
                # This gets complex. Let's use the sweep line.
                
                # Sweep line for 2D HV:
                # Sort points by y ascending.
                # The HV = integral from y=0 to ref[1] of (ref[2] - min_z(y)) dy
                # where min_z(y) = min{z_j : y_j <= y} over all j, or ref[2] if none.
                # Actually min_z(y) should be the minimum z among points with y_j <= y.
                # But we want the area NOT covered, which is where z > min_z(y)... 
                # No: the covered region is { (y,z) : exists j with y >= y_j and z >= z_j }
                # = { (y,z) : z >= min{z_j : y_j <= y} } for y where such j exists.
                # So area = integral_0^{ref[1]} max(0, ref[2] - min_z(y)) dy
                # where min_z(y) = min{z_j : y_j <= y} if set nonempty, else ref[2].
                
                # Discretize at the y_j values
                y_vals = yz[:, 0]
                z_vals = yz[:, 1]
                # Add ref[1] as a boundary
                all_y = np.append(y_vals, ref[1])
                all_y = np.sort(np.unique(all_y))
                
                # For each interval [all_y[k], all_y[k+1]], min_z is constant
                # min_z = min{z_j : y_j <= all_y[k+1]} (since y_j <= y for y in [all_y[k], all_y[k+1])
                # Actually min_z(y) = min{z_j : y_j <= y}, which is non-increasing in y.
                # At y = all_y[k+1], min_z = min{z_j : y_j <= all_y[k+1]}
                
                area = 0.0
                min_z_so_far = ref[2]
                # Process y in ascending order
                # For each y_j, update min_z
                for idx in range(m):
                    y_j = yz[idx, 0]
                    z_j = yz[idx, 1]
                    # The interval from previous y to y_j has min_z = min_z_so_far
                    # Previous y is all_y[idx] if we're tracking correctly...
                    # Let me redo this more carefully.
                    pass
                
                # Simpler: just compute directly for small m
                # 2D HV by sorting and sweeping
                yz_sorted = yz[np.argsort(yz[:, 0])]
                area = 0.0
                max_z_covered = 0.0  # This isn't right either.
                
                # Let me use the standard algorithm:
                # Sort by y ascending. 
                # The HV is the sum over each point of the area of the rectangle
                # [y_i, y_{i+1}] x [z_i, ref[2]] where y_{i+1} is the next point's y or ref[1],
                # BUT only the part not covered by subsequent points.
                # Actually the correct formula: sort by y. For point i, the unique contribution
                # is [y_i, min(y_{i+1}, ref[1])] x [z_i, ref[2]] minus what's covered by points
                # with y in that range and z < z_i.
                
                # For m < 30, just do O(m^2) brute force
                area = 0.0
                for i in range(m):
                    yi = yz_sorted[i, 0]
                    zi = yz_sorted[i, 1]
                    y_next = yz_sorted[i+1, 0] if i + 1 < m else ref[1]
                    y_next = min(y_next, ref[1])
                    if y_next <= yi:
                        continue
                    dy = y_next - yi
                    # In [yi, y_next], the z-coverage from point i is [zi, ref[2]]
                    # But points j > i with y_j in [yi, y_next] and z_j < zi may cover part
                    # The uncovered z-range is [zi, ref[2]] minus union of [z_j, ref[2]] for j>i with y_j < y_next
                    # Since all such j have z_j potentially < zi, the uncovered part is [zi, min_z_above]
                    # where min_z_above = min{z_j : j > i, y_j < y_next} or ref[2] if none
                    min_z_above = ref[2]
                    for j in range(i + 1, m):
                        if yz_sorted[j, 0] >= y_next:
                            break
                        if yz_sorted[j, 1] < min_z_above:
                            min_z_above = yz_sorted[j, 1]
                    if min_z_above < zi:
                        dz = ref[2] - zi  # full height, since min_z_above < zi means no coverage from below
                        # Wait, if min_z_above < zi, then points j cover [z_j, ref[2]] which includes [zi, ref[2]]
                        # So the uncovered part is empty? No...
                        # The covered region from j is [z_j, ref[2]]. If z_j < zi, then [zi, ref[2]] is covered.
                        # So uncovered = max(0, ref[2] - max(zi, min_z_above))
                        # If min_z_above < zi, then max(zi, min_z_above) = zi, so uncovered = ref[2] - zi
                        # That's the full height. But that's wrong because j covers [z_j, ref[2]] which includes [zi, ref[2]].
                        # I'm confusing myself. Let me think again.
                        
                        # The union of [z_j, ref[2]] for j>i with y_j < y_next, combined with [zi, ref[2]],
                        # is [min(zi, min_z_above), ref[2]] if any such j exists, else [zi, ref[2]].
                        # So the total covered z-range in [yi, y_next] is [min_z, ref[2]] where
                        # min_z = min(zi, min_z_above) if min_z_above < ref[2], else zi.
                        # The area contributed by point i's unique part is... this is getting circular.
                        
                        # Let me just use a different approach: compute the 2D HV by direct integration.
                        # The 2D HV = area of { (y,z) in [0,ref[1]]x[0,ref[2]] : exists j with y >= y_j and z >= z_j }
                        # = ref[1]*ref[2] - area of { (y,z) : for all j, y < y_j or z < z_j }
                        # The complement is the set of points dominated by all points, which is complex.
                        
                        # For small m, let's just use a grid-free exact method:
                        # The 2D HV can be computed as:
                        # Sort by y. The HV = sum_{i=0}^{m-1} (y_{i+1} - y_i) * (ref[2] - min_z_i)
                        # where y_{-1}=0, y_m=ref[1], and min_z_i = min{z_j : y_j <= y_i} 
                        # No wait, that's for the lower envelope.
                        
                        # Correct formula: sort points by y ascending. 
                        # Let y_0 <= y_1 <= ... <= y_{m-1}.
                        # Define z_min(k) = min{z_j : y_j <= y_k} for k=0..m-1, and z_min(-1) = ref[2].
                        # The HV = sum_{k=0}^{m-1} (y_{k+1} - y_k) * (ref[2] - z_min(k))
                        # where y_m = ref[1].
                        # Wait, that's not right either. z_min(k) should be the minimum z among points with y <= current y.
                        # As y increases, z_min can only decrease.
                        
                        # Let me just implement it properly:
                        pass
                    break  # This whole approach is getting too complex inline
                
                # Let me use a clean 2D HV function
                def hv_2d(pts2d, ref2d):
                    """Exact 2D hypervolume for points in 2D with reference point."""
                    m2 = len(pts2d)
                    if m2 == 0:
                        return 0.0
                    # Filter
                    valid = [(p[0], p[1]) for p in pts2d if p[0] < ref2d[0] and p[1] < ref2d[1]]
                    if len(valid) == 0:
                        return 0.0
                    valid = np.array(valid)
                    # Sort by x (y in our case)
                    valid = valid[np.argsort(valid[:, 0])]
                    n2 = len(valid)
                    # The 2D HV: sort by x. The contribution of each point is
                    # the rectangle it uniquely owns.
                    # Standard: sort by x ascending. For point i, the x-range it owns is
                    # [x_i, x_{i+1}] (or ref[0] if last). The z-range is [z_i, ref[1]]
                    # but clipped by subsequent points.
                    # Actually: the HV = integral of (ref[1] - lower_envelope(x)) dx
                    # where lower_envelope(x) = min{z_j : x_j <= x} for x where such j exists.
                    # 
                    # Discretize at x-values:
                    x_vals = valid[:, 0]
                    z_vals = valid[:, 1]
                    # Add boundaries
                    all_x = np.concatenate([[0], x_vals, [ref2d[0]]])
                    all_x = np.sort(np.unique(all_x))
                    
                    # For each interval [all_x[k], all_x[k+1]], lower_envelope is constant
                    # lower_envelope = min{z_j : x_j <= all_x[k+1]} (using right endpoint)
                    # Actually for x in [all_x[k], all_x[k+1]), lower_env = min{z_j : x_j <= x}
                    # which is min{z_j : x_j <= all_x[k]} (since no x_j in (all_x[k], all_x[k+1]))
                    # Wait, x_j values are in all_x, so for x in [all_x[k], all_x[k+1]),
                    # the set {j : x_j <= x} = {j : x_j <= all_x[k]} (since next x_j is all_x[k+1] > x)
                    # Hmm, not quite. If x_j = all_x[k], then for x >= all_x[k], j is included.
                    
                    # Let me just compute it by sweeping:
                    hv = 0.0
                    min_z = ref2d[1]  # Start with no points covering
                    prev_x = 0.0
                    # Process points in order of x
                    for idx in range(n2):
                        xi = valid[idx, 0]
                        zi = valid[idx, 1]
                        # From prev_x to xi, the lower envelope is min_z
                        if xi > prev_x:
                            if min_z < ref2d[1]:
                                hv += (xi - prev_x) * (ref2d[1] - min_z)
                        # Update min_z with this point
                        if zi < min_z:
                            min_z = zi
                        prev_x = xi
                    # From prev_x to ref2d[0]
                    if ref2d[0] > prev_x:
                        if min_z < ref2d[1]:
                            hv += (ref2d[0] - prev_x) * (ref2d[1] - min_z)
                    return hv
                
                # Now use hv_2d for our 2D subproblem
                # Points in yz-plane: (y_j, z_j) for j=0..i, reference (ref[1], ref[2])
                yz_sub = pts[:i+1, 1:3]
                area = hv_2d(yz_sub, ref[1:3])
                total += dx * area
            
            return max(0.0, total)

        # This is getting very complex and slow. Let me use a simpler but correct approach.
        # For 3D with small n, use the standard O(n^2) algorithm.
        
        def hypervolume_3d_simple(pts_list):
            """3D HV using O(n^2) sweep."""
            if len(pts_list) == 0:
                return 0.0
            pts = []
            for p in pts_list:
                if p[0] < ref[0] and p[1] < ref[1] and p[2] < ref[2]:
                    pts.append(p)
            if len(pts) == 0:
                return 0.0
            pts = np.array(pts, dtype=np.float64)
            n = len(pts)
            # Sort by obj1
            pts = pts[np.argsort(pts[:, 0])]
            
            total_hv = 0.0
            for i in range(n):
                xi = pts[i, 0]
                xi_next = pts[i+1, 0] if i+1 < n else ref[0]
                dx = xi_next - xi
                if dx <= 0:
                    continue
                # 2D HV of points[0..i] in (obj2, obj3) plane
                # Points: (pts[j,1], pts[j,2]) for j=0..i
                # Reference: (ref[1], ref[2])
                sub_pts = pts[:i+1, 1:3]
                # Compute 2D HV
                m = len(sub_pts)
                if m == 0:
                    continue
                # Sort by first coord
                sub_pts = sub_pts[np.argsort(sub_pts[:, 0])]
                # Sweep: 
                area = 0.0
                min_z = ref[2]
                prev_y = 0.0
                for j in range(m):
                    yj = sub_pts[j, 0]
                    zj = sub_pts[j, 1]
                    if yj > prev_y:
                        if min_z < ref[2]:
                            area += (yj - prev_y) * (ref[2] - min_z)
                    if zj < min_z:
                        min_z = zj
                    prev_y = yj
                if ref[1] > prev_y:
                    if min_z < ref[2]:
                        area += (ref[1] - prev_y) * (ref[2] - min_z)
                total_hv += dx * area
            return max(0.0, total_hv)

        current_hv = hypervolume_3d_simple(archive_F)

        F_current = F.copy()
        X_current = X.copy()
        F_factor = 0.8
        CR = 0.85

        while used < budget:
            if time.time() > deadline:
                break

            # Select base from archive if available
            if len(archive_X) > 1:
                base_idx = rng.integers(len(archive_X))
                base_vec = archive_X[base_idx].copy()
            else:
                base_idx = rng.integers(pop_size)
                base_vec = X_current[base_idx].copy()

            # DE mutation from population
            idx_a = rng.integers(pop_size)
            idx_b = rng.integers(pop_size)
            while idx_b == idx_a:
                idx_b = rng.integers(pop_size)
            
            mutant = X_current[idx_a] + F_factor * (X_current[idx_a] - X_current[idx_b])
            # Blend with archive base
            mutant = 0.7 * mutant + 0.3 * base_vec
            mutant = np.clip(mutant, lo, hi)

            # Crossover
            trial = base_vec.copy()
            j_rand = rng.integers(dim)
            for j in range(dim):
                if rng.random() < CR or j == j_rand:
                    trial[j] = mutant[j]
            trial = np.clip(trial, lo, hi)

            f_trial = np.array(problem(trial), dtype=np.float64)
            used += 1

            # Check if non-dominated by archive
            dominated = False
            for af in archive_F:
                if np.all(af <= f_trial) and np.any(af < f_trial):
                    dominated = True
                    break

            if not dominated:
                # Compute new HV
                temp_archive = archive_F + [f_trial]
                # Remove points dominated by f_trial
                filtered = []
                for af in temp_archive:
                    if not (np.all(f_trial <= af) and np.any(f_trial < af)):
                        filtered.append(af)
                new_hv = hypervolume_3d_simple(filtered)
                
                if new_hv > current_hv:
                    update_archive(trial, f_trial)
                    current_hv = hypervolume_3d_simple(archive_F)
                
                # Update population
                replace_idx = rng.integers(pop_size)
                X_current[replace_idx] = trial
                F_current[replace_idx] = f_trial
            else:
                # Occasionally accept for diversity
                if rng.random() < 0.1:
                    sums = np.sum(F_current, axis=1)
                    worst_idx = np.argmax(sums)
                    if np.sum(f_trial) < sums[worst_idx]:
                        X_current[worst_idx] = trial
                        F_current[worst_idx] = f_trial

        return X_current
# EVOLVE-BLOCK-END
