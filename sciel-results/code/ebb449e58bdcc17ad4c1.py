# EVOLVE-BLOCK-START
import time
import numpy as np

def _hex_vertices(cx, cy, s, a_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.radians(a_deg)
    angs = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angs), cy + s * np.sin(angs)])

def _hex_normals(a_deg):
    """12 edge normals (unit vectors) for a hexagon of rotation a_deg."""
    a = np.radians(a_deg)
    edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
    normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
    return np.vstack([normals, -normals])

def _separation_ok(v1, v2, n1, n2):
    """Check if two hexagons are separated."""
    # v1, v2 are (6,2) arrays of vertices
    # n1, n2 are (12,2) arrays of normals
    # Project onto normals
    # If any normal has disjoint projection intervals, they are separated.
    
    # For efficiency, we can vectorize the projection check slightly, 
    # but with only 11 hexagons (55 pairs) and 24 normals per pair, 
    # a simple loop is fast enough in Python/NumPy hybrid.
    
    # proj1: (12, 6) -> max/min along axis 1 -> (12,)
    proj1 = v1.T @ n1.T 
    min1 = proj1.min(axis=1)
    max1 = proj1.max(axis=1)
    
    proj2 = v2.T @ n2.T
    min2 = proj2.min(axis=1)
    max2 = proj2.max(axis=1)
    
    # Separation condition: max1 < min2 - eps OR max2 < min1 - eps
    eps = 1e-6
    sep1 = max1 < min2 - eps
    sep2 = max2 < min1 - eps
    
    return np.any(sep1) or np.any(sep2)

def _hexagons_separated(v1, v2, a1, a2):
    n1 = _hex_normals(a1)
    n2 = _hex_normals(a2)
    return _separation_ok(v1, v2, n1, n2)

def _point_in_hexagon(p, outer_verts):
    """Check if point p is inside (or on boundary of) the outer hexagon."""
    # outer_verts is (6,2)
    # Check cross product for each edge
    for i in range(6):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % 6]
        edge = p2 - p1
        to_p = p - p1
        cross = edge[0] * to_p[1] - edge[1] * to_p[0]
        if cross < -1e-6:
            return False
    return True

def _all_inner_in_outer(inner_verts_list, outer_verts):
    """Check all vertices of all inner hexagons are inside the outer hexagon."""
    for verts in inner_verts_list:
        for p in verts:
            if not _point_in_hexagon(p, outer_verts):
                return False
    return True

def _compute_minimal_outer_hex(all_verts):
    """
    Compute the minimal enclosing regular hexagon for a set of points.
    Returns (center_x, center_y, angle_deg, side_length).
    """
    if len(all_verts) == 0:
        return (0.0, 0.0, 0.0, 0.0)
    
    # The optimal orientation of the enclosing hexagon is likely aligned with one of the hull edges or vertices.
    # We can sample angles.
    # Also, the center is likely the centroid of the hull or a point that balances the max projections.
    
    # Strategy: Iterate over angles. For each angle, find the optimal center that minimizes the max projection range.
    # For a fixed angle, the minimal enclosing hexagon centered at C has side s such that:
    # For each of 6 normals n_i: max(p @ n_i) - C @ n_i <= apothem and C @ n_i - min(p @ n_i) <= apothem
    # So 2 * apothem >= max_i (max(p @ n_i) - min(p @ n_i))
    # This implies apothem >= 0.5 * max_i (range_i)
    # But the center C affects the projections.
    # Actually, if we fix the angle, the minimal apothem is determined by the width of the point set in the 6 normal directions.
    # The width in direction n is max(p @ n) - min(p @ n).
    # The center must be chosen such that the projections fit in [C @ n - apothem, C @ n + apothem].
    # This is possible iff C @ n is between min(p @ n) + apothem and max(p @ n) - apothem.
    # So we need an intersection of intervals for C @ n_i for all i.
    # This is a linear programming problem or can be solved by checking if the intersection is non-empty.
    # To minimize s (and thus apothem), we find the smallest apothem such that the intersection is non-empty.
    # This is equivalent to finding the minimal width of the set in the hexagonal metric.
    
    # A simpler approach for "minimal enclosing regular hexagon":
    # The optimal hexagon will have at least 3 vertices of the inner set touching its edges.
    # We can use a rotating calipers-like approach or just grid search over angle and center.
    
    # Given the small number of points (<= 66), we can afford a finer search.
    # Let's assume the center is near the centroid.
    centroid = np.mean(all_verts, axis=0)
    
    best_s = np.inf
    best_center = centroid
    best_angle = 0.0
    
    # Search over angle 0 to 60 degrees
    # And search over center in a small region around centroid
    # Or better: for each angle, the optimal center is the one that minimizes the max "excess" projection.
    # Let's just do a coarse-to-fine search over angle and center.
    
    # Actually, there is a known result: the minimal enclosing regular polygon can be found by rotating the polygon.
    # Let's try sampling angles.
    
    # Fine angle search
    angles = np.linspace(0, 60, 361) # 0.1 deg resolution
    
    for a_deg in angles:
        a = np.radians(a_deg)
        edge_angles = a + np.deg2rad(60.0 * np.arange(6)) + np.deg2rad(30.0)
        normals = np.column_stack([np.cos(edge_angles), np.sin(edge_angles)])
        
        # Projections of all verts onto normals
        proj = all_verts @ normals.T  # (M, 6)
        max_proj = proj.max(axis=0)
        min_proj = proj.min(axis=0)
        
        # We need to find center C and apothem A such that:
        # C @ n_i - A <= min_proj_i
        # C @ n_i + A >= max_proj_i
        # => A >= C @ n_i - min_proj_i
        # => A >= max_proj_i - C @ n_i
        # So A >= max_i (max(C @ n_i - min_proj_i, max_proj_i - C @ n_i))
        # Let f(C) = max_i (max(C @ n_i - min_proj_i, max_proj_i - C @ n_i))
        # We want to minimize f(C). This is a convex function of C.
        
        # The optimal C is the center of the "hexagonal minimum enclosing circle".
        # A good heuristic is the centroid, but we can improve it.
        # For a fixed set of constraints, the optimal C often lies at the intersection of the "midpoints" of the widest ranges.
        
        # Let's use a simple local search for C around the centroid.
        # Or better, solve for C that equalizes the max constraints.
        
        # For now, use centroid as a strong baseline.
        C = centroid
        c_proj = C @ normals.T
        # Required apothem
        A = np.max(np.maximum(c_proj - min_proj, max_proj - c_proj))
        s = A / (np.sqrt(3) / 2)
        
        # Try a few local optimizations for C using Nelder-Mead from scipy? 
        # To keep dependencies light and speed high, let's do a manual gradient-free step.
        # Actually, since M is small, we can just check if shifting C towards the "bottleneck" helps.
        
        # Let's refine C with a few iterations of minimizing the max excess.
        # Gradient of max is direction of the max constraint.
        # Subgradient descent.
        C_curr = C.copy()
        for _ in range(20):
            c_proj = C_curr @ normals.T
            excess1 = c_proj - min_proj
            excess2 = max_proj - c_proj
            excess = np.maximum(excess1, excess2)
            max_idx = np.argmax(excess)
            
            # Which constraint is active?
            if excess1[max_idx] >= excess2[max_idx]:
                # C @ n - min > max - C @ n => C @ n is too large
                # Move C opposite to n
                grad = normals[max_idx]
            else:
                # max - C @ n > C @ n - min => C @ n is too small
                # Move C along n
                grad = -normals[max_idx]
            
            # Step size
            step_size = 0.1 * (excess[max_idx] / 10.0 + 0.01)
            C_new = C_curr - grad * step_size
            
            # Check if better
            c_proj_new = C_new @ normals.T
            A_new = np.max(np.maximum(c_proj_new - min_proj, max_proj - c_proj_new))
            if A_new < A:
                A = A_new
                C_curr = C_new
                C = C_curr
            
        s = A / (np.sqrt(3) / 2)
        
        if s < best_s:
            best_s = s
            best_center = C
            best_angle = a_deg
            
    return (best_center[0], best_center[1], best_angle, best_s)

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580.0
    
    # Strategy: Symmetry-reduced parameterization
    # 1 center hex at (0,0)
    # 6 inner ring hexes at radius r1, angles 0, 60, ...
    # 4 outer ring hexes at radius r2, angles 30, 90, ... (offset by 30)
    
    # Variables: r1, r2, angle_offset (rotation of outer ring relative to inner? 
    # Due to symmetry, the inner ring is fixed at 0, 60... 
    # The outer ring can be rotated. Let's fix outer ring at 30, 90... 
    # But we can also rotate the whole thing? No, relative angle matters.
    # Let's parameterize:
    # r1: radius of inner ring
    # r2: radius of outer ring
    # theta: angular offset of outer ring (base 30 + theta)
    
    # Also, the hexagons themselves can be rotated. 
    # For minimal packing, aligning edges is usually best. 
    # Let's assume all hexagons have angle 0. 
    # If we allow rotation, the space grows. 
    # Given the symmetry, angle 0 is a strong candidate. 
    # Let's stick to angle 0 for all inner hexagons to reduce variables.
    
    best_side = np.inf
    best_config = None
    
    # Grid search parameters
    # r1 must be at least sqrt(3) for touching inner ring.
    # r2 must be large enough.
    
    r1_range = np.linspace(1.7, 2.2, 50) # 1.732 to 2.2
    r2_range = np.linspace(3.2, 4.2, 50) # 3.464 to 4.2
    theta_range = np.linspace(-10, 10, 41) # -10 to 10 degrees
    
    # Precompute vertices for angle 0
    # We need to check separation for pairs.
    
    # To speed up, we can precompute normals for angle 0
    n0 = _hex_normals(0.0)
    
    for r1 in r1_range:
        if time.time() > deadline - 30:
            break
            
        # Inner ring positions
        inner_pos = np.array([[0.0, 0.0]] + [
            [r1 * np.cos(np.deg2rad(60*i)), r1 * np.sin(np.deg2rad(60*i))] for i in range(6)
        ])
        
        # Check inner ring separation (center to inner, inner to inner)
        # Center to inner: dist r1. Min dist for touching is sqrt(3). 
        # If r1 < sqrt(3), they overlap. 
        # Let's verify with exact test.
        
        inner_verts = [_hex_vertices(p[0], p[1], 1.0, 0.0) for p in inner_pos]
        
        valid_inner = True
        for i in range(7):
            for j in range(i+1, 7):
                if not _separation_ok(inner_verts[i], inner_verts[j], n0, n0):
                    valid_inner = False
                    break
            if not valid_inner:
                break
        
        if not valid_inner:
            continue
            
        for r2 in r2_range:
            if time.time() > deadline - 30:
                break
                
            for theta in theta_range:
                # Outer ring positions
                outer_pos = [
                    [r2 * np.cos(np.deg2rad(30 + 60*i + theta)), r2 * np.sin(np.deg2rad(30 + 60*i + theta))] for i in range(4)
                ]
                
                all_pos = np.vstack([inner_pos, np.array(outer_pos)])
                all_verts = inner_verts + [_hex_vertices(p[0], p[1], 1.0, 0.0) for p in outer_pos]
                
                # Check all pairs
                valid = True
                # Inner vs Outer
                for i in range(7):
                    for j in range(7, 11):
                        if not _separation_ok(all_verts[i], all_verts[j], n0, n0):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                    
                # Outer vs Outer
                for i in range(7, 11):
                    for j in range(i+1, 11):
                        if not _separation_ok(all_verts[i], all_verts[j], n0, n0):
                            valid = False
                            break
                    if not valid:
                        break
                if not valid:
                    continue
                
                # If valid, compute outer hexagon
                all_verts_arr = np.vstack(all_verts)
                ox, oy, oa, s = _compute_minimal_outer_hex(all_verts_arr)
                
                if s < best_side:
                    best_side = s
                    best_config = (all_pos, all_verts, ox, oy, oa, s)
                    
    if best_config is None:
        # Fallback: return a valid but large packing
        # Simple grid
        positions = []
        angles = []
        positions.append([0.0, 0.0])
        angles.append(0.0)
        r1 = np.sqrt(3)
        for i in range(6):
            ang = np.deg2rad(60 * i)
            positions.append([r1 * np.cos(ang), r1 * np.sin(ang)])
            angles.append(0.0)
        r2 = 2 * np.sqrt(3)
        for i in range(4):
            ang = np.deg2rad(60 * i + 30)
            positions.append([r2 * np.cos(ang), r2 * np.sin(ang)])
            angles.append(0.0)
            
        positions = np.array(positions)
        angles = np.array(angles)
        all_verts = [_hex_vertices(cx, cy, 1.0, ang) for (cx, cy, ang) in zip(positions, angles)]
        all_verts_arr = np.vstack(all_verts)
        ox, oy, oa, s = _compute_minimal_outer_hex(all_verts_arr)
        
        inner_hex_data = np.array([(p[0], p[1], a) for p, a in zip(positions, angles)])
        outer_hex_data = np.array([ox, oy, oa])
        return inner_hex_data, outer_hex_data, float(s)
        
    all_pos, all_verts, ox, oy, oa, s = best_config
    
    # Final verification and adjustment
    outer_verts = _hex_vertices(ox, oy, s, oa)
    
    if not _all_inner_in_outer(all_verts, outer_verts):
        # Expand s slightly
        for _ in range(20):
            s *= 1.01
            outer_verts = _hex_vertices(ox, oy, s, oa)
            if _all_inner_in_outer(all_verts, outer_verts):
                break
                
    # Ensure s >= sqrt(11)
    s = max(s, np.sqrt(11.0))
    
    inner_hex_data = np.array([(p[0], p[1], 0.0) for p in all_pos])
    outer_hex_data = np.array([ox, oy, oa])
    
    return inner_hex_data, outer_hex_data, float(s)

# EVOLVE-BLOCK-END
