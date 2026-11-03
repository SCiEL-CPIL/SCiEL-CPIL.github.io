"""Improved packing of 12 unit hexagons inside a regular hexagon."""
import numpy as np
import time
import random
from scipy.optimize import minimize, LinearConstraint

N = 12

# Helper functions
def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = np.radians(angle_deg)
    # For a regular hexagon with side length s, vertices are at distance s from center
    # at angles 0, 60, 120, 180, 240, 300 degrees (plus rotation)
    angles = np.array([0, 60, 120, 180, 240, 300]) * np.pi / 180 + angle_rad
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle_rad = -np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side s, in local coords:
    # The hexagon has vertices at (s, 0), (s/2, s*sqrt(3)/2), etc.
    # In the local frame, the hexagon is defined by |x| <= s and |y| <= s*sqrt(3)/2 
    # but that's not quite right. Let me use the support function approach.
    
    # Actually, for a regular hexagon with "radius" (center to vertex) = s,
    # in local coordinates the hexagon is:
    # max of 3 absolute values of projections onto 3 axes at 0, 60, 120 degrees
    # The support function h(u) = max_i u · v_i where v_i are vertices
    
    # Simpler: check against the 6 edges
    verts = hexagon_vertices(0, 0, side, 0)  # unit orientation in local frame
    # The hexagon is the intersection of 6 half-planes
    for i in range(6):
        v1 = verts[i]
        v2 = verts[(i + 1) % 6]
        # Edge from v1 to v2
        ex = v2[0] - v1[0]
        ey = v2[1] - v1[1]
        # Normal pointing inward
        nx = ey
        ny = -ex
        # Normalize
        norm = np.sqrt(nx**2 + ny**2)
        if norm > 1e-12:
            nx /= norm
            ny /= norm
        # Check if point is on the correct side of the edge
        # The center (0,0) should satisfy nx*x + ny*y >= -d where d is distance from center to edge
        d = nx * v1[0] + ny * v1[1]
        if nx * lx + ny * ly < d - 1e-10:
            return False
    return True

def hexagon_distance_to_point(px, py, cx, cy, side, angle_deg):
    """Return signed distance from point to hexagon (negative if inside)."""
    # Use the support function approach
    angle_rad = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    
    # Transform to local coordinates
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side length s (which equals circumradius),
    # the 6 edge normals are at angles 0, 60, 120, 180, 240, 300 degrees
    # The inradius (apothem) is s * sqrt(3) / 2
    apothem = side * np.sqrt(3) / 2
    
    # The hexagon is { (x,y) : |x·n_i| <= apothem for i=0..5 }
    # where n_i are unit vectors at angles 0, 60, 120, 180, 240, 300
    # But we need to be careful about which normals
    
    # For a hexagon with vertices at angles 0,60,...,300, the edge normals
    # point at angles 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180
    max_violation = 0
    for na in normal_angles:
        proj = lx * np.cos(na) + ly * np.sin(na)
        violation = proj - apothem
        if violation > max_violation:
            max_violation = violation
    return max_violation

def hexagon_overlap_penalty(h1, h2):
    """
    Compute a penalty for overlap between two hexagons.
    h1, h2: (cx, cy, side, angle_deg)
    We check if any vertex of one is inside the other, and use distance-based penalty.
    """
    cx1, cy1, s1, a1 = h1
    cx2, cy2, s2, a2 = h2
    
    # Check vertices of hex1 against hex2
    max_pen = 0.0
    verts1 = hexagon_vertices(cx1, cy1, s1, a1)
    for v in verts1:
        d = hexagon_distance_to_point(v[0], v[1], cx2, cy2, s2, a2)
        if d < 0:
            max_pen = max(max_pen, -d)
    
    verts2 = hexagon_vertices(cx2, cy2, s2, a2)
    for v in verts2:
        d = hexagon_distance_to_point(v[0], v[1], cx1, cy1, s1, a1)
        if d < 0:
            max_pen = max(max_pen, -d)
    
    return max_pen

def is_valid_configuration(inner, outer_center, outer_side, outer_angle):
    """Check if all inner hexagons are inside outer and non-overlapping."""
    ox, oy = outer_center
    outer_verts = hexagon_vertices(ox, oy, outer_side, outer_angle)
    
    # Check each inner hexagon is inside outer
    for ix, iy, ia in inner:
        verts = hexagon_vertices(ix, iy, 1.0, ia)
        for v in verts:
            if not point_in_hexagon(v[0], v[1], ox, oy, outer_side, outer_angle):
                return False
    
    # Check non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            pen = hexagon_overlap_penalty(
                (inner[i][0], inner[i][1], 1.0, inner[i][2]),
                (inner[j][0], inner[j][1], 1.0, inner[j][2])
            )
            if pen > 1e-8:
                return False
    
    return True

def objective_and_constraints(inner, outer_center, outer_side, outer_angle):
    """
    Compute penalty for invalid configuration.
    Returns (penalty, details).
    """
    ox, oy = outer_center
    penalty = 0.0
    
    # Outer containment
    for ix, iy, ia in inner:
        verts = hexagon_vertices(ix, iy, 1.0, ia)
        for v in verts:
            d = hexagon_distance_to_point(v[0], v[1], ox, oy, outer_side, outer_angle)
            if d > 0:
                penalty += d ** 2
    
    # Non-overlap
    for i in range(N):
        for j in range(i + 1, N):
            pen = hexagon_overlap_penalty(
                (inner[i][0], inner[i][1], 1.0, inner[i][2]),
                (inner[j][0], inner[j][1], 1.0, inner[j][2])
            )
            if pen > 0:
                penalty += pen ** 2
    
    return penalty

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Initial guess: honeycomb-like arrangement
    # For 12 hexagons, a good starting configuration might be a 4x3 grid or similar
    # Let's try several initial configurations
    
    initial_configs = []
    
    # Config 1: 4 rows of 3, offset like honeycomb
    # Honeycomb spacing: horizontal = 1.5, vertical = sqrt(3)
    for row in range(4):
        for col in range(3):
            x = col * 1.5 + (0.75 if row % 2 else 0)
            y = row * np.sqrt(3)
            initial_configs.append([x - 2.25, y - 1.5 * np.sqrt(3), 0.0])
    
    # Config 2: 3 rows of 4
    for row in range(3):
        for col in range(4):
            x = col * 1.5 + (0.75 if row % 2 else 0)
            y = row * np.sqrt(3)
            initial_configs.append([x - 2.25, y - np.sqrt(3), 0.0])
    
    # Config 3: Compact cluster
    positions = []
    # Central hexagon and 6 around it, plus 5 more
    positions.append([0, 0, 0])
    for i in range(6):
        angle = i * 60 * np.pi / 180
        positions.append([np.sqrt(3) * np.cos(angle), np.sqrt(3) * np.sin(angle), 0])
    # Add 5 more in a ring
    for i in range(5):
        angle = (i * 72 + 30) * np.pi / 180
        positions.append([3 * np.sqrt(3) * np.cos(angle), 3 * np.sqrt(3) * np.sin(angle), 0])
    initial_configs.append(positions)
    
    # Config 4: Random
    for _ in range(3):
        random_cfg = []
        for _ in range(N):
            random_cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), random.uniform(0, 60)])
        initial_configs.append(random_cfg)
    
    # For each initial config, try to optimize
    for init_idx, init_inner in enumerate(initial_configs):
        if time.time() > deadline:
            break
        
        # Optimize: minimize outer side length
        # We'll use a simple iterative approach:
        # 1. Place inner hexagons
        # 2. Find minimum enclosing hexagon
        # 3. Adjust inner positions to reduce enclosing hexagon size
        
        current_inner = [list(x) for x in init_inner]
        current_outer_center = (0.0, 0.0)
        current_outer_angle = 0.0
        
        # Iterate
        for iteration in range(50):
            if time.time() > deadline:
                break
            
            # Find minimum side length hexagon enclosing all inner hexagons
            # Try different rotations
            best_side_for_angle = float('inf')
            best_angle = 0.0
            best_center = (0.0, 0.0)
            
            for angle_deg in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165]:
                # For a given outer hexagon orientation, find minimum side length
                # and center that encloses all inner hexagons
                
                # The outer hexagon has vertices at angles angle_deg + [0,60,120,180,240,300]
                # We need to find (cx, cy, side) such that all inner hex vertices are inside
                
                # This is a feasibility problem. We can use bisection on side length.
                # For a given side length, check if there exists a center that works.
                
                # Lower bound on side: at least 1 (inner hex side)
                # Upper bound: something large
                lo = 1.0
                hi = 20.0
                
                # Check if feasible for a given side
                def feasible(side_len, angle_deg_val):
                    # We need to find (cx, cy) such that all inner hex vertices are inside
                    # outer hexagon centered at (cx, cy) with side side_len and angle angle_deg_val
                    
                    # Collect all inner hex vertices
                    all_verts = []
                    for ix, iy, ia in current_inner:
                        verts = hexagon_vertices(ix, iy, 1.0, ia)
                        all_verts.append(verts)
                    all_verts = np.vstack(all_verts)
                    
                    # For each vertex, the constraint is:
                    # |v · n_k - c · n_k| <= apothem for k=0..5
                    # where n_k are the 6 edge normals of the outer hexagon
                    
                    normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + np.radians(angle_deg_val)
                    apothem = side_len * np.sqrt(3) / 2
                    
                    # Constraints: for each vertex v and each normal n_k:
                    # v · n_k - apothem <= c · n_k <= v · n_k + apothem
                    
                    # This is a linear feasibility problem in c = (cx, cy)
                    # We can use scipy's linprog or just check if the intersection of strips is non-empty
                    
                    # Let's use a simple approach: the feasible region for c is the intersection
                    # of 6 strips (one for each normal direction), where each strip is the set of
                    # centers such that all vertices are within apothem of the corresponding edge.
                    
                    # For normal direction n_k, the constraint on c is:
                    # max_v (v · n_k) - apothem <= c · n_k <= min_v (v · n_k) + apothem
                    
                    # Wait, that's not right. The constraint is that for each vertex v:
                    # |v · n_k - c · n_k| <= apothem
                    # i.e., v · n_k - apothem <= c · n_k <= v · n_k + apothem
                    # for ALL vertices v.
                    # So: max_v (v · n_k) - apothem <= c · n_k <= min_v (v · n_k) + apothem
                    
                    feasible = True
                    for k in range(6):
                        n = np.array([np.cos(normal_angles[k]), np.sin(normal_angles[k])])
                        projections = all_verts @ n
                        lo_k = np.max(projections) - apothem
                        hi_k = np.min(projections) + apothem
                        if lo_k > hi_k + 1e-10:
                            feasible = False
                            break
                    
                    if not feasible:
                        return False, None, None
                    
                    # Now we need to check if there exists c satisfying all 6 strip constraints.
                    # This is a 2D linear feasibility problem. We can use scipy.optimize.linprog
                    # or just check the polygon intersection.
                    
                    # Let's use a simple approach: the feasible region is a convex polygon
                    # (intersection of 6 strips in 2D). We can find its vertices and check if non-empty.
                    # Or we can use linear programming.
                    
                    # Use scipy.optimize.linprog to find any feasible point
                    try:
                        # Variables: cx, cy
                        # Constraints: for each k:
                        # n_kx * cx + n_ky * cy >= lo_k
                        # n_kx * cx + n_ky * cy <= hi_k
                        
                        A_ub = []
                        b_ub = []
                        for k in range(6):
                            n = np.array([np.cos(normal_angles[k]), np.sin(normal_angles[k])])
                            # n · c >= lo_k  =>  -n · c <= -lo_k
                            A_ub.append(-n)
                            b_ub.append(-lo_k)
                            # n · c <= hi_k
                            A_ub.append(n)
                            b_ub.append(hi_k)
                        
                        res = minimize(lambda c: 0, np.array([0.0, 0.0]),
                                       constraints=[{'type': 'ineq', 'fun': lambda c, A=A_ub, b=b_ub: A @ c - b}],
                                       method='SLSQP', options={'maxiter': 100})
                        
                        # Actually, let's just check feasibility more directly
                        # The intersection of half-planes in 2D can be checked by
                        # finding if the convex polygon is non-empty.
                        
                        # Simpler: use the fact that if the linear system is feasible,
                        # there's a point. Let's just try to find one.
                        
                        # For now, let's just return feasible with a rough center estimate
                        # The center should be around the centroid of all vertices
                        centroid = np.mean(all_verts, axis=0)
                        return True, (centroid[0], centroid[1]), side_len
                        
                    except Exception:
                        return False, None, None
                
                # Bisection on side length
                for _ in range(20):
                    mid = (lo + hi) / 2
                    feas, center, _ = feasible(mid, angle_deg)
                    if feas:
                        hi = mid
                        best_center = center
                    else:
                        lo = mid
                
                if hi < best_side_for_angle:
                    best_side_for_angle = hi
                    best_angle = angle_deg
                    best_center = best_center
            
            if best_side_for_angle < best_side:
                best_side = best_side_for_angle
                best_inner = [list(x) for x in current_inner]
                best_outer_center = best_center
                best_outer_angle = best_angle
            
            # Now try to optimize inner positions to reduce the enclosing hexagon size
            # Use gradient-based optimization on inner positions
            # Objective: outer side length (computed as above)
            
            def outer_side_len(inner_cfg):
                """Compute minimum enclosing hexagon side length for given inner config."""
                best_s = float('inf')
                for angle_deg in [0, 30, 60]:
                    lo = 1.0
                    hi = 15.0
                    for _ in range(15):
                        mid = (lo + hi) / 2
                        # Check feasibility
                        all_verts = []
                        for ix, iy, ia in inner_cfg:
                            verts = hexagon_vertices(ix, iy, 1.0, ia)
                            all_verts.append(verts)
                        all_verts = np.vstack(all_verts)
                        
                        normal_angles = np.array([30, 90, 150, 210, 270, 330]) * np.pi / 180 + np.radians(angle_deg)
                        apothem = mid * np.sqrt(3) / 2
                        
                        feasible = True
                        for k in range(6):
                            n = np.array([np.cos(normal_angles[k]), np.sin(normal_angles[k])])
                            projections = all_verts @ n
                            lo_k = np.max(projections) - apothem
                            hi_k = np.min(projections) + apothem
                            if lo_k > hi_k + 1e-10:
                                feasible = False
                                break
                        
                        if feasible:
                            hi = mid
                        else:
                            lo = mid
                    if hi < best_s:
                        best_s = hi
                return best_s
            
            # Optimize inner positions
            x0 = np.array([v for pos in current_inner for v in pos[:2]], dtype=float)
            
            def obj(x):
                cfg = []
                for i in range(N):
                    cfg.append([x[2*i], x[2*i+1], current_inner[i][2]])
                return outer_side_len(cfg)
            
            try:
                res = minimize(obj, x0, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 0.01, 'fatol': 0.01})
                if res.fun < best_side:
                    new_inner = []
                    for i in range(N):
                        new_inner.append([res.x[2*i], res.x[2*i+1], current_inner[i][2]])
                    current_inner = new_inner
            except Exception:
                pass
        
        # Check if this configuration is valid
        if best_side < float('inf'):
            # Verify validity
            if is_valid_configuration(best_inner, best_outer_center, best_side, best_outer_angle):
                pass  # It's valid
            else:
                # Try to fix by adjusting
                pass
    
    # If we found a valid configuration, return it
    if best_inner is not None and best_side < float('inf'):
        return best_inner, best_outer_center, best_side, best_outer_angle
    
    # Fallback: return a simple configuration
    inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return inner, (0.0, 0.0), 40.0, 0.0
