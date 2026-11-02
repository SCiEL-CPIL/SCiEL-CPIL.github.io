import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

def _hex_vertices(cx, cy, s, angle_deg):
    """
    Returns vertices of a regular hexagon centered at (cx, cy) with side s
    and rotation angle_deg (degrees).
    Vertices are in CCW order.
    """
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        x = cx + s * math.cos(theta)
        y = cy + s * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex_dist_sq(px, py, cx, cy, s, angle_deg):
    """
    Returns squared distance from point (px, py) to the hexagon if outside,
    or 0 if inside.
    Uses the 3-strip representation of a regular hexagon.
    """
    a = math.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    rx = dx * math.cos(-a) - dy * math.sin(-a)
    ry = dx * math.sin(-a) + dy * math.cos(-a)
    
    apothem = s * math.sqrt(3) / 2
    sum_sq_viol = 0.0
    for i in range(3):
        theta_n = math.radians(30 + i * 60)
        proj = rx * math.cos(theta_n) + ry * math.sin(theta_n)
        if proj > apothem:
            sum_sq_viol += (proj - apothem)**2
        if proj < -apothem:
            sum_sq_viol += (-apothem - proj)**2
    return sum_sq_viol

def _pairwise_penalty(p1, p2):
    """
    p1, p2: (cx, cy, angle_deg)
    Returns penalty if they overlap.
    Uses vertex checks and edge intersection checks for robustness.
    """
    c1x, c1y, a1 = p1
    c2x, c2y, a2 = p2
    
    # Quick reject
    dx = c1x - c2x
    dy = c1y - c2y
    if dx*dx + dy*dy > 4.0:
        return 0.0
        
    v1 = _hex_vertices(c1x, c1y, 1.0, a1)
    v2 = _hex_vertices(c2x, c2y, 1.0, a2)
    
    penalty = 0.0
    
    # Vertices of 1 in 2
    for pt in v1:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c2x, c2y, 1.0, a2)
        
    # Vertices of 2 in 1
    for pt in v2:
        penalty += _point_in_hex_dist_sq(pt[0], pt[1], c1x, c1y, 1.0, a1)
        
    # If penalty is 0, check edge intersections
    if penalty == 0.0:
        def segments_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
            
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False

        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i+1)%6]
            for j in range(6):
                p3 = v2[j]
                p4 = v2[(j+1)%6]
                if segments_intersect(p1, p2, p3, p4):
                    mid1 = ((p1[0]+p2[0])/2, (p1[1]+p2[1])/2)
                    mid2 = ((p3[0]+p4[0])/2, (p3[1]+p4[1])/2)
                    penalty += _point_in_hex_dist_sq(mid1[0], mid1[1], c2x, c2y, 1.0, a2)
                    penalty += _point_in_hex_dist_sq(mid2[0], mid2[1], c1x, c1y, 1.0, a1)
                    if penalty == 0.0:
                        penalty += 1.0
    return penalty

def _outer_penalty(inner_params, outer_params):
    """
    Returns penalty if any inner hexagon is not fully inside the outer hexagon.
    """
    ocx, ocy, os, oang = outer_params
    penalty = 0.0
    for ip in inner_params:
        icx, icy, iang = ip
        v_in = _hex_vertices(icx, icy, 1.0, iang)
        for pt in v_in:
            penalty += _point_in_hex_dist_sq(pt[0], pt[1], ocx, ocy, os, oang)
    return penalty

def _objective(x):
    """
    x: [cx1, cy1, a1, ..., cxN, cyN, aN, ocx, ocy, os, oang]
    Objective: Minimize outer side length + penalty for invalidity.
    """
    n_inner = N
    inner_params = []
    for i in range(n_inner):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner_params.append((cx, cy, a))
        
    ocx = x[n_inner*3]
    ocy = x[n_inner*3+1]
    os = x[n_inner*3+2]
    oang = x[n_inner*3+3]
    outer_params = (ocx, ocy, os, oang)
    
    pen_pair = 0.0
    for i in range(n_inner):
        for j in range(i+1, n_inner):
            pen_pair += _pairwise_penalty(inner_params[i], inner_params[j])
            
    pen_outer = _outer_penalty(inner_params, outer_params)
    
    return os + 100.0 * (pen_pair + pen_outer)

def _compute_min_outer_side(inner_params, outer_angle_deg):
    """
    Given fixed inner positions and outer angle, compute the minimum outer side length
    required to contain them, assuming the outer hexagon is centered at the centroid of inner hexagons.
    This is an approximation but often tight.
    """
    # Calculate centroid of inner hexagons
    cx = sum(p[0] for p in inner_params) / len(inner_params)
    cy = sum(p[1] for p in inner_params) / len(inner_params)
    
    # For a regular hexagon, the distance from center to vertex is R = s.
    # The distance from center to side (apothem) is a = s * sqrt(3)/2.
    # If we align the outer hexagon such that its vertices point towards the farthest inner points,
    # we can estimate the required side length.
    
    # A more robust way: The outer hexagon is the intersection of 3 strips.
    # For a fixed orientation, the side length s is determined by the maximum projection
    # of any point onto the normals of the strips.
    
    a_outer = math.radians(outer_angle_deg)
    max_proj = 0.0
    
    # Normals of the outer hexagon sides are at angles 30, 90, 150 (relative to outer angle)
    # Wait, if outer angle is alpha, vertices are at alpha + k*60.
    # Sides are perpendicular to directions alpha + 30 + k*60.
    # So normals are at alpha + 30 + k*60.
    
    for i in range(3):
        theta_n = a_outer + math.radians(30 + i * 60)
        nx = math.cos(theta_n)
        ny = math.sin(theta_n)
        
        for ip in inner_params:
            icx, icy, iang = ip
            v_in = _hex_vertices(icx, icy, 1.0, iang)
            for pt in v_in:
                proj = (pt[0] - cx) * nx + (pt[1] - cy) * ny
                if proj > max_proj:
                    max_proj = proj
                if -proj > max_proj:
                    max_proj = -proj
                    
    # max_proj is the distance from center to the farthest point in the direction of the normal.
    # This distance must be <= apothem of outer hexagon.
    # apothem = s * sqrt(3) / 2
    # s = max_proj * 2 / sqrt(3)
    s = max_proj * 2 / math.sqrt(3)
    return s

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin for 150s limit
    
    best_result = None
    best_score = float('inf')
    
    # Strategy: Generate structured initial guesses based on hexagonal lattice symmetries.
    # Then refine each using local optimization.
    
    # 1. Standard Hexagonal Lattice (Flat-top vs Pointy-top)
    # Hex grid vectors for unit hexagons (side 1):
    # Distance between centers of touching hexes is sqrt(3) if flat sides touch?
    # No, for regular hexagons of side 1, the distance between centers of touching hexagons is 2 * apothem = sqrt(3).
    # Wait, apothem is sqrt(3)/2. So distance is sqrt(3).
    
    # Lattice vectors:
    # v1 = (sqrt(3), 0)
    # v2 = (sqrt(3)/2, 1.5)
    
    sqrt3 = math.sqrt(3)
    v1 = np.array([sqrt3, 0])
    v2 = np.array([sqrt3/2, 1.5])
    
    # Generate a set of 11 positions in a compact cluster on this lattice.
    # We need to choose 11 cells from the infinite grid such that they form a connected, compact shape.
    # The "hexagonal number" sequence: 1, 7, 19...
    # 11 is not a perfect hexagonal number.
    # We can take the center, ring 1 (6), and 4 from ring 2.
    
    # Coordinates of ring 0: (0,0)
    # Coordinates of ring 1:
    # (sqrt3, 0), (sqrt3/2, 1.5), (-sqrt3/2, 1.5), (-sqrt3, 0), (-sqrt3/2, -1.5), (sqrt3/2, -1.5)
    
    # Let's define a function to generate lattice points.
    def get_lattice_points(i, j):
        # i, j are integer coordinates in the oblique basis
        x = i * v1[0] + j * v2[0]
        y = i * v1[1] + j * v2[1]
        return (x, y)
    
    # We want to pick 11 points.
    # Let's try a few different "shapes" of 11 points on the lattice.
    
    # Shape 1: Center + Ring 1 + 4 specific points from Ring 2
    # Ring 2 points have max(|i|, |j|, |i+j|) = 2? No, that's for axial coordinates.
    # Let's just manually define a few tight clusters.
    
    # Cluster 1: (0,0) and 6 neighbors and 4 more.
    # Neighbors of (0,0) in lattice coords (i,j):
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    
    # Let's define the set of lattice indices.
    # Index 0: (0,0)
    # Indices 1-6: Ring 1
    # Indices 7-10: 4 from Ring 2
    
    # We will try different rotations of the outer hexagon and different subsets of Ring 2.
    
    initial_configs = []
    
    # Base set of 7 (Center + Ring 1)
    base_7 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)
    ]
    
    # Ring 2 candidates (neighbors of Ring 1 that are not in Ring 1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # There are 12 points in Ring 2.
    ring_2_candidates = [
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), 
        (-1, -1), (0, -2), (1, -2), (2, -1)
    ]
    # Wait, ring 2 has 12 points. My list has 10.
    # Let's generate them properly.
    # Axial coordinates (q, r).
    # Ring 2: max(abs(q), abs(r), abs(q+r)) == 2.
    # (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Missing: (-1, 3)? No.
    # Let's just use a brute force generation of points within a radius.
    
    def generate_lattice_points(radius):
        points = []
        for q in range(-radius, radius + 1):
            for r in range(-radius, radius + 1):
                if max(abs(q), abs(r), abs(q+r)) <= radius:
                    points.append((q, r))
        return points
    
    # We need 11 points.
    # Let's pick the 11 closest points to (0,0) on the lattice.
    # Distance in lattice space? Or Euclidean distance of the resulting Cartesian coords?
    # Euclidean distance is better for compactness.
    
    all_points = generate_lattice_points(3)
    # Calculate Cartesian coords
    cart_points = []
    for q, r in all_points:
        x, y = get_lattice_points(q, r)
        cart_points.append((x, y, q, r))
        
    # Sort by distance from origin
    cart_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    # Take the first 11
    selected_points = cart_points[:11]
    
    # This gives one specific cluster.
    # To explore variations, we can rotate the entire cluster or shift the selection.
    # But since the lattice is symmetric, rotating the cluster by 60 degrees just permutes the points.
    # However, the *outer* hexagon orientation matters.
    
    # Let's create a few initial configurations:
    # 1. The 11 closest points, outer hexagon aligned with lattice (0 deg)
    # 2. The 11 closest points, outer hexagon rotated by 30 deg
    # 3. A slightly different set of 11 points (swap one boundary point)
    
    configs = []
    
    # Config 1
    inner1 = []
    for x, y, q, r in selected_points:
        inner1.append((x, y, 0.0)) # Inner hexagons aligned with lattice (0 deg)
    configs.append(inner1)
    
    # Config 2: Same positions, but try rotating inner hexagons?
    # Usually, aligning inner hexagons with the lattice is optimal for hexagonal packing.
    # But we can try 30 deg rotation for inner hexagons too.
    inner2 = []
    for x, y, q, r in selected_points:
        inner2.append((x, y, 30.0))
    configs.append(inner2)
    
    # Config 3: Try a different cluster shape.
    # Exclude the furthest point and include the next one.
    selected_points_alt = cart_points[:10] + [cart_points[11]]
    inner3 = []
    for x, y, q, r in selected_points_alt:
        inner3.append((x, y, 0.0))
    configs.append(inner3)
    
    # Now, for each config, we will:
    # 1. Compute a good initial outer side length and center.
    # 2. Run local optimization to refine positions and outer hexagon.
    
    for idx, inner_init in enumerate(configs):
        if time.time() > deadline:
            break
            
        # Estimate outer hexagon
        # Center at centroid
        cx = sum(p[0] for p in inner_init) / len(inner_init)
        cy = sum(p[1] for p in inner_init) / len(inner_init)
        
        # Try a few outer angles
        for outer_angle in [0.0, 30.0, 60.0]:
            # Compute min side for this angle
            s_init = _compute_min_outer_side(inner_init, outer_angle)
            # Add a small buffer to ensure validity initially
            s_init += 0.1
            
            x0 = []
            for p in inner_init:
                x0.append(p[0])
                x0.append(p[1])
                x0.append(p[2])
            x0.extend([cx, cy, s_init, outer_angle])
            
            # Bounds
            bounds = []
            for i in range(N):
                bounds.append((-10, 10))
                bounds.append((-10, 10))
                bounds.append((0, 360))
            bounds.append((-10, 10))
            bounds.append((-10, 10))
            bounds.append((2.0, 10.0)) # Min side length
            bounds.append((0, 360))
            
            # Run local optimization
            try:
                # Use Nelder-Mead for robustness with non-differentiable penalties
                res = minimize(_objective, x0, method='Nelder-Mead', 
                               options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
                
                if res.fun < best_score:
                    best_score = res.fun
                    best_result = res.x
                    
            except Exception as e:
                pass
                
        # If we have time, try a global search from this start point?
        # No, DE is slow. Let's stick to local refinement of good starts.
        
    # If no result found, or if we have time, run a quick DE?
    # The current program used DE. If our structured starts are good, local refinement should suffice.
    # But if all local refinements fail, we need a fallback.
    
    if best_result is None:
        # Fallback: Simple grid search or random restarts?
        # Let's try a very small DE run as a last resort.
        x0 = []
        for p in configs[0]:
            x0.append(p[0])
            x0.append(p[1])
            x0.append(p[2])
        cx = sum(p[0] for p in configs[0]) / len(configs[0])
        cy = sum(p[1] for p in configs[0]) / len(configs[0])
        s_init = _compute_min_outer_side(configs[0], 0.0) + 0.5
        x0.extend([cx, cy, s_init, 0.0])
        
        bounds = []
        for i in range(N):
            bounds.append((-10, 10))
            bounds.append((-10, 10))
            bounds.append((0, 360))
        bounds.append((-10, 10))
        bounds.append((-10, 10))
        bounds.append((2.0, 10.0))
        bounds.append((0, 360))
        
        try:
            res = differential_evolution(_objective, bounds, maxiter=10, popsize=5, seed=42, polish=False)
            if res.fun < best_score:
                best_score = res.fun
                best_result = res.x
        except Exception:
            pass
    
    if best_result is None:
        # Absolute fallback
        return _format_result(x0)
        
    return _format_result(best_result)

def _format_result(x):
    inner = []
    for i in range(N):
        cx = x[i*3]
        cy = x[i*3+1]
        a = x[i*3+2]
        inner.append([cx, cy, a])
        
    ocx = x[N*3]
    ocy = x[N*3+1]
    os = x[N*3+2]
    oang = x[N*3+3]
    
    return inner, (ocx, ocy), os, oang

if __name__ == "__main__":
    r = construct()
    print(r)
