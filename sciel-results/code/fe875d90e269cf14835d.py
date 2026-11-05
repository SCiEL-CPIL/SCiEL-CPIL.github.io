import numpy as np
import time
from scipy.optimize import minimize

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
    n = len(poly)
    sign = 0
    for i in range(n):
        a = poly[i]
        b = poly[(i+1) % n]
        cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cross > 1e-12:
            if sign < 0:
                return False
            sign = 1
        elif cross < -1e-12:
            if sign > 0:
                return False
            sign = -1
    return True

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap (interiors intersect or share boundary),
    positive if separated.
    Uses a simple approach: check if any vertex of one is inside the other,
    and check edge-edge distances.
    For unit hexagons, a simpler approach: use the minimum distance between
    all vertex-edge pairs.
    """
    # Check if any vertex of p1 is inside p2
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    # Check if any vertex of p2 is inside p1
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    # Check edge-edge distances
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            # Compute min distance between two line segments
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        # Parallel
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _outer_hex_vertices(cx, cy, side, angle_deg):
    """Vertices of outer regular hexagon."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([side * np.cos(angles), side * np.sin(angles)])
    return vs + np.array([cx, cy])

def _compute_side_analytic(inner_params, outer_angle_deg):
    """
    Compute the minimum outer hexagon side length such that all inner hexagons
    are contained within the outer hexagon.
    
    The outer hexagon is centered at (0,0) with rotation outer_angle_deg.
    The distance from the center to any edge of a regular hexagon with side s is 
    apothem = s * sqrt(3) / 2.
    
    For each of the 6 edge normals of the outer hexagon, we calculate the maximum 
    projection of any vertex of any inner hexagon onto that normal.
    The required apothem is the maximum of these projections.
    Then s = apothem / (sqrt(3) / 2).
    """
    n = len(inner_params)
    # Normals of the outer hexagon edges
    # Vertices are at angles 0, 60, ...
    # Edge normals are perpendicular to edges.
    # For a hexagon with vertices at angles theta_k, the edge between k and k+1 
    # has a normal pointing in the direction of the bisector (theta_k + 30).
    # Wait, standard regular hexagon vertices:
    # v_k = R * (cos(theta_k), sin(theta_k))
    # The edge normal for the edge opposite to the vertex? 
    # No, the apothem is the distance to the edge.
    # The edges are perpendicular to the radii of the inscribed circle.
    # If vertices are at 0, 60, 120..., the edges are "facing" directions 30, 90, 150...
    # Let's verify:
    # Vertex 0: (s, 0). Vertex 1: (s/2, s*sqrt(3)/2).
    # Edge 0-1 midpoint: (3s/4, s*sqrt(3)/4).
    # Vector from center to midpoint: (3, sqrt(3)) -> angle 30 deg.
    # So the normals to the edges are at 30, 90, 150, 210, 270, 330 degrees.
    
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    max_projection = 0.0
    
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        
        # Project all vertices onto all normals
        # hex_v is (6, 2), normals is (6, 2)
        # We want max over all vertices and all normals of dot(hex_v, normals)
        # This is equivalent to finding the support function.
        # For a convex polygon, the max projection onto a direction d is 
        # max_k dot(v_k, d).
        
        # We can vectorize this:
        # projections = hex_v @ normals.T  (6x6)
        # max_proj = max(projections)
        proj = hex_v @ normals.T
        current_max = np.max(proj)
        if current_max > max_projection:
            max_projection = current_max
            
    # The outer hexagon must contain all these projections.
    # The distance from center to edge (apothem) must be at least max_projection.
    # apothem = side * sqrt(3) / 2
    # side = apothem / (sqrt(3) / 2) = 2 * apothem / sqrt(3)
    
    # Note: If max_projection is negative (all hexagons are in negative quadrant relative to normal),
    # the hexagon still needs to have a positive side length. 
    # However, since the outer hexagon is centered at (0,0) and contains the origin? 
    # Actually, the inner hexagons might not be centered at 0. 
    # The condition is that the polygon is inside.
    # The support function of the outer hexagon in direction n is apothem.
    # So we need apothem >= max_projection.
    
    side = 2.0 * max_projection / np.sqrt(3)
    
    # Side length must be at least 1.0 (though for packing 11 hexagons it will be much larger)
    if side < 1.0:
        side = 1.0
        
    return side

def _overlap_penalty(inner_params, penalty_weight=100.0):
    """
    Compute penalty for overlapping inner hexagons.
    Returns 0 if all hexagons are disjoint, positive otherwise.
    """
    n = len(inner_params)
    penalty = 0.0
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < 0.01:
                # Penalty for being too close
                penalty += penalty_weight * (0.01 - sep) ** 2
    
    return penalty

def _objective(params, outer_angle_deg, n):
    """
    Objective function: outer side length + overlap penalty.
    params: flat array of length 3*n: [x1, y1, a1, x2, y2, a2, ...]
    """
    inner_params = params.reshape(n, 3)
    side = _compute_side_analytic(inner_params, outer_angle_deg)
    overlap_pen = _overlap_penalty(inner_params)
    return side + overlap_pen

def _is_valid(inner_params, side, outer_angle_deg, min_sep=0.0):
    """Check if the configuration is valid."""
    n = len(inner_params)
    outer_v = _outer_hex_vertices(0.0, 0.0, side, outer_angle_deg)
    
    # Check containment
    for i in range(n):
        x = inner_params[i, 0]
        y = inner_params[i, 1]
        a = inner_params[i, 2]
        hex_v = _hex_vertices(x, y, a)
        for v in hex_v:
            if not _point_in_polygon(v, outer_v):
                return False
    
    # Check non-overlap
    for i in range(n):
        x1 = inner_params[i, 0]
        y1 = inner_params[i, 1]
        a1 = inner_params[i, 2]
        v1 = _hex_vertices(x1, y1, a1)
        
        for j in range(i+1, n):
            x2 = inner_params[j, 0]
            y2 = inner_params[j, 1]
            a2 = inner_params[j, 2]
            v2 = _hex_vertices(x2, y2, a2)
            
            sep = _poly_separation(v1, v2)
            if sep < min_sep:
                return False
    
    return True

def _seed_solution():
    """Return the concentric ring seed solution."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def construct():
    start_time = time.time()
    DEADLINE = 145.0  # Safety margin for 150s limit
    
    n = 11
    outer_angle = 30.0
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner
    best_side = seed_side
    best_valid = True
    
    # Initial guess: concentric ring
    init_params = np.zeros(3 * n)
    for i, (x, y, a) in enumerate(seed_inner):
        init_params[i*3] = x
        init_params[i*3+1] = y
        init_params[i*3+2] = a
    
    # List of initial configurations to try
    initial_configs = [init_params]
    
    # Add some rotated/perturbed versions
    rng = np.random.default_rng(42)
    for k in range(5):
        perturbed = init_params.copy()
        # Small random perturbations to break symmetry
        noise = rng.normal(0, 0.5, (n, 3))
        noise[:, 2] = rng.normal(0, 5.0, n) # Angle noise
        perturbed = perturbed.reshape(n, 3) + noise
        perturbed[:, 2] = (perturbed[:, 2] + 180) % 360 - 180
        initial_configs.append(perturbed.flatten())
        
        # Check time
        if time.time() - start_time > DEADLINE - 30:
            break

    for config in initial_configs:
        elapsed = time.time() - start_time
        if elapsed > DEADLINE - 10:
            break
            
        try:
            # Use Nelder-Mead as it handles non-differentiable objectives better
            result = minimize(
                _objective,
                config,
                args=(outer_angle, n),
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            
            opt_params = result.x.reshape(n, 3)
            opt_side = _compute_side_analytic(opt_params, outer_angle)
            
            # Check validity with strict separation
            if _is_valid(opt_params, opt_side, outer_angle, min_sep=1e-7):
                if opt_side < best_side:
                    best_side = opt_side
                    best_inner = [[float(opt_params[i,0]), float(opt_params[i,1]), float(opt_params[i,2])] for i in range(n)]
                    best_valid = True
                    
        except Exception as e:
            pass

    # Final polish on the best found
    if best_valid:
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 5:
            try:
                polish_params = np.zeros(3 * n)
                for i, (x, y, a) in enumerate(best_inner):
                    polish_params[i*3] = x
                    polish_params[i*3+1] = y
                    polish_params[i*3+2] = a
                
                result_polish = minimize(
                    _objective,
                    polish_params,
                    args=(outer_angle, n),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
                )
                
                polish_params = result_polish.x.reshape(n, 3)
                polish_side = _compute_side_analytic(polish_params, outer_angle)
                
                if _is_valid(polish_params, polish_side, outer_angle, min_sep=1e-8):
                    if polish_side < best_side:
                        best_side = polish_side
                        best_inner = [[float(polish_params[i,0]), float(polish_params[i,1]), float(polish_params[i,2])] for i in range(n)]
            except Exception as e:
                pass

    return best_inner, (0.0, 0.0), best_side, 30.0
