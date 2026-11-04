"""Hexagon packing: 11 unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11
TIME_LIMIT = 140.0  # safety margin before 150s

# EVOLVE-BLOCK-START

def _hex_vertices(angle_deg=0.0, cx=0.0, cy=0.0):
    """Return 6 vertices of a unit regular hexagon rotated by angle_deg, centered at (cx, cy)."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append([cx + math.cos(theta), cy + math.sin(theta)])
    return np.array(verts)

def _compute_side_length(centers, angles, outer_angle_deg=0.0):
    """
    Compute the minimum side length of an outer regular hexagon (at given rotation)
    that contains all inner hexagons.
    
    For a regular hexagon with side S centered at origin, the distance from center
    to each edge (apothem) is S * sqrt(3)/2. The edge normals are at angles
    outer_angle_deg + k*60 degrees.
    
    A point p is inside if for all k: |p . n_k| <= S * sqrt(3)/2
    So S >= max over all vertices v and all normals n_k of |v . n_k| / (sqrt(3)/2)
    """
    a = math.radians(outer_angle_deg)
    normals = np.array([[math.cos(a + k * math.pi / 3.0),
                         math.sin(a + k * math.pi / 3.0)] for k in range(6)])
    
    max_extent = 0.0
    for i in range(N):
        verts = _hex_vertices(angles[i], centers[i][0], centers[i][1])
        projections = verts @ normals.T  # (6, 6)
        abs_proj = np.abs(projections)
        max_extent = max(max_extent, np.max(abs_proj))
    
    S = max_extent / (math.sqrt(3) / 2.0)
    return S

def _min_gap(centers, angles):
    """
    Compute the minimum gap between any pair of inner hexagons.
    Positive means separated, zero means touching, negative means overlapping.
    Uses the fact that for two unit hexagons, the minimum distance between their
    boundaries can be computed via vertex-to-edge distances.
    """
    min_gap = float('inf')
    polys = []
    for i in range(N):
        polys.append(_hex_vertices(angles[i], centers[i][0], centers[i][1]))
    
    for i in range(N):
        for j in range(i + 1, N):
            gap = _poly_min_gap(polys[i], polys[j])
            min_gap = min(min_gap, gap)
    return min_gap

def _point_in_hexagon(pt, center, angle_deg):
    """Check if point pt is inside a unit hexagon centered at center with rotation angle_deg."""
    a = math.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx = pt[0] - center[0]
    dy = pt[1] - center[1]
    # Rotate back
    cos_a = math.cos(-a)
    sin_a = math.sin(-a)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes:
    # |x * cos(k*60) + y * sin(k*60)| <= sqrt(3)/2 for k=0..5
    for k in range(3):
        theta = k * math.pi / 3.0
        proj = lx * math.cos(theta) + ly * math.sin(theta)
        if abs(proj) > math.sqrt(3) / 2.0 + 1e-9:
            return False
    return True

def _poly_min_gap(poly1, poly2):
    """
    Minimum gap between two convex polygons. Positive if separated,
    negative if overlapping (using penetration depth approximation).
    """
    # Check if any vertex of poly1 is inside poly2
    n1 = len(poly1)
    n2 = len(poly2)
    
    # First check for overlap via vertex containment
    for v in poly1:
        if _point_in_polygon(v, poly2):
            return -_penetration_depth(v, poly2)
    for v in poly2:
        if _point_in_polygon(v, poly1):
            return -_penetration_depth(v, poly1)
    
    # No overlap, compute minimum distance
    min_dist = float('inf')
    for i in range(n1):
        for j in range(n2):
            d = _seg_seg_dist(poly1[i], poly1[(i+1)%n1], poly2[j], poly2[(j+1)%n2])
            if d < min_dist:
                min_dist = d
    return min_dist

def _penetration_depth(pt, polygon):
    """Approximate penetration depth of point into convex polygon."""
    # For a hexagon, the minimum distance from pt to the boundary
    min_d = float('inf')
    n = len(polygon)
    for i in range(n):
        d = _point_seg_dist(pt, polygon[i], polygon[(i+1)%n])
        min_d = min(min_d, d)
    return min_d

def _point_seg_dist(pt, a, b):
    """Distance from point to segment a-b."""
    ab = b - a
    t = np.dot(pt - a, ab) / (np.dot(ab, ab) + 1e-15)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(pt - closest)

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-10 and e < 1e-10:
        return np.linalg.norm(r)
    elif a < 1e-10:
        t2 = max(0, min(1, f / (e + 1e-15)))
        return np.linalg.norm(r + d2 * t2)
    elif e < 1e-10:
        t1 = max(0, min(1, -np.dot(d1, r) / (a + 1e-15)))
        return np.linalg.norm(r + d1 * t1)
    
    c = np.dot(d1, r)
    b = np.dot(d1, d2)
    denom = a * e - b * b
    if denom < 1e-10:
        t1 = 0.0
    else:
        t1 = (b * f - c * e) / denom
        t1 = max(0, min(1, t1))
    
    t2 = (b * t1 + f) / (e + 1e-15)
    t2 = max(0, min(1, t2))
    
    if t2 < 0:
        t2 = 0
        t1 = max(0, min(1, -c / (a + 1e-15)))
    elif t2 > 1:
        t2 = 1
        t1 = max(0, min(1, (b - c) / (a + 1e-15)))
    
    return np.linalg.norm(r + d1 * t1 - d2 * t2)

def _point_in_polygon(pt, polygon):
    """Ray casting point-in-polygon test."""
    x, y = pt
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi + 1e-30) + xi):
            inside = not inside
        j = i
    return inside

def _objective(params, outer_angle=0.0):
    """
    Objective: minimize outer side length subject to non-overlap constraint.
    Uses a penalty approach with a hard constraint on minimum gap.
    """
    centers = params[:2*N].reshape(N, 2)
    angles = params[2*N:]
    
    S = _compute_side_length(centers, angles, outer_angle)
    
    # Gap constraint: require gap >= 1e-4
    gap = _min_gap(centers, angles)
    if gap < 1e-4:
        penalty = (1e-4 - gap) ** 2 * 1000.0
    else:
        penalty = 0.0
    
    return S + penalty

def _seed_hex_lattice():
    """
    Generate a valid seed using hexagonal lattice arrangement.
    Center + 6 neighbors + 4 from second ring.
    """
    d = math.sqrt(3)  # distance between centers of adjacent unit hexagons
    
    centers = [[0.0, 0.0]]
    angles = [0.0]
    
    # 6 surrounding hexagons
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d * math.cos(theta), d * math.sin(theta)])
        angles.append(0.0)
    
    # 4 more from second ring - pick positions that are valid
    # Second ring has 12 positions at hex coords with max(|q|,|r|,|s|)=2
    # We pick 4 that are well-separated
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2.append([x, y])
    
    # Pick 4 evenly spaced from the 12
    for idx in [0, 3, 6, 9]:
        centers.append(ring2[idx])
        angles.append(0.0)
    
    return np.array(centers), np.array(angles)

def _seed_spread():
    """
    Alternative seed: spread hexagons more to ensure validity.
    """
    centers = [[0.0, 0.0]]
    angles = [0.0]
    
    d1 = math.sqrt(3) * 1.1  # slightly more than minimum
    for i in range(6):
        theta = i * math.pi / 3.0
        centers.append([d1 * math.cos(theta), d1 * math.sin(theta)])
        angles.append(0.0)
    
    d2 = math.sqrt(3) * 2.2
    for i in range(4):
        theta = (2*i + 1) * math.pi / 12.0
        centers.append([d2 * math.cos(theta), d2 * math.sin(theta)])
        angles.append(0.0)
    
    return np.array(centers), np.array(angles)

def construct():
    start_time = time.time()
    
    # Get initial seed
    centers0, angles0 = _seed_hex_lattice()
    
    # Verify seed validity
    gap0 = _min_gap(centers0, angles0)
    if gap0 < 1e-6:
        centers0, angles0 = _seed_spread()
        gap0 = _min_gap(centers0, angles0)
    
    S0 = _compute_side_length(centers0, angles0)
    best_S = S0
    best_centers = centers0.copy()
    best_angles = angles0.copy()
    
    def time_ok():
        return time.time() - start_time < TIME_LIMIT
    
    # Phase 1: Differential Evolution with multiple outer rotations
    # Try different outer hexagon rotations to find better fit
    outer_rotations = [0.0, 30.0, 15.0, 45.0]
    
    for outer_rot in outer_rotations:
        if not time_ok():
            break
        
        def obj_de(params):
            if not time_ok():
                return 1e10
            return _objective(params, outer_rot)
        
        # Bounds
        bounds = []
        for i in range(N):
            bounds.append((-5.0, 5.0))
            bounds.append((-5.0, 5.0))
        for i in range(N):
            bounds.append((0.0, 60.0))  # hexagon symmetry: 60 deg period
        
        # Initial population from seed
        init = [centers0.flatten().tolist() + angles0.tolist()]
        # Add perturbed versions
        rng = np.random.RandomState(hash(outer_rot) % 10000)
        for _ in range(14):
            c = centers0 + rng.uniform(-1.0, 1.0, size=(N, 2))
            a = (angles0 + rng.uniform(-15, 15, size=N)) % 60.0
            init.append(c.flatten().tolist() + a.tolist())
        
        try:
            res = differential_evolution(
                obj_de, bounds,
                seed=42,
                maxiter=40,
                popsize=15,
                tol=1e-5,
                mutation=(0.5, 1.5),
                recombination=0.8,
                polish=False,
                init=init,
                updating='deferred'
            )
            
            if res.fun < 1e9:
                params = res.x
                c = params[:2*N].reshape(N, 2)
                a = params[2*N:]
                S_val = _compute_side_length(c, a, outer_rot)
                gap_val = _min_gap(c, a)
                if gap_val >= 1e-5 and S_val < best_S:
                    best_S = S_val
                    best_centers = c
                    best_angles = a
                    # Also try with outer rotation 0 for final answer
                    S0_rot = _compute_side_length(c, a, 0.0)
                    if S0_rot < best_S:
                        best_S = S0_rot
        except Exception:
            pass
    
    # Phase 2: Local refinement with Nelder-Mead
    if time_ok():
        def obj_nm(params):
            if not time_ok():
                return 1e10
            centers = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            S = _compute_side_length(centers, angles, 0.0)
            gap = _min_gap(centers, angles)
            if gap < 1e-4:
                return S + (1e-4 - gap) ** 2 * 500.0
            return S
        
        x0 = best_centers.flatten().tolist() + best_angles.tolist()
        try:
            res = minimize(obj_nm, x0, method='Nelder-Mead',
                           options={'xatol': 1e-7, 'fatol': 1e-7,
                                    'maxiter': 3000, 'maxfev': 8000})
            if res.fun < 1e9:
                params = res.x
                c = params[:2*N].reshape(N, 2)
                a = params[2*N:]
                S_val = _compute_side_length(c, a, 0.0)
                gap_val = _min_gap(c, a)
                if gap_val >= 1e-5 and S_val < best_S:
                    best_S = S_val
                    best_centers = c
                    best_angles = a
        except Exception:
            pass
    
    # Phase 3: Fine-tune outer rotation
    if time_ok():
        # Try small outer rotations around 0
        for rot in [0.0, 2.0, -2.0, 4.0, -4.0, 1.0, -1.0]:
            S_rot = _compute_side_length(best_centers, best_angles, rot)
            if S_rot < best_S:
                best_S = S_rot
    
    # Final validation
    final_gap = _min_gap(best_centers, best_angles)
    if final_gap < 1e-5:
        # Fallback to seed
        best_centers = centers0
        best_angles = angles0
        best_S = _compute_side_length(best_centers, best_angles)
    
    # Ensure valid output
    final_S = _compute_side_length(best_centers, best_angles, 0.0)
    if final_S > best_S:
        best_S = final_S
    
    # Add tiny margin for numerical safety
    best_S *= 1.0001
    
    inner = []
    for i in range(N):
        inner.append([float(best_centers[i][0]), float(best_centers[i][1]), float(best_angles[i])])
    
    return inner, (0.0, 0.0), float(best_S), 0.0

# EVOLVE-BLOCK-END
