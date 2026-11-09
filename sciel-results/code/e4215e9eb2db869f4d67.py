"""Pack 11 unit hexagons inside a regular hexagon, minimizing outer side length."""
import math
import time
import random
import numpy as np

N = 11
DEADLINE = 145.0  # seconds, safety margin before 150s kill

# Precompute unit hexagon vertices (flat-top orientation, angle 0)
# Regular hexagon with side length 1: vertices at angles 0, 60, ..., 300
_UNIT_HEX_VERTS = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    R = np.array([[c, -s], [s, c]])
    verts = side * _UNIT_HEX_VERTS @ R.T
    return verts + np.array([cx, cy])

def _signed_dist_to_hexagon(px, py, cx, cy, side, angle_deg):
    """
    Approximate signed distance from point (px,py) to hexagon.
    Positive if outside. For regular hexagon, we can use the support function approach.
    Actually, let's use a simpler method: project onto the 3 axes of symmetry.
    
    For a regular hexagon with side s centered at origin, the distance from center to each face
    (apothem) is s * sqrt(3)/2. The hexagon is the intersection of 6 half-planes.
    
    A point is inside if for all 6 face normals n_i, n_i . p <= apothem.
    The signed distance is max(0, max_i(n_i . p - apothem)) for outside points.
    For inside points, it's negative of min_i(apothem - n_i . p).
    
    Let's use: d = max_i (n_i . (p - center) - apothem)
    If d < 0, point is inside, distance = -(-d) = d (negative).
    If d >= 0, point is outside, distance = d.
    
    Actually the signed distance to a convex polygon isn't simply the max of face distances
    when you're in the corner region. But for our purposes (checking overlap), we can use
    a more robust method.
    """
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    theta = math.radians(angle_deg)
    c, s = math.cos(theta), math.sin(theta)
    # Rotate point by -theta to align hexagon with axes
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    
    # For unit hexagon (side=1), the 6 face normals point at angles 30, 90, 150, 210, 270, 330
    # Apothem = sqrt(3)/2
    apothem = side * math.sqrt(3) / 2
    
    # The 6 outward normals
    normals = []
    for i in range(6):
        ang = math.radians(30 + 60 * i)
        normals.append((math.cos(ang), math.sin(ang)))
    
    max_proj = -1e18
    for nx, ny in normals:
        proj = nx * lx + ny * ly
        if proj > max_proj:
            max_proj = proj
    
    # Signed distance: if max_proj > apothem, outside by (max_proj - apothem)
    # If max_proj <= apothem, inside. The distance to boundary is apothem - max_proj (but this
    # is only exact if the closest feature is a face, not a vertex)
    # For simplicity and since we mainly care about overlap detection, use:
    if max_proj > apothem:
        return max_proj - apothem
    else:
        return -(apothem - max_proj)  # negative means inside

def _hexagons_overlap(x1, y1, a1, x2, y2, a2, side1=1.0, side2=1.0, tol=1e-9):
    """
    Check if two regular hexagons overlap.
    Uses separating axis theorem with all 12 edge normals (6 from each hex).
    """
    verts1 = _hex_vertices(x1, y1, side1, a1)
    verts2 = _hex_vertices(x2, y2, side2, a2)
    
    # Get all edge normals from both polygons
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            edge = p2 - p1
            # Normal perpendicular to edge
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal /= norm
            normals.append(normal)
        return normals
    
    normals1 = get_normals(verts1)
    normals2 = get_normals(verts2)
    
    all_normals = normals1 + normals2
    
    for normal in all_normals:
        # Project both polygons onto this normal
        proj1 = verts1 @ normal
        proj2 = verts2 @ normal
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Check for separation
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    
    return True

def _inside_outer_hexagon(x, y, angle_deg, outer_cx, outer_cy, outer_side, outer_angle):
    """Check if a unit hexagon centered at (x,y) with given angle is inside the outer hexagon."""
    verts = _hex_vertices(x, y, 1.0, angle_deg)
    outer_verts = _hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    # Check if all vertices of inner hex are inside or on the outer hex
    # Use point-in-convex-polygon test
    def point_in_convex_poly(px, py, poly):
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            # Cross product to check if point is to the left of edge
            cross = (p2[0] - p1[0]) * (py - p1[1]) - (p2[1] - p1[1]) * (px - p1[0])
            if cross < -1e-9:  # Assuming counter-clockwise winding
                return False
        return True
    
    for v in verts:
        if not point_in_convex_poly(v[0], v[1], outer_verts):
            return False
    return True

def _compute_outer_side(inner_positions):
    """
    Given inner hexagon positions [(x, y, angle), ...], find the minimum outer hexagon side length
    and center that contains all of them.
    
    Strategy: The outer hexagon is also regular. We need to find center (cx, cy), side s, and angle
    such that all inner hexagons fit inside.
    
    For a given outer angle, the minimal enclosing regular hexagon can be found by:
    - The outer hexagon is defined by 6 half-planes.
    - For each of the 6 face normals, the constraint is: n_i . p <= s * sqrt(3)/2 + n_i . center
    - We want to minimize s subject to all inner hex vertices satisfying these constraints.
    
    Simplified approach: 
    1. Fix outer angle (try a few).
    2. For a given outer angle, the problem reduces to finding center and side.
    3. The minimal side for a given center and angle is max over all inner vertices of
       (n_i . v - n_i . center) / (sqrt(3)/2) for the relevant face.
    
    Actually, let's use a different approach:
    - The outer hexagon must contain all inner hexagons.
    - For a fixed outer angle, the minimal enclosing regular hexagon of a set of points
      can be found by considering that the hexagon is the intersection of 6 strips.
    
    Let me use a simpler but effective method:
    - For a given outer angle theta, the 6 face normals are at angles theta+30, theta+90, etc.
    - The constraint for each face i: n_i . p <= d_i for all points p in all inner hexagons.
    - We need d_i >= max over all points p of n_i . p.
    - The side length s satisfies d_i - n_i . center = s * apothem_unit, where apothem_unit = sqrt(3)/2.
    - Wait, let me reconsider.
    
    For a regular hexagon centered at C with side s and rotation phi:
    The 6 vertices are at C + s * R(phi) * [cos(60k), sin(60k)]
    The 6 face midpoints are at C + s * sqrt(3)/2 * R(phi) * [cos(60k+30), sin(60k+30)]
    The outward normal of face i is R(phi) * [cos(60i+30), sin(60i+30)]
    
    A point P is inside if for all i: n_i . (P - C) <= s * sqrt(3)/2
    
    So: n_i . P <= n_i . C + s * sqrt(3)/2
    
    For all inner hex vertices P, we need: n_i . P <= n_i . C + s * sqrt(3)/2
    
    This means: s >= (n_i . P - n_i . C) / (sqrt(3)/2) for all i and all P.
    
    So s >= max_i max_P (n_i . P - n_i . C) / (sqrt(3)/2)
    
    To minimize s, we can choose C. For a fixed set of constraints, the optimal C is the one
    that minimizes the maximum of (n_i . P - n_i . C) over all i, P.
    
    This is a linear programming problem. But let's use a heuristic:
    - The center C should be near the centroid of all inner hex vertices.
    - Then compute the required s.
    
    Let's try: C = centroid of all inner hex vertices, then s = max over all constraints.
    """
    # Collect all vertices of all inner hexagons
    all_verts = []
    for x, y, ang in inner_positions:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.extend(verts.tolist())
    
    all_verts = np.array(all_verts)
    
    best_side = 1e18
    best_center = (0, 0)
    best_outer_angle = 0
    
    # Try a few outer angles
    for outer_angle in range(0, 60, 5):
        # Compute face normals for this outer angle
        normals = []
        for i in range(6):
            ang = math.radians(outer_angle + 30 + 60 * i)
            normals.append(np.array([math.cos(ang), math.sin(ang)]))
        
        # Estimate center as centroid
        centroid = np.mean(all_verts, axis=0)
        
        # For this center, compute required side
        max_val = 0
        for n in normals:
            proj = all_verts @ n
            max_proj = np.max(proj)
            val = (max_proj - n @ centroid) / (math.sqrt(3) / 2)
            if val > max_val:
                max_val = val
        
        # Also try a few offsets around centroid
        for dx in [-0.5, 0, 0.5]:
            for dy in [-0.5, 0, 0.5]:
                C = centroid + np.array([dx, dy])
                max_val2 = 0
                for n in normals:
                    proj = all_verts @ n
                    max_proj = np.max(proj)
                    val = (max_proj - n @ C) / (math.sqrt(3) / 2)
                    if val > max_val2:
                        max_val2 = val
                
                if max_val2 < best_side:
                    best_side = max_val2
                    best_center = (C[0], C[1])
                    best_outer_angle = outer_angle
    
    return best_center, best_side, best_outer_angle

def _is_valid(inner_positions, outer_center, outer_side, outer_angle, tol=1e-8):
    """Check if the configuration is valid: no overlaps, all inside outer hexagon."""
    # Check overlaps between inner hexagons
    for i in range(len(inner_positions)):
        for j in range(i + 1, len(inner_positions)):
            x1, y1, a1 = inner_positions[i]
            x2, y2, a2 = inner_positions[j]
            if _hexagons_overlap(x1, y1, a1, x2, y2, a2):
                return False
    
    # Check all inner hexagons are inside outer
    for x, y, a in inner_positions:
        if not _inside_outer_hexagon(x, y, a, outer_center[0], outer_center[1], outer_side, outer_angle):
            return False
    
    return True

def _score(inner_positions):
    """Compute the outer side length for a given inner configuration."""
    center, side, angle = _compute_outer_side(inner_positions)
    if not _is_valid(inner_positions, center, side, angle):
        return 1e10
    return side

def construct():
    start_time = time.time()
    random.seed(42)
    
    # Initial configuration: hexagonal lattice arrangement
    # Place 11 hexagons in a hexagonal pattern
    # Center + 6 around + 4 more? Let's do a compact arrangement.
    
    # Hexagonal lattice spacing for unit hexagons:
    # For flat-top hexagons, the distance between centers of adjacent hexagons is 2 * apothem = sqrt(3)
    # Actually, for regular hexagons of side 1, the center-to-center distance for touching is 2 * (sqrt(3)/2) = sqrt(3)
    
    spacing = math.sqrt(3)  # distance between adjacent centers
    
    # Create a hexagonal cluster
    # Row 0: 3 hexagons
    # Row 1: 4 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 1 hexagon
    # That's 11. Let me think of a better arrangement.
    
    # Actually, let's use a simple grid-like arrangement and then optimize.
    # Start with a hexagonal packing:
    positions = []
    
    # Hexagonal lattice: 
    # (0,0), (sqrt(3), 0), (0, 1.5), (-sqrt(3)/2, 1.5), (sqrt(3)/2, 1.5), etc.
    # For flat-top hexagons, neighbors are at:
    # (sqrt(3), 0), (0, 1.5), (-sqrt(3)/2, 1.5), (-sqrt(3), 0), (-sqrt(3)/2, -1.5), (0, -1.5)
    
    # Let's just use a simple initial guess and optimize with gradient-free methods
    # Start with a linear arrangement and improve
    
    # Better initial: hexagonal cluster
    # 1 in center, 6 around, 4 more
    init_positions = [
        (0, 0, 0),
        (spacing, 0, 0),
        (-spacing/2, 1.5, 0),
        (spacing/2, 1.5, 0),
        (-spacing/2, -1.5, 0),
        (spacing/2, -1.5, 0),
        (-spacing, 0, 0),
        # 4 more: extend the cluster
        (1.5*spacing, 0, 0),
        (0, 3.0, 0),
        (-spacing, 1.5, 0),
        (spacing, 1.5, 0),
    ]
    
    # Convert to list of [x, y, angle]
    inner = [[p[0], p[1], p[2]] for p in init_positions]
    
    # Compute initial score
    center, side, angle = _compute_outer_side(inner)
    current_score = side if _is_valid(inner, center, side, angle) else 1e10
    
    best_inner = [row[:] for row in inner]
    best_center = center
    best_side = side
    best_angle = angle
    best_score = current_score
    
    # If initial is invalid, try a simpler arrangement
    if best_score > 10:
        # Try a line arrangement
        inner = [[k * spacing, 0, 0] for k in range(-5, 6)]
        center, side, angle = _compute_outer_side(inner)
        if _is_valid(inner, center, side, angle):
            best_inner = [row[:] for row in inner]
            best_center = center
            best_side = side
            best_angle = angle
            best_score = side
    
    # Now do local optimization: perturb positions and angles
    # Use a simple hill-climbing / simulated annealing approach
    
    elapsed = time.time() - start_time
    max_time = DEADLINE - 5  # leave 5s margin
    
    # Simple iterative improvement
    step_size = 0.5
    for iteration in range(5000):
        if time.time() - start_time > max_time:
            break
        
        # Pick a random hexagon to move
        idx = random.randint(0, N - 1)
        
        # Save current
        old_x, old_y, old_a = inner[idx][:]
        
        # Perturb
        inner[idx][0] += random.gauss(0, step_size)
        inner[idx][1] += random.gauss(0, step_size)
        inner[idx][2] += random.gauss(0, 10)  # degrees
        
        # Compute new score
        center_new, side_new, angle_new = _compute_outer_side(inner)
        new_score = side_new if _is_valid(inner, center_new, side_new, angle_new) else 1e10
        
        if new_score < best_score:
            best_inner = [row[:] for row in inner]
            best_center = center_new
            best_side = side_new
            best_angle = angle_new
            best_score = new_score
            # Reduce step size occasionally
            step_size = max(0.05, step_size * 0.99)
        else:
            # Revert
            inner[idx] = [old_x, old_y, old_a]
            step_size = max(0.05, step_size * 0.995)
    
    # Final check
    if best_score < 1e9:
        return best_inner, best_center, best_side, best_angle
    else:
        # Fallback: return a valid but possibly suboptimal configuration
        # Use the line arrangement which should always be valid
        inner_line = [[k * spacing, 0, 0] for k in range(-5, 6)]
        center_l, side_l, angle_l = _compute_outer_side(inner_line)
        if _is_valid(inner_line, center_l, side_l, angle_l):
            return inner_line, center_l, side_l, angle_l
        else:
            # Very conservative fallback
            inner_fallback = [[k * 2.0, 0, 0] for k in range(-5, 6)]
            center_f, side_f, angle_f = _compute_outer_side(inner_fallback)
            return inner_fallback, center_f, side_f, angle_f
