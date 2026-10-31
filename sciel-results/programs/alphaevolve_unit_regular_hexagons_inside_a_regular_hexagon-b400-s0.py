import numpy as np
import time
import math

N = 11
SQRT3 = math.sqrt(3)

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts

def hexagons_overlap(v1, v2):
    """
    Check if two hexagons (defined by their 6 vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap.
    """
    axes = []
    
    # Get axes from edges of v1
    for i in range(6):
        p1 = v1[i]
        p2 = v1[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        # Normal to edge
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    
    # Get axes from edges of v2
    for i in range(6):
        p1 = v2[i]
        p2 = v2[(i + 1) % 6]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0]**2 + normal[1]**2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = float('inf')
        max1 = float('-inf')
        min2 = float('inf')
        max2 = float('-inf')
        
        for p in v1:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min1: min1 = proj
            if proj > max1: max1 = proj
            
        for p in v2:
            proj = p[0] * axis[0] + p[1] * axis[1]
            if proj < min2: min2 = proj
            if proj > max2: max2 = proj
            
        # Check for gap
        if max1 < min2 or max2 < min1:
            return False
    return True

def is_layout_valid(inner, eps=1e-5):
    """Check if all inner hexagons are disjoint, allowing for a small epsilon gap."""
    verts_list = []
    for (ix, iy, iangle) in inner:
        # We use angle_degrees in input, but internal check can be simplified if we just check distance
        # However, SAT is robust. Let's use SAT.
        # Note: The 'inner' list contains (x, y, angle_deg).
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        verts_list.append(verts)
    
    n = len(verts_list)
    for i in range(n):
        for j in range(i + 1, n):
            if hexagons_overlap(verts_list[i], verts_list[j]):
                return False
    return True

def compute_required_side(inner, outer_center, outer_angle):
    """Compute the minimum outer side length to contain all inner hexagons."""
    if not inner:
        return 0.0
    
    ox, oy = outer_center
    # Normals for a hexagon with angle 0 (flat top/bottom? No, vertex at 0 is pointy right).
    # If angle=0, vertices are at 0, 60, 120...
    # The sides are perpendicular to these.
    # The outward normals for the sides of a hexagon centered at origin with side S and rotation 0:
    # The vertices are at angles 0, 60, ...
    # The edges connect v0-v1, v1-v2, etc.
    # The normal to the edge between v0(1,0) and v1(0.5, sqrt3/2) is perpendicular to (v1-v0).
    # v1-v0 = (-0.5, sqrt3/2). Normal = (sqrt3/2, 0.5) which is angle 30.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_projection = 0.0
    
    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle))
    sin_a = math.sin(math.radians(outer_angle))
    
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate to align with outer hexagon axes
            # If outer hex is rotated by alpha, we rotate points by -alpha to check against standard hex
            lx = dx * cos_a + dy * sin_a  # Wait, rotation matrix for -alpha:
            # [cos(-a) -sin(-a)] [cos(a) sin(a)]
            # [sin(-a)  cos(-a)] [-sin(a) cos(a)]
            # So lx = dx cos(a) + dy sin(a)
            # ly = -dx sin(a) + dy cos(a)
            ly = -dx * sin_a + dy * cos_a
            
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj
    
    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    # 1. Define 11 hexagons on a hexagonal lattice
    # Axial coordinates (q, r)
    # Center (0,0)
    # Ring 1 (6 hexagons)
    # Ring 2 partial (4 hexagons)
    
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]
    
    # Ensure we have exactly N
    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]
    elif len(axial_coords) < N:
        # Fallback: add more from ring 2
        # This shouldn't happen for N=11 with the above list
        pass

    # 2. Convert axial to Cartesian
    # Flat-Top lattice (vertices at 0, 60, ...):
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    # Add epsilon to prevent floating point overlap detection issues
    epsilon = 1e-5
    scale = 1.0 + epsilon
    
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))
        
    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]
    
    # Verify initial layout validity
    if not is_layout_valid(initial_inner):
        # Fallback to sparse layout if something unexpected happens
        initial_inner = [[i * 2.5, 0.0, 0.0] for i in range(N)]
        centers = [(i * 2.5, 0.0) for i in range(N)]

    best_side = float('inf')
    best_inner = initial_inner
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Optimize Outer Hexagon
    # Try different outer rotations
    for outer_angle_deg in [0, 30, 60, 90, 120, 150]:
        side = compute_required_side(initial_inner, (0.0, 0.0), outer_angle_deg)
        if side < best_side:
            best_side = side
            best_outer_angle = outer_angle_deg

    # Optimize outer center
    # Centroid of inner hexagons
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)
    
    # Local search on outer center
    current_cx = cx_avg
    current_cy = cy_avg
    
    # Coarse grid search around centroid
    step = 0.5
    for _ in range(2):
        improved = False
        for dx in np.linspace(-step, step, 5):
            for dy in np.linspace(-step, step, 5):
                oc = (current_cx + dx, current_cy + dy)
                side = compute_required_side(best_inner, oc, best_outer_angle)
                if side < best_side - 1e-5:
                    best_side = side
                    best_outer_center = oc
                    current_cx = oc[0]
                    current_cy = oc[1]
                    improved = True
        if not improved:
            break
        step *= 0.5

    # 4. Try rotating individual inner hexagons to see if we can pack tighter?
    # For a hexagonal lattice of identical hexagons, rotating them usually doesn't help
    # unless the lattice itself is distorted, which we can't do easily without breaking the lattice structure.
    # However, we can try a few global rotations of the entire cluster relative to the outer hex.
    # This is equivalent to changing the outer angle, which we already did.
    
    # Let's try one more thing: Perturb the positions slightly?
    # Given the tightness of the lattice packing (they touch), perturbing might cause overlaps.
    # The epsilon scale ensures they are disjoint.
    
    # Add a tiny epsilon to the side length for floating point safety in the evaluator
    final_side = best_side + 1e-9
    
    return best_inner, best_outer_center, final_side, best_outer_angle
