"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11

def hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon with center (cx, cy), side length, and rotation in degrees."""
    angle = np.radians(angle_deg)
    angles = angle + np.pi / 6 + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside or on the boundary of a regular hexagon."""
    # Transform to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle = np.radians(angle_deg)
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    # Rotate point by -angle
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # In local coordinates, a regular hexagon with side length s centered at origin
    # has vertices at angles 30, 90, 150, 210, 270, 330 degrees
    # The hexagon is defined by |x| <= s*sqrt(3)/2 for the "flat" sides and constraints for pointy sides
    # Actually, let's use the standard form: a regular hexagon with flat top/bottom
    # Vertices at (s*cos(30+k*60), s*sin(30+k*60)) for k=0..5
    # This means the hexagon extends from -s*sqrt(3)/2 to s*sqrt(3)/2 in x
    # and from -s to s in y? No.
    
    # Let me reconsider. A regular hexagon with side length s:
    # If oriented with flat top and bottom, vertices are at:
    # (s*sqrt(3)/2, s/2), (s*sqrt(3)/2, -s/2), (0, -s), (-s*sqrt(3)/2, -s/2), (-s*sqrt(3)/2, s/2), (0, s)
    # Wait, that's not right either. Let me use the vertex positions I defined above.
    
    # The distance from center to each side (apothem) is s * sqrt(3) / 2
    # The hexagon can be described as the intersection of 6 half-planes.
    # For a hexagon with vertices at angles 30, 90, 150, 210, 270, 330 (in local coords),
    # the sides are perpendicular to directions 0, 60, 120, 180, 240, 300 degrees.
    # The apothem (distance from center to each side) is s * sqrt(3) / 2.
    
    apothem = side * np.sqrt(3) / 2
    
    # The 6 normal directions for the sides (pointing outward)
    normal_angles = np.arange(6) * np.pi / 3  # 0, 60, 120, 180, 240, 300 degrees
    
    for na in normal_angles:
        nx = np.cos(na)
        ny = np.sin(na)
        # Distance from center to the side along this normal
        dist = lx * nx + ly * ny
        if dist > apothem + 1e-9:
            return False
    
    return True

def min_clearance_to_outer_boundary(vertices, outer_cx, outer_cy, outer_side, outer_angle):
    """Compute the minimum distance from any inner vertex to the outer hexagon boundary.
    Returns negative if any vertex is outside."""
    min_dist = float('inf')
    
    # Outer hexagon vertices
    outer_verts = hex_vertices(outer_cx, outer_cy, outer_side, outer_angle)
    
    for vx, vy in vertices:
        # Check if point is inside outer hexagon
        if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            # Point is outside, compute distance to boundary
            # Find minimum distance to any edge of outer hexagon
            min_edge_dist = float('inf')
            for i in range(6):
                x1, y1 = outer_verts[i]
                x2, y2 = outer_verts[(i + 1) % 6]
                # Distance from point to line segment
                dx = x2 - x1
                dy = y2 - y1
                if dx == 0 and dy == 0:
                    dist = np.sqrt((vx - x1)**2 + (vy - y1)**2)
                else:
                    t = ((vx - x1) * dx + (vy - y1) * dy) / (dx**2 + dy**2)
                    t = max(0, min(1, t))
                    proj_x = x1 + t * dx
                    proj_y = y1 + t * dy
                    dist = np.sqrt((vx - proj_x)**2 + (vy - proj_y)**2)
                min_edge_dist = min(min_edge_dist, dist)
            return -min_edge_dist  # Negative indicates outside
        
        # Point is inside, compute distance to boundary
        # Distance to boundary = min over all 6 sides of (distance from point to side line)
        # But only for sides where the projection falls within the side segment
        # Actually, for a convex polygon, the distance to the boundary is the minimum
        # distance to any edge.
        
        # Use the apothem-based approach: for each of the 6 half-planes defining the hexagon,
        # the distance to that side is apothem - (lx * nx + ly * ny) where (lx, ly) is the point
        # in the outer hexagon's local coordinates.
        
        dx = vx - outer_cx
        dy = vy - outer_cy
        angle = np.radians(outer_angle)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        
        apothem = outer_side * np.sqrt(3) / 2
        normal_angles = np.arange(6) * np.pi / 3
        
        min_side_dist = float('inf')
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist_to_side = apothem - (lx * nx + ly * ny)
            min_side_dist = min(min_side_dist, dist_to_side)
        
        min_dist = min(min_dist, min_side_dist)
    
    return min_dist

def check_inner_overlap(inner_positions):
    """Check if any two inner hexagons overlap. Returns True if no overlap."""
    n = len(inner_positions)
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, a1 = inner_positions[i]
            cx2, cy2, a2 = inner_positions[j]
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            # Two unit hexagons overlap if center distance < 2 (for same orientation)
            # More precisely, the minimum distance between centers for non-overlap
            # depends on orientations. A safe lower bound is 2 * (sqrt(3)/2) = sqrt(3)
            # but for different orientations it can be less.
            # For simplicity, use a conservative check: if dist < 2*0.9, they likely overlap
            # Actually, the minimum center-to-center distance for two non-overlapping unit hexagons
            # is at least sqrt(3) (when aligned) and can be as low as ~1.73 for some orientations.
            # Let's use a threshold of 1.9 as a rough check.
            if dist < 1.85:
                return False
    return True

def objective(params, N):
    """Objective function: minimize negative min clearance (i.e., maximize min clearance).
    params: [cx1, cy1, a1, cx2, cy2, a2, ..., cX, cY, S, A]
    where the last 4 are outer center x, y, side, angle."""
    n = N
    inner_params = params[:3 * n]
    outer_cx = params[3 * n]
    outer_cy = params[3 * n + 1]
    outer_side = params[3 * n + 2]
    outer_angle = params[3 * n + 3]
    
    if outer_side <= 0:
        return 1e6
    
    # Parse inner positions
    inner_positions = []
    all_vertices = []
    for i in range(n):
        cx = inner_params[3 * i]
        cy = inner_params[3 * i + 1]
        a = inner_params[3 * i + 2]
        inner_positions.append((cx, cy, a))
        verts = hex_vertices(cx, cy, 1.0, a)
        all_vertices.append(verts)
    
    # Check inner overlaps
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, a1 = inner_positions[i]
            cx2, cy2, a2 = inner_positions[j]
            dist = np.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2)
            if dist < 1.85:
                return 1e6 + (1.85 - dist)
    
    # Compute min clearance to outer boundary
    min_clearance = float('inf')
    for verts in all_vertices:
        clearance = min_clearance_to_outer_boundary(verts, outer_cx, outer_cy, outer_side, outer_angle)
        if clearance < 0:
            return 1e6 - clearance  # Large penalty if outside
        min_clearance = min(min_clearance, clearance)
    
    # We want to minimize outer_side while maximizing clearance
    # The objective should be: outer_side - lambda * min_clearance
    # But we want to find the minimum outer_side such that clearance >= 0
    # So let's just return outer_side if all clearances are >= 0, else a penalty
    if min_clearance < 0:
        return 1e6 + abs(min_clearance)
    
    return outer_side

def construct():
    N = 11
    
    # Initial guess: place hexagons in a grid pattern
    # For N=11, use a 3x4 or similar arrangement
    # Let's start with a hexagonal packing pattern
    
    # Hexagonal lattice spacing: for unit hexagons, center-to-center distance is 2*sqrt(3)/2 = sqrt(3) for adjacent
    dx = np.sqrt(3)
    dy = 1.5
    
    # Arrange in rows
    positions = []
    # Row 0: 4 hexagons
    for k in range(4):
        positions.append([k * dx, 0.0, 0.0])
    # Row 1: 3 hexagons
    for k in range(3):
        positions.append([(k + 0.5) * dx, dy, 0.0])
    # Row 2: 4 hexagons
    for k in range(4):
        positions.append([k * dx, 2 * dy, 0.0])
    
    # That's 11 positions
    assert len(positions) == N
    
    # Initial outer hexagon
    # Compute bounding box
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    # Estimate side length
    width = max(xs) - min(xs)
    height = max(ys) - min(ys)
    # For a hexagon, side length S has width = S*sqrt(3) and height = 2*S
    # So S = max(width/sqrt(3), height/2)
    S = max(width / np.sqrt(3), height / 2) * 1.2  # Add some margin
    A = 0.0
    
    # Initial params
    params = []
    for p in positions:
        params.extend([p[0], p[1], p[2]])
    params.extend([cx, cy, S, A])
    
    # Use differential evolution for global search
    # Bounds
    bounds = []
    for i in range(N):
        bounds.extend([(-10, 10), (-10, 10), (-180, 180)])
    bounds.extend([(-10, 10), (-10, 10), (1, 20), (-180, 180)])
    
    # Run differential evolution
    result = differential_evolution(
        objective, bounds, args=(N,),
        maxiter=100, popsize=15, tol=1e-4,
        mutation=(0.5, 1.0), recombination=0.7,
        seed=42, polish=False
    )
    
    best_params = result.x
    n = N
    inner_positions = []
    for i in range(n):
        inner_positions.append([best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]])
    
    outer_cx = best_params[3 * n]
    outer_cy = best_params[3 * n + 1]
    outer_side = best_params[3 * n + 2]
    outer_angle = best_params[3 * n + 3]
    
    # Refine with L-BFGS-B
    def obj_refine(params):
        return objective(params, N)
    
    result2 = minimize(
        obj_refine, best_params, method='L-BFGS-B',
        bounds=bounds, options={'maxiter': 200, 'ftol': 1e-6}
    )
    
    if result2.fun < result.fun:
        best_params = result2.x
    
    n = N
    inner_positions = []
    for i in range(n):
        inner_positions.append([best_params[3 * i], best_params[3 * i + 1], best_params[3 * i + 2]])
    
    outer_cx = best_params[3 * n]
    outer_cy = best_params[3 * n + 1]
    outer_side = best_params[3 * n + 2]
    outer_angle = best_params[3 * n + 3]
    
    return inner_positions, (outer_cx, outer_cy), outer_side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
