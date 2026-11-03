"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angle_rad = np.deg2rad(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees (flat-top orientation at 0)
    angles = np.array([0, 60, 120, 180, 240, 300]) + angle_deg
    angles_rad = np.deg2rad(angles)
    verts = np.column_stack([
        cx + side * np.cos(angles_rad),
        cy + side * np.sin(angles_rad)
    ])
    return verts

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside hexagon centered at (cx, cy) with side and rotation."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    angle_rad = np.deg2rad(angle_deg)
    # Rotate by -angle
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with flat-top orientation (angle=0), the vertices are at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The hexagon can be defined by 6 half-planes.
    # For flat-top hexagon with side length s:
    # |x| <= s
    # |sqrt(3)*x + 3*y| <= 3*s  (for y >= 0, this is the top-right edge constraint)
    # Actually, let's use the standard 6 inequalities.
    
    # The 6 edges of a flat-top hexagon (centered at origin, side s) have normals:
    # Edge 1: from (s,0) to (s/2, s*sqrt(3)/2) - normal pointing outward
    # Let's compute the 6 supporting lines.
    
    s = side
    sqrt3 = np.sqrt(3)
    
    # For a regular hexagon with vertices at angles 0,60,120,180,240,300 (flat-top):
    # The 6 face normals (outward) point at angles 30, 90, 150, 210, 270, 330 degrees
    # Face at angle 30: x*cos(30) + y*sin(30) = s*cos(30) = s*sqrt(3)/2
    # Wait, let me reconsider. The distance from center to each edge (apothem) is s*sqrt(3)/2.
    
    # The 6 outward normals for a flat-top hexagon are at angles 30, 90, 150, 210, 270, 330
    normals_angles = np.array([30, 90, 150, 210, 270, 330])
    normals_rad = np.deg2rad(normals_angles)
    normals = np.column_stack([np.cos(normals_rad), np.sin(normals_rad)])
    apothem = s * np.sqrt(3) / 2
    
    # Point is inside if dot(local_point, normal) <= apothem for all normals
    dots = lx * normals[:, 0] + ly * normals[:, 1]
    return np.all(dots <= apothem + 1e-10)

def _hexagon_overlap_area_approx(v1, v2, v3, v4, side1, side2, angle1, angle2):
    """
    Approximate whether two hexagons overlap by checking if any vertex of one is inside the other
    or vice versa. This is a conservative check.
    """
    # Get vertices of both hexagons
    verts1 = _hexagon_vertices(v1[0], v1[1], side1, angle1)
    verts2 = _hexagon_vertices(v2[0], v2[1], side2, angle2)
    
    # Check if any vertex of hex1 is inside hex2
    for vx, vy in verts1:
        if _point_in_hexagon(vx, vy, v2[0], v2[1], side2, angle2):
            return True
    
    # Check if any vertex of hex2 is inside hex1
    for vx, vy in verts2:
        if _point_in_hexagon(vx, vy, v1[0], v1[1], side1, angle1):
            return True
    
    return False

def _min_side_for_packing(inner, center, angle):
    """
    Find the minimum side length of the outer hexagon such that all inner hexagons fit inside.
    inner: list of (x, y, angle_deg)
    center: (cx, cy)
    angle: outer hexagon rotation
    Returns the minimum side length.
    """
    cx, cy = center
    outer_angle = angle
    
    # For a given outer hexagon, we need to find the minimum side length such that
    # all inner hexagons are contained within it.
    # 
    # The outer hexagon contains a point (px, py) if and only if:
    # For each of the 6 face normals n_i of the outer hexagon:
    #   dot((px - cx, py - cy) rotated by -outer_angle, n_i) <= side * apothem_factor
    #
    # Actually, let's think differently. The outer hexagon is defined by 6 half-planes.
    # A point is inside if it satisfies all 6 half-plane constraints.
    # For an inner hexagon to be inside, ALL its vertices must be inside the outer hexagon.
    #
    # So for a given side length S, the outer hexagon has 6 constraints.
    # For each inner hexagon vertex, we can compute the minimum S required.
    
    sqrt3 = np.sqrt(3)
    
    # Outer hexagon face normals (outward) for flat-top orientation:
    normals_angles = np.array([30, 90, 150, 210, 270, 330]) + outer_angle
    normals_rad = np.deg2rad(normals_angles)
    normals = np.column_stack([np.cos(normals_rad), np.sin(normals_rad)])
    
    # The apothem (distance from center to edge) for side S is S * sqrt(3) / 2
    # So the constraint is: dot(p - c, n) <= S * sqrt(3) / 2
    # => S >= dot(p - c, n) / (sqrt(3) / 2) = 2 * dot(p - c, n) / sqrt(3)
    
    min_side = 0.0
    
    for (ix, iy, iangle) in inner:
        verts = _hexagon_vertices(ix, iy, 1.0, iangle)
        for vx, vy in verts:
            dx = vx - cx
            dy = vy - cy
            for n in normals:
                dot = dx * n[0] + dy * n[1]
                required_S = 2.0 * dot / sqrt3
                if required_S > min_side:
                    min_side = required_S
    
    return max(min_side, 0.0)

def _packing_objective(params, inner_angles, center, outer_angle):
    """
    Objective function for optimization.
    params: flattened array of [x0, y0, x1, y1, ..., x10, y10]
    inner_angles: list of angles for each inner hexagon
    center: (cx, cy) for outer hexagon
    outer_angle: rotation of outer hexagon
    """
    n = len(inner_angles)
    inner = []
    for i in range(n):
        x = params[2*i]
        y = params[2*i+1]
        angle = inner_angles[i]
        inner.append((x, y, angle))
    
    # Check for overlaps between inner hexagons
    for i in range(n):
        for j in range(i+1, n):
            if _hexagon_overlap_area_approx(
                (inner[i][0], inner[i][1]), (inner[j][0], inner[j][1]),
                1.0, 1.0, inner[i][2], inner[j][2]
            ):
                return 1e6  # Large penalty for overlap
    
    # Compute the minimum outer side length
    side = _min_side_for_packing(inner, center, outer_angle)
    return side

def construct():
    # Arrange 11 hexagons in a 1-2-3-2-1 pattern with one rotated
    # Start with a symmetric arrangement, then optimize
    
    sqrt3 = np.sqrt(3)
    
    # Initial arrangement: 1-2-3-2-1 cluster
    # Row 0 (top): 1 hexagon
    # Row 1: 2 hexagons
    # Row 2 (middle): 3 hexagons
    # Row 3: 2 hexagons
    # Row 4 (bottom): 1 hexagon
    
    # For unit hexagons in a tight hexagonal lattice:
    # Horizontal spacing between centers in same row: 2 * (sqrt(3)/2) = sqrt(3)
    # Vertical spacing between rows: 1.5
    # Horizontal offset between rows: sqrt(3)/2
    
    positions = []
    angles = []
    
    # Row 0: 1 hex at top
    positions.append((0.0, 3.0))
    angles.append(0.0)
    
    # Row 1: 2 hexes
    positions.append((-sqrt3/2, 1.5))
    angles.append(0.0)
    positions.append((sqrt3/2, 1.5))
    angles.append(0.0)
    
    # Row 2: 3 hexes (middle)
    positions.append((-sqrt3, 0.0))
    angles.append(0.0)
    positions.append((0.0, 0.0))
    angles.append(0.0)  # This one will be rotated
    positions.append((sqrt3, 0.0))
    angles.append(0.0)
    
    # Row 3: 2 hexes
    positions.append((-sqrt3/2, -1.5))
    angles.append(0.0)
    positions.append((sqrt3/2, -1.5))
    angles.append(0.0)
    
    # Row 4: 1 hex at bottom
    positions.append((0.0, -3.0))
    angles.append(0.0)
    
    # Rotate the center hexagon by 30 degrees to create a "defect"
    angles[4] = 30.0
    
    n = len(positions)
    assert n == 11
    
    # Initial parameter vector
    x0 = np.array([p[0] for p in positions] + [p[1] for p in positions])
    
    # Optimize positions
    center = (0.0, 0.0)
    outer_angle = 0.0
    
    best_result = None
    best_side = _min_side_for_packing(
        [(positions[i][0], positions[i][1], angles[i]) for i in range(n)],
        center, outer_angle
    )
    
    # Try optimization with different outer angles
    for oa in [0.0, 30.0]:
        center = (0.0, 0.0)
        outer_angle = oa
        
        def objective(params):
            return _packing_objective(params, angles, center, outer_angle)
        
        try:
            result = minimize(
                objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if result.success or result.fun < best_side:
                # Extract final positions
                final_inner = []
                for i in range(n):
                    fx = result.x[2*i]
                    fy = result.x[2*i+1]
                    fa = angles[i]
                    final_inner.append((fx, fy, fa))
                
                # Check for overlaps
                has_overlap = False
                for i in range(n):
                    for j in range(i+1, n):
                        if _hexagon_overlap_area_approx(
                            (final_inner[i][0], final_inner[i][1]),
                            (final_inner[j][0], final_inner[j][1]),
                            1.0, 1.0, final_inner[i][2], final_inner[j][2]
                        ):
                            has_overlap = True
                            break
                    if has_overlap:
                        break
                
                if not has_overlap:
                    side = _min_side_for_packing(final_inner, center, outer_angle)
                    if side < best_side:
                        best_side = side
                        best_result = (final_inner, center, outer_angle)
        except Exception:
            pass
    
    if best_result is not None:
        final_inner, center, outer_angle = best_result
        side = best_side
    else:
        # Fallback to initial arrangement
        final_inner = [(positions[i][0], positions[i][1], angles[i]) for i in range(n)]
        center = (0.0, 0.0)
        outer_angle = 0.0
        side = _min_side_for_packing(final_inner, center, outer_angle)
    
    # Convert to the required format
    inner = [[x, y, a] for (x, y, a) in final_inner]
    
    return inner, center, side, outer_angle
# EVOLVE-BLOCK-END
