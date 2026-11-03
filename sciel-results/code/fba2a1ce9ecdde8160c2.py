import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon with given center, rotation, side length."""
    angle = np.radians(angle_deg)
    # Vertices at angles: angle + k*60 degrees, k=0..5
    # A regular hexagon with side s has circumradius = s
    angles = angle + np.pi / 3 * np.arange(6)
    xs = cx + side * np.cos(angles)
    ys = cy + side * np.sin(angles)
    return np.column_stack([xs, ys])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon with center (cx,cy), side, rotation."""
    angle = np.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    ca, sa = np.cos(-angle), np.sin(-angle)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # In local frame, hexagon vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with circumradius = side, inradius = side*sqrt(3)/2
    # The 6 normal directions are at angles 30, 90, 150, 210, 270, 330 (perpendicular to edges)
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        theta = np.pi / 6 + k * np.pi / 3  # 30 + 60k degrees
        proj = lx * np.cos(theta) + ly * np.sin(theta)
        if proj > inradius + 1e-9:
            return False
    return True

def _hexagon_contains_hexagon(inner_cx, inner_cy, inner_angle, inner_side,
                               outer_cx, outer_cy, outer_side, outer_angle):
    """Check if inner hexagon is fully contained in outer hexagon."""
    verts = _hex_vertices(inner_cx, inner_cy, inner_angle, inner_side)
    for vx, vy in verts:
        if not _point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
            return False
    return True

def _check_no_overlap(hex_positions, tolerance=1e-9):
    """Check that no two hexagons overlap. hex_positions: list of (cx, cy, angle_deg)."""
    n = len(hex_positions)
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, a1 = hex_positions[i]
            cx2, cy2, a2 = hex_positions[j]
            # Get vertices of both
            v1 = _hex_vertices(cx1, cy1, a1)
            v2 = _hex_vertices(cx2, cy2, a2)
            # Check if any vertex of one is inside the other, or if edges intersect
            # Simple check: minimum distance between centers must be >= 2*inradius for same orientation
            # But for arbitrary rotations, we need proper check
            # Use: check if any vertex of hex1 is inside hex2 or vice versa
            for vx, vy in v1:
                if _point_in_hexagon(vx, vy, cx2, cy2, 1.0, a2):
                    return False
            for vx, vy in v2:
                if _point_in_hexagon(vx, vy, cx1, cy1, 1.0, a1):
                    return False
            # Also check edge-edge intersections (simplified: check if midpoints of edges are inside)
            # For robustness, also check if any edge of one crosses any edge of other
            # This is expensive but necessary for correctness
            # Simplified: if no vertex is inside the other and the hexagons are convex,
            # they don't overlap unless edges cross. Let's do a more careful check.
            # Actually for convex polygons, they overlap iff one vertex is inside the other
            # OR edges intersect. Let's add edge intersection check.
            if _convex_polygons_overlap(v1, v2):
                return False
    return True

def _convex_polygons_overlap(p1, p2):
    """Check if two convex polygons overlap using separating axis theorem."""
    # Get all edge normals from both polygons
    def get_normals(poly):
        normals = []
        n = len(poly)
        for i in range(n):
            dx = poly[(i+1) % n, 0] - poly[i, 0]
            dy = poly[(i+1) % n, 1] - poly[i, 1]
            # Normal (perpendicular)
            nx, ny = -dy, dx
            length = np.sqrt(nx*nx + ny*ny)
            if length > 1e-12:
                normals.append((nx/length, ny/length))
        return normals
    
    normals = get_normals(p1) + get_normals(p2)
    
    for nx, ny in normals:
        # Project both polygons onto this normal
        proj1 = p1[:, 0] * nx + p1[:, 1] * ny
        proj2 = p2[:, 0] * nx + p2[:, 1] * ny
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
            return False  # Separating axis found, no overlap
    return True  # No separating axis, they overlap

def _min_enclosing_hexagon_size(points, cx, cy, angle_deg):
    """Given center and orientation, find minimum side length of hexagon containing all points."""
    angle = np.radians(angle_deg)
    ca, sa = np.cos(-angle), np.sin(-angle)
    # Transform points to local frame
    lx = (points[:, 0] - cx) * ca - (points[:, 1] - cy) * sa
    ly = (points[:, 0] - cx) * sa + (points[:, 1] - cy) * ca
    # In local frame, hexagon with side s has inradius = s*sqrt(3)/2
    # Point is inside if for all 6 normal directions, projection <= inradius
    # Normal directions at 30 + 60k degrees
    max_proj = 0
    for k in range(6):
        theta = np.pi / 6 + k * np.pi / 3
        proj = lx * np.cos(theta) + ly * np.sin(theta)
        max_proj = max(max_proj, np.abs(proj).max())
    # inradius = s * sqrt(3)/2, so s = 2 * inradius / sqrt(3)
    return 2 * max_proj / np.sqrt(3)

def _evaluate_layout(hex_positions, outer_cx, outer_cy, outer_angle):
    """Evaluate a layout: return (outer_side, valid)."""
    # Get all vertices of all inner hexagons
    all_verts = []
    for cx, cy, ang in hex_positions:
        verts = _hex_vertices(cx, cy, ang)
        all_verts.append(verts)
    all_verts = np.vstack(all_verts)
    
    # Check containment
    side = _min_enclosing_hexagon_size(all_verts, outer_cx, outer_cy, outer_angle)
    
    # Check no overlap
    if not _check_no_overlap(hex_positions):
        return float('inf'), False
    
    return side, True

def _optimize_layout(hex_positions, outer_angle, max_time=10):
    """Optimize positions of hexagons to minimize enclosing hexagon side length."""
    start_time = time.time()
    current_positions = [list(p) for p in hex_positions]
    current_side, valid = _evaluate_layout(current_positions, 0, 0, outer_angle)
    if not valid:
        return current_positions, float('inf')
    
    best_positions = [list(p) for p in current_positions]
    best_side = current_side
    
    # Local optimization: move each hexagon slightly
    step = 0.1
    for iteration in range(200):
        if time.time() - start_time > max_time:
            break
        improved = False
        for i in range(len(current_positions)):
            cx, cy, ang = current_positions[i]
            # Try moving in 8 directions
            for dx, dy in [(step,0),(-step,0),(0,step),(0,-step),
                          (step,step),(step,-step),(-step,step),(-step,-step),
                          (step*0.5, step*0.5), (step*0.5, -step*0.5),
                          (-step*0.5, step*0.5), (-step*0.5, -step*0.5)]:
                new_pos = [list(p) for p in current_positions]
                new_pos[i] = [cx + dx, cy + dy, ang]
                side, valid = _evaluate_layout(new_pos, 0, 0, outer_angle)
                if valid and side < best_side - 1e-10:
                    current_positions = new_pos
                    best_side = side
                    best_positions = [list(p) for p in current_positions]
                    improved = True
        if not improved:
            step *= 0.7
            if step < 0.001:
                break
    
    return best_positions, best_side

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s
    
    # Generate initial layouts
    layouts = []
    
    # Layout 1: Hexagonal cluster (center + 6 around + 4 extra)
    # Center at origin, 6 at distance 2 in 60-degree increments
    # This gives 7 hexagons in a tight cluster
    hex_positions_1 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        hex_positions_1.append((2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0))
    # Add 4 more hexagons extending the cluster
    # Place them at the "corners" of the hexagonal arrangement
    # The 6 surrounding hexagons are at angles 0,60,120,180,240,300
    # Extend by adding hexagons at positions that continue the pattern
    # For a hexagonal lattice, next ring would be at distance 4
    # But we only need 4 more. Place them at 4 of the 6 outer positions
    # Actually, let's think: the 7-hex cluster has a hexagonal shape.
    # To add 4 more, we can extend in 4 directions.
    # Better: use a 3x3-ish arrangement
    # Let me try a different approach: linear arrangement with hexagonal offset
    
    # Layout 2: Row-based (3 rows of 4,3,4 or similar)
    # Hexagonal lattice spacing: horizontal = 2, vertical = sqrt(3)
    # Row 0: 4 hexagons at x = -3, -1, 1, 3, y = 0
    # Row 1: 3 hexagons at x = -2, 0, 2, y = sqrt(3)
    # Row 2: 4 hexagons at x = -3, -1, 1, 3, y = 2*sqrt(3)
    # That's 11 total
    sqrt3 = np.sqrt(3)
    hex_positions_2 = []
    for x in [-3.0, -1.0, 1.0, 3.0]:
        hex_positions_2.append((x, 0.0, 0.0))
    for x in [-2.0, 0.0, 2.0]:
        hex_positions_2.append((x, sqrt3, 0.0))
    for x in [-3.0, -1.0, 1.0, 3.0]:
        hex_positions_2.append((x, 2*sqrt3, 0.0))
    
    # Layout 3: Compact hexagonal (1 center, 6 ring, 4 in second partial ring)
    hex_positions_3 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        hex_positions_3.append((2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0))
    # 4 more at distance ~4 in 4 directions
    for k in range(4):
        angle = np.radians(60 * k + 30)  # Offset by 30 degrees
        hex_positions_3.append((4.0 * np.cos(angle), 4.0 * np.sin(angle), 0.0))
    
    # Layout 4: Two rows of 5 and 6, or 5 and 6 with offset
    hex_positions_4 = []
    for x in [-4.0, -2.0, 0.0, 2.0, 4.0]:
        hex_positions_4.append((x, 0.0, 0.0))
    for x in [-3.0, -1.0, 1.0, 3.0, 5.0, -5.0]:
        hex_positions_4.append((x, sqrt3, 0.0))
    
    # Layout 5: More compact - hexagonal with 1+6+4
    # The 4 extra should be placed to minimize bounding hexagon
    hex_positions_5 = [(0.0, 0.0, 0.0)]
    for k in range(6):
        angle = np.radians(60 * k)
        hex_positions_5.append((2.0 * np.cos(angle), 2.0 * np.sin(angle), 0.0))
    # Place 4 more at the "flat" sides, extending outward
    # In a hexagonal lattice, the next positions from the 6 surrounding would be
    # at (4,0), (2, 2*sqrt(3)), (-2, 2*sqrt(3)), (-4,0), (-2,-2*sqrt(3)), (2,-2*sqrt(3))
    # Pick 4 of these that give most symmetric extension
    extra_positions = [
        (4.0, 0.0), (-4.0, 0.0),
        (2.0, 2*sqrt3), (-2.0, -2*sqrt3)
    ]
    for ex, ey in extra_positions:
        hex_positions_5.append((ex, ey, 0.0))
    
    layouts = [hex_positions_1, hex_positions_2, hex_positions_3, hex_positions_4, hex_positions_5]
    
    best_side = float('inf')
    best_result = None
    
    # Try different outer hexagon orientations
    outer_angles = [0.0, 30.0, 15.0, 45.0]
    
    for layout in layouts:
        if time.time() > deadline:
            break
        # First check validity
        if not _check_no_overlap(layout):
            continue
        
        for outer_angle in outer_angles:
            if time.time() > deadline:
                break
            
            # Get all vertices
            all_verts = []
            for cx, cy, ang in layout:
                verts = _hex_vertices(cx, cy, ang)
                all_verts.append(verts)
            all_verts = np.vstack(all_verts)
            
            # Find optimal center (centroid of all vertices)
            cx_opt = all_verts[:, 0].mean()
            cy_opt = all_verts[:, 1].mean()
            
            # Optimize the layout positions
            opt_positions, side = _optimize_layout(layout, outer_angle, max_time=15)
            
            if side < best_side:
                # Recompute with optimized positions
                all_verts_opt = []
                for cx, cy, ang in opt_positions:
                    verts = _hex_vertices(cx, cy, ang)
                    all_verts_opt.append(verts)
                all_verts_opt = np.vstack(all_verts_opt)
                
                cx_final = all_verts_opt[:, 0].mean()
                cy_final = all_verts_opt[:, 1].mean()
                
                # Final side length
                final_side = _min_enclosing_hexagon_size(all_verts_opt, cx_final, cy_final, outer_angle)
                
                if final_side < best_side:
                    best_side = final_side
                    best_result = (opt_positions, (cx_final, cy_final), final_side, outer_angle)
    
    if best_result is None:
        # Fallback: return the seed-like result
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    positions, center, side, angle = best_result
    inner = [[p[0], p[1], p[2]] for p in positions]
    return inner, center, side, angle

# EVOLVE-BLOCK-END
