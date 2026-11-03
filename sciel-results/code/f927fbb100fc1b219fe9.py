"""Hexagon packing: 12 unit hexagons in minimal regular hexagon."""
import numpy as np
import time

N = 12
TIME_LIMIT = 140.0  # safety margin below 150s


def hexagon_vertices(cx, cy, side, angle_deg=0.0):
    """Return vertices of a regular hexagon (side length `side`) centered at (cx,cy).
    angle_deg rotates the hexagon. Vertices ordered CCW."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    r = side
    xs = cx + r * np.cos(angles)
    ys = cy + r * np.sin(angles)
    return np.column_stack([xs, ys])


def _pt_seg_dist(p, a, b):
    """Distance from point p to segment a-b."""
    ab = b - a
    ap = p - a
    ab_sq = np.dot(ab, ab)
    if ab_sq < 1e-20:
        return np.linalg.norm(p - a)
    t = np.dot(ap, ab) / ab_sq
    t = max(0.0, min(1.0, t))
    proj = a + t * ab
    return np.linalg.norm(p - proj)


def _segments_intersect(p1, p2, p3, p4):
    """Check if segments p1-p2 and p3-p4 intersect."""
    def orient(a, b, c):
        val = (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
        if abs(val) < 1e-10:
            return 0
        return 1 if val > 0 else -1
    
    o1 = orient(p1, p2, p3)
    o2 = orient(p1, p2, p4)
    o3 = orient(p3, p4, p1)
    o4 = orient(p3, p4, p2)
    
    if o1 != o2 and o3 != o4:
        return True
    
    if o1 == 0 and _on_segment(p1, p3, p2): return True
    if o2 == 0 and _on_segment(p1, p4, p2): return True
    if o3 == 0 and _on_segment(p3, p1, p4): return True
    if o4 == 0 and _on_segment(p3, p2, p4): return True
    
    return False


def _on_segment(a, b, c):
    """Check if point b lies on segment a-c (collinear)."""
    if (min(a[0], c[0]) - 1e-10 <= b[0] <= max(a[0], c[0]) + 1e-10 and
        min(a[1], c[1]) - 1e-10 <= b[1] <= max(a[1], c[1]) + 1e-10):
        return True
    return False


def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    if _segments_intersect(p1, p2, p3, p4):
        return 0.0
    
    d1 = _pt_seg_dist(p1, p3, p4)
    d2 = _pt_seg_dist(p2, p3, p4)
    d3 = _pt_seg_dist(p3, p1, p2)
    d4 = _pt_seg_dist(p4, p1, p2)
    return min(d1, d2, d3, d4)


def min_distance_hexagons(v1, v2):
    """Minimum distance between two convex polygons (hexagons).
    Returns the minimum distance between any edge of v1 and any edge of v2.
    If polygons overlap, returns negative value."""
    n1, n2 = len(v1), len(v2)
    min_d = np.inf
    
    for i in range(n1):
        for j in range(n2):
            d = _seg_seg_dist(v1[i], v1[(i+1) % n1], v2[j], v2[(j+1) % n2])
            if d < min_d:
                min_d = d
    return min_d


def point_in_hexagon(p, cx, cy, side, angle_deg=0.0):
    """Check if point p is inside or on the boundary of the hexagon."""
    verts = hexagon_vertices(cx, cy, side, angle_deg)
    n = len(verts)
    inside = True
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i + 1) % n]
        cross = (v2[0] - v1[0]) * (p[1] - v1[1]) - (v2[1] - v1[1]) * (p[0] - v1[0])
        if cross < -1e-9:
            inside = False
            break
    return inside


def compute_min_outer_side(inner_centers, inner_angles, outer_center, outer_angle):
    """Compute the minimum outer side length needed to contain all inner hexagons,
    given fixed outer center and angle. Returns the required side length."""
    theta0 = np.radians(outer_angle)
    normals = []
    for k in range(6):
        ang = theta0 + np.pi/6 + k * np.pi / 3
        normals.append(np.array([np.cos(ang), np.sin(ang)]))
    
    max_val = 0.0
    for ic, ia in zip(inner_centers, inner_angles):
        verts = hexagon_vertices(ic[0], ic[1], 1.0, ia)
        for v in verts:
            rel = v - np.array(outer_center)
            for n in normals:
                proj = np.dot(rel, n)
                if proj > max_val:
                    max_val = proj
    
    required_R = max_val / (np.sqrt(3) / 2)
    return required_R


def is_valid_packing(centers, angles, outer_center, outer_side, outer_angle, tolerance=1e-7):
    """Check if the packing is valid: no overlaps and all inside outer hexagon."""
    # Check inner-inner overlaps
    for i in range(len(centers)):
        v1 = hexagon_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        for j in range(i + 1, len(centers)):
            v2 = hexagon_vertices(centers[j][0], centers[j][1], 1.0, angles[j])
            d = min_distance_hexagons(v1, v2)
            if d < -tolerance:
                return False
    
    # Check all inner hexagons are inside outer hexagon
    outer_verts = hexagon_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
    for i in range(len(centers)):
        v1 = hexagon_vertices(centers[i][0], centers[i][1], 1.0, angles[i])
        for v in v1:
            if not point_in_hexagon(v, outer_center[0], outer_center[1], outer_side, outer_angle):
                return False
    
    return True


def generate_ring12(scale=1.0):
    """Generate a ring of 12 hexagons around a center."""
    centers = []
    angles = []
    # 12 hexagons in a ring, radius ~ 2*sqrt(3) for unit hexagons
    r = 2 * np.sqrt(3) * scale
    for i in range(12):
        ang = i * np.pi / 6
        x = r * np.cos(ang)
        y = r * np.sin(ang)
        centers.append([x, y])
        angles.append(0.0)
    return centers, angles


def generate_hex6plus6(scale=1.0):
    """Generate two concentric rings of 6 hexagons each."""
    centers = []
    angles = []
    # Inner ring of 6
    r1 = np.sqrt(3) * scale
    for i in range(6):
        ang = i * np.pi / 3
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        centers.append([x, y])
        angles.append(0.0)
    # Outer ring of 6
    r2 = 2 * np.sqrt(3) * scale
    for i in range(6):
        ang = i * np.pi / 3 + np.pi / 6
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        centers.append([x, y])
        angles.append(0.0)
    return centers, angles


def generate_rect3x4(scale=1.0):
    """Generate a 3x4 rectangular grid of hexagons."""
    centers = []
    angles = []
    dx = 1.75 * scale
    dy = 1.5 * scale
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx * 0.5
            y = row * dy
            centers.append([x - 1.5 * dx, y - dy])
            angles.append(0.0)
    return centers, angles


def generate_hex_grid(scale=1.0):
    """Generate a hexagonal grid arrangement."""
    centers = []
    angles = []
    # 4-3-4-1 or similar hex grid pattern
    # Let's use a 3x4 offset grid
    dx = 1.75 * scale
    dy = 1.5 * scale
    for row in range(3):
        for col in range(4):
            x = col * dx + (row % 2) * dx * 0.5
            y = row * dy
            centers.append([x - 1.5 * dx, y - dy])
            angles.append(0.0)
    return centers, angles


def find_best_outer(centers, angles, center_offsets=None, angle_range=None):
    """Find the best outer hexagon (center, angle, side) for given inner arrangement."""
    if center_offsets is None:
        center_offsets = [(dx, dy) for dx in [-0.5, 0, 0.5] for dy in [-0.5, 0, 0.5]]
    if angle_range is None:
        angle_range = list(range(0, 360, 15))
    
    centroid = np.mean(centers, axis=0)
    best_side = np.inf
    best_center = None
    best_angle = None
    
    for ang in angle_range:
        for dx, dy in center_offsets:
            oc = [centroid[0] + dx, centroid[1] + dy]
            side = compute_min_outer_side(centers, angles, oc, ang)
            if side < best_side:
                best_side = side
                best_center = oc
                best_angle = ang
    
    return best_center, best_angle, best_side


def construct():
    start_time = time.time()
    
    # Generate multiple packing topologies
    generators = [
        ("ring12", generate_ring12),
        ("hex6+6", generate_hex6plus6),
        ("rect3x4", generate_rect3x4),
        ("hex_grid", generate_hex_grid),
    ]
    
    best_result = None
    best_side = np.inf
    
    for name, gen_func in generators:
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        
        # Try multiple scales
        for scale in [0.8, 0.9, 1.0, 1.1, 1.2]:
            if time.time() - start_time > TIME_LIMIT - 30:
                break
            
            centers, angles = gen_func(scale)
            
            # Find best outer hexagon
            center, angle, side = find_best_outer(centers, angles)
            
            if side < best_side:
                # Verify validity
                if is_valid_packing(centers, angles, center, side, angle):
                    best_side = side
                    best_result = (centers, angles, center, angle, side)
    
    if best_result is None:
        # Fallback: use a simple grid with generous outer hexagon
        centers, angles = generate_rect3x4(1.0)
        center, angle, side = find_best_outer(centers, angles)
        # If still invalid, just use a large outer hexagon
        if not is_valid_packing(centers, angles, center, side, angle):
            # Expand outer hexagon until valid
            for expansion in [1.1, 1.2, 1.5, 2.0]:
                side_new = side * expansion
                if is_valid_packing(centers, angles, center, side_new, angle):
                    side = side_new
                    break
        best_result = (centers, angles, center, angle, side)
    
    centers, angles, outer_center, outer_angle, outer_side = best_result
    
    # Build inner list
    inner = []
    for c, a in zip(centers, angles):
        inner.append([float(c[0]), float(c[1]), float(a)])
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)


if __name__ == '__main__':
    result = construct()
    print(f"Outer side length: {result[2]:.6f}")
    print(f"Outer center: {result[1]}, angle: {result[3]}")
    for i, row in enumerate(result[0]):
        print(f"  Inner {i}: ({row[0]:.4f}, {row[1]:.4f}, {row[2]:.2f} deg)")
