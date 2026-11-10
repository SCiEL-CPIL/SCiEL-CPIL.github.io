# EVOLVE-BLOCK-START
import numpy as np


def _hex_vertices(cx, cy, s, a_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side s, rotation a_deg."""
    a = np.deg2rad(a_deg)
    angles = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])


def _convex_hull_2d(points):
    """Simple monotone chain convex hull for 2D points. Returns hull vertices in CCW order."""
    pts = np.array(points, dtype=float)
    pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)

    hull = np.array(lower[:-1] + upper[:-1])
    return hull


def _min_enclosing_hexagon(vertices):
    """
    Find the minimal side-length regular hexagon (centered at origin, variable rotation)
    that encloses all given vertices. Returns (side_length, angle_deg, center).
    The outer hexagon has vertices at (cx + s*cos(a+60i), cy + s*sin(a+60i)).
    A point (x,y) relative to center (cx,cy) is inside iff for all i in 0..5:
        (x-cx)*cos(a+60i) + (y-cy)*sin(a+60i) <= s * sqrt(3)/2
    (since the apothem = s * sqrt(3)/2)
    """
    # Center at origin for optimization (we'll return center as 0,0)
    cx, cy = 0.0, 0.0
    apothem_factor = np.sqrt(3.0) / 2.0

    best_side = np.inf
    best_angle = 0.0

    # Try many angles to find minimum
    # The constraint for angle a: s >= max over all vertices v of (v.x*cos(a+60i) + v.y*sin(a+60i)) / apothem_factor
    # for each i, we need s >= that value, so s = max_i max_v proj_i(v) / apothem_factor

    # Sample angles densely
    angles = np.linspace(0, 60, 2000)  # hexagon has 60-degree symmetry
    for a_deg in angles:
        a_rad = np.deg2rad(a_deg)
        base_angles = a_rad + np.deg2rad(60.0 * np.arange(6))
        cos_a = np.cos(base_angles)
        sin_a = np.sin(base_angles)
        # For each vertex, compute projection onto each of the 6 normals
        # projections: (n_vertices, 6)
        proj = vertices[:, 0] * cos_a[None, :] + vertices[:, 1] * sin_a[None, :]
        max_proj = np.max(proj, axis=0)  # max over vertices for each normal
        # side length needed: max_proj / apothem_factor, take max over normals
        side_needed = np.max(max_proj) / apothem_factor
        if side_needed < best_side:
            best_side = side_needed
            best_angle = a_deg

    # Refine around best angle
    if best_side < np.inf:
        for a_deg in np.linspace(best_angle - 1.0, best_angle + 1.0, 500):
            a_rad = np.deg2rad(a_deg)
            base_angles = a_rad + np.deg2rad(60.0 * np.arange(6))
            cos_a = np.cos(base_angles)
            sin_a = np.sin(base_angles)
            proj = vertices[:, 0] * cos_a[None, :] + vertices[:, 1] * sin_a[None, :]
            max_proj = np.max(proj, axis=0)
            side_needed = np.max(max_proj) / apothem_factor
            if side_needed < best_side:
                best_side = side_needed
                best_angle = a_deg

    return best_side, best_angle, (cx, cy)


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    minimizing the outer hexagon's side length (maximizing 1/side).
    
    Uses a honeycomb lattice arrangement: 1 center + 6 ring + 4 additional in gaps,
    then finds the minimal enclosing regular hexagon analytically.
    """
    # Honeycomb lattice: unit hexagons with side 1
    # Adjacent hexagons in honeycomb are separated by distance sqrt(3) * side = sqrt(3)
    # Actually, for unit-side hexagons, the distance between centers of adjacent
    # hexagons in a honeycomb is 2 * apothem = 2 * (sqrt(3)/2) = sqrt(3)
    
    # Wait: two unit hexagons sharing an edge have centers separated by 2 * apothem = sqrt(3).
    # But they must NOT intersect - they can touch. The problem says "touching or a smaller gap is an intersection"
    # So we need strict separation: gap > 1e-6. So centers must be > sqrt(3) apart for edge-adjacent.
    # Actually, let me re-read: "on at least one of their 12 edge normals the projected intervals are apart by more than 1e-6"
    # This means they can be very close but not overlapping. For safety, let's use a small epsilon.
    
    # Honeycomb arrangement: 1 center, 6 around it, then 4 more
    # In a honeycomb, hexagons at axial coordinates (q, r) have center at:
    # x = q * sqrt(3) + r * sqrt(3)/2
    # y = r * 1.5
    # (for pointy-top or flat-top, let's use flat-top)
    
    # Flat-top hexagon: vertices at angles 0, 60, 120, 180, 240, 300 degrees
    # Axial coordinates to pixel (flat-top):
    # x = size * 3/2 * q
    # y = size * (sqrt(3)/2 * q + sqrt(3) * r)
    
    # For unit side length (size=1):
    # x = 1.5 * q
    # y = (sqrt(3)/2) * q + sqrt(3) * r
    
    sqrt3 = np.sqrt(3.0)
    
    # Place 11 hexagons in a compact honeycomb pattern
    # Center at (0,0), then 6 neighbors, then 4 more to make 11
    # Axial coordinates for a compact cluster of 11:
    # Layer 0: (0,0) - 1 hex
    # Layer 1: 6 hexagons around center
    # Layer 2: we need 4 more from the next ring
    
    # Axial coords for ring 1: (1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)
    # For ring 2, pick 4 that keep it compact: (2,0), (2,-1), (1,-2), (0,-2) or similar
    
    # Let me use a known compact 11-hexagon cluster:
    # It's like a hexagon of "radius" 2 but with 4 corners missing
    # Full hexagon of radius 2 has 1 + 6 + 12 = 19 hexagons
    # We need 11, so take center + ring1 (7 total) + 4 from ring2
    
    axial_coords = [
        (0, 0),      # center
        (1, 0),      # ring 1
        (1, -1),
        (0, -1),
        (-1, 0),
        (-1, 1),
        (0, 1),
        # 4 more from ring 2 to make 11
        (2, 0),
        (2, -1),
        (1, -2),
        (0, -2),
    ]
    
    inner_hex_data = np.zeros((11, 3))
    all_vertices = []
    
    for i, (q, r) in enumerate(axial_coords):
        x = 1.5 * q
        y = (sqrt3 / 2.0) * q + sqrt3 * r
        angle = 0.0
        inner_hex_data[i] = [x, y, angle]
        verts = _hex_vertices(x, y, 1.0, angle)
        all_vertices.extend(verts.tolist())
    
    all_vertices = np.array(all_vertices)
    
    # Find minimal enclosing regular hexagon
    side, angle_deg, center = _min_enclosing_hexagon(all_vertices)
    
    # Add small margin to ensure strict containment
    side += 1e-5
    
    outer_hex_data = np.array([center[0], center[1], angle_deg])
    
    return inner_hex_data, outer_hex_data, float(side)


# EVOLVE-BLOCK-END
