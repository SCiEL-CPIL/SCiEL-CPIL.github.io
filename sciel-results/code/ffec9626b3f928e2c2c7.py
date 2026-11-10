# EVOLVE-BLOCK-START
import numpy as np
import time


def _hex_vertices(cx, cy, d, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy), side d, rotated by angle_deg."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(60 * np.arange(6))
    xs = cx + d * np.cos(angles)
    ys = cy + d * np.sin(angles)
    return np.column_stack([xs, ys])


def _edge_normals(vertices):
    """Return unit outward normals for each edge of a polygon (vertices in CCW order)."""
    n = len(vertices)
    normals = np.zeros((n, 2))
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        # Outward normal for CCW polygon: rotate edge 90 deg clockwise
        normals[i] = np.array([edge[1], -edge[0]])
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normals /= norms
    return normals


def _sat_overlap(v1, v2, tol=1e-6):
    """
    Separating axis test: return True if two convex polygons overlap.
    v1, v2: (6,2) arrays of vertices in CCW order.
    """
    n1 = _edge_normals(v1)
    n2 = _edge_normals(v2)
    normals = np.vstack([n1, n2])
    for normal in normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        # If projections don't overlap (with tolerance), polygons are separated
        if max1 < min2 - tol or max2 < min1 - tol:
            return False
    return True


def _point_in_hexagon(v, vertices, tol=1e-6):
    """
    Check if point v is inside or on boundary of convex polygon (vertices CCW).
    For each edge (p1,p2), cross(p2-p1, v-p1) must be >= -tol.
    """
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        cross = (p2[0] - p1[0]) * (v[1] - p1[1]) - (p2[1] - p1[1]) * (v[0] - p1[0])
        if cross < -tol:
            return False
    return True


def _all_inner_in_outer(inner_data, outer_data):
    """Check all inner hexagon vertices are inside the outer hexagon."""
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], outer_data[2], 0)
    # outer_data[2] is side length, not angle; angle is 0
    for row in inner_data:
        cx, cy, ang = row[0], row[1], row[2]
        verts = _hex_vertices(cx, cy, 1.0, ang)
        for v in verts:
            if not _point_in_hexagon(v, outer_verts):
                return False
    return True


def _check_no_overlap(inner_data, tol=1e-6):
    """Check no two inner hexagons overlap using SAT."""
    n = len(inner_data)
    verts_list = []
    for row in inner_data:
        verts_list.append(_hex_vertices(row[0], row[1], 1.0, row[2]))
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_overlap(verts_list[i], verts_list[j], tol):
                return False
    return True


def _validate(inner_data, outer_data, s):
    """Full validation. Returns (valid, score) where score = 1/s if valid."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False, 0.0
    if s < np.sqrt(12) - 1e-10:
        return False, 0.0
    # Check outer vertices distinct
    outer_verts = _hex_vertices(outer_data[0], outer_data[1], s, outer_data[2])
    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-12:
                return False, 0.0
    # Check no overlap
    if not _check_no_overlap(inner_data):
        return False, 0.0
    # Check containment
    if not _all_inner_in_outer(inner_data, outer_data):
        return False, 0.0
    return True, 1.0 / s


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_margin = 30.0  # seconds

    # Initial configuration: symmetric hexagonal arrangement
    # Place 12 hexagons in a pattern that fits reasonably
    # Center + 6 around it + 6 more in a ring
    r = 2.0  # approximate distance between centers for unit hexagons
    h = np.sqrt(3) / 2  # height factor

    # Start with a compact arrangement
    # Hexagon centers in a triangular lattice pattern
    # For unit hexagons, center-to-center distance for touching is 2*sqrt(3)/2 = sqrt(3) ~ 1.732
    # But we need clearance, so use slightly larger

    # Let's use a pattern:
    # Row 0: 3 hexagons
    # Row 1: 3 hexagons  
    # Row 2: 3 hexagons
    # Row 3: 3 hexagons
    # Offset alternating rows

    dx = np.sqrt(3)  # horizontal spacing for touching hexagons
    dy = 3.0  # vertical spacing (hex height is 2, so centers 3 apart for touching? No...)
    # Actually for unit hexagons (side 1):
    # Width = 2, Height = sqrt(3)
    # For flat-top hexagons, horizontal center distance = 2*sqrt(3)/2 * 1 = sqrt(3) for touching
    # Vertical center distance = 3/2 * ... let me think more carefully

    # For flat-top regular hexagon with side 1:
    # The distance between centers of two touching flat-top hexagons horizontally is 2*sqrt(3)/2 = sqrt(3)
    # Vertically, if offset, it's 3/2 = 1.5... no.
    # Actually, the "radius" (center to vertex) is 1, "apothem" (center to edge) is sqrt(3)/2
    # Two flat-top hexagons side by side: centers separated by 2*apothem = sqrt(3) horizontally
    # Two flat-top hexagons with one above: vertical separation = 3*apothem = 3*sqrt(3)/2? No.
    # Let me just use a reasonable initial guess and let optimization fix it.

    # Simple initial: grid-like
    inner = np.array([
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        [-2.0, 0.0, 0.0],
        [1.0, 1.732, 0.0],
        [-1.0, 1.732, 0.0],
        [1.0, -1.732, 0.0],
        [-1.0, -1.732, 0.0],
        [3.0, 1.732, 0.0],
        [-3.0, 1.732, 0.0],
        [3.0, -1.732, 0.0],
        [-3.0, -1.732, 0.0],
        [0.0, 3.464, 0.0],
    ])

    outer = np.array([0.0, 0.0, 0.0])
    s = 8.0

    # Validate initial
    valid, score = _validate(inner, outer, s)
    best_inner = inner.copy()
    best_outer = outer.copy()
    best_s = s
    best_score = score

    if not valid:
        # Increase s until valid
        s = 10.0
        valid, score = _validate(inner, outer, s)
        best_inner = inner.copy()
        best_outer = outer.copy()
        best_s = s
        best_score = score

    # Simulated annealing
    np.random.seed(42)
    T = 0.5  # initial temperature
    T_min = 0.001
    cooling = 0.995
    max_steps = 50000

    for step in range(max_steps):
        if time.time() > deadline - safety_margin:
            break

        # Generate perturbation
        new_inner = best_inner.copy()
        new_outer = best_outer.copy()
        new_s = best_s

        # Perturb a random inner hexagon
        idx = np.random.randint(12)
        new_inner[idx, 0] += np.random.normal(0, T)
        new_inner[idx, 1] += np.random.normal(0, T)
        new_inner[idx, 2] += np.random.normal(0, T * 0.1)  # angle perturbation smaller

        # Perturb outer
        new_outer[0] += np.random.normal(0, T * 0.5)
        new_outer[1] += np.random.normal(0, T * 0.5)
        new_outer[2] += np.random.normal(0, T * 0.05)

        # Try to reduce s slightly
        new_s = best_s * (1.0 - np.random.exponential(0.01))
        new_s = max(new_s, np.sqrt(12))

        # Validate
        valid, new_score = _validate(new_inner, new_outer, new_s)

        if valid:
            if new_score > best_score:
                best_inner = new_inner.copy()
                best_outer = new_outer.copy()
                best_s = new_s
                best_score = new_score
            # Accept with some probability even if not better (to escape local minima)
            elif np.random.random() < np.exp((new_score - best_score) / T):
                best_inner = new_inner.copy()
                best_outer = new_outer.copy()
                best_s = new_s
                best_score = new_score
        else:
            # Invalid: reject, but maybe accept with very low prob to explore
            # Don't accept invalid states
            pass

        T *= cooling
        if T < T_min:
            T = T_min

    # Final polish: try to reduce s further with small perturbations
    for polish_step in range(2000):
        if time.time() > deadline - safety_margin:
            break
        new_s = best_s * (1.0 - np.random.exponential(0.005))
        new_s = max(new_s, np.sqrt(12))
        valid, new_score = _validate(best_inner, best_outer, new_s)
        if valid and new_score > best_score:
            best_s = new_s
            best_score = new_score

    return best_inner, best_outer, best_s


# EVOLVE-BLOCK-END
