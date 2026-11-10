# EVOLVE-BLOCK-START
import numpy as np
import time

_START = time.time()
_DEADLINE = 600.0  # seconds


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.stack([cx + s * np.cos(th), cy + s * np.sin(th)], axis=1)


def _edge_normals(cx, cy, s, a_deg):
    """Return 12 outward edge normals (unit vectors) for a unit hexagon."""
    V = _hex_vertices(cx, cy, s, a_deg)
    normals = []
    for i in range(6):
        p1 = V[i]
        p2 = V[(i + 1) % 6]
        e = p2 - p1
        L = np.linalg.norm(e)
        if L < 1e-12:
            continue
        # outward normal: rotate e by -90 deg (since vertices CCW, outward is right side)
        n = np.array([e[1], -e[0]])
        n = n / np.linalg.norm(n)
        normals.append(n)
    normals = np.array(normals)
    # add inward normals
    return np.vstack([normals, -normals])


def _min_gap(h1, h2):
    """h1, h2 each (cx, cy, s, a_deg). Return minimum positive separation along any of the 12 normals of either hexagon, or 0 if intersecting."""
    n1 = _edge_normals(h1[0], h1[1], h1[2], h1[3])
    n2 = _edge_normals(h2[0], h2[1], h2[2], h2[3])
    V1 = _hex_vertices(h1[0], h1[1], h1[2], h1[3])
    V2 = _hex_vertices(h2[0], h2[1], h2[2], h2[3])
    proj1 = V1 @ n1.T  # (6, 12)
    proj2 = V2 @ n2.T  # (6, 12)
    # For each normal of hex1, check against hex2 vertices
    # We need: for normal n (unit), the gap = min over the two hexagons' projections
    # Actually the standard: gap along normal n = (max projection of A on n) - (min projection of B on n) if A is on + side
    # Better: for each normal n (from either set), compute:
    #   projA = V1 @ n, projB = V2 @ n
    #   gap = min( (max projA - min projB), (max projB - min projA) )
    # but only if the two intervals are separated, i.e., the gap > 0 means separated.
    # If gap <= 0 for all normals, they intersect.
    # The minimum gap over all normals = the actual separation distance (if separated).
    all_normals = np.vstack([n1, n2])
    best = 0.0
    for n in all_normals:
        pA = V1 @ n
        pB = V2 @ n
        gap = min(max(pA) - min(pB), max(pB) - min(pA))
        if gap > best:
            best = gap
    return best


def _pairwise_separated(inner, tol=1e-6):
    """inner: (11, 3) rows (x, y, angle_deg). Check all pairs separated by > tol on some normal."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            h1 = (inner[i, 0], inner[i, 1], 1.0, inner[i, 2])
            h2 = (inner[j, 0], inner[j, 1], 1.0, inner[j, 2])
            g = _min_gap(h1, h2)
            if g <= tol:
                return False
    return True


def _outer_vertices(cx, cy, s, a_deg):
    return _hex_vertices(cx, cy, s, a_deg)


def _required_side(inner, outer_angle_deg):
    """Given inner hexagons (unit side) and outer hexagon angle (centered at origin),
    find the minimum side length S of outer hexagon (centered at origin) that contains all inner vertices."""
    # For a regular hexagon centered at origin with side S and angle a,
    # the distance from center to each edge (inradius) = S * sqrt(3)/2
    # The distance from center to each vertex (circumradius) = S
    # A point p is inside iff for all 6 outward edge normals n_i: p . n_i <= inradius
    # So we need S such that S * sqrt(3)/2 >= max over all inner vertices p and all 6 normals n_i of (p . n_i)
    a = np.deg2rad(outer_angle_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    # outward normals: for vertex i to i+1, outward normal direction
    V = np.stack([np.cos(th), np.sin(th)], axis=1)  # unit circumradius vertices
    normals = []
    for i in range(6):
        p1 = V[i]
        p2 = V[(i + 1) % 6]
        e = p2 - p1
        n = np.array([e[1], -e[0]])
        n = n / np.linalg.norm(n)
        normals.append(n)
    normals = np.array(normals)  # (6, 2)
    # inradius for unit circumradius hexagon = sqrt(3)/2
    inradius_unit = np.sqrt(3.0) / 2.0
    # Collect all inner vertices
    pts = []
    for row in inner:
        Vh = _hex_vertices(row[0], row[1], 1.0, row[2])
        pts.append(Vh)
    pts = np.vstack(pts)  # (66, 2)
    # For each normal, max dot
    dots = pts @ normals.T  # (66, 6)
    max_dot = dots.max(axis=0)  # (6,)
    max_dot = max(max_dot)
    # Need S * inradius_unit >= max_dot  =>  S >= max_dot / inradius_unit
    S = max_dot / inradius_unit
    return S


def _validate(inner, outer, side, tol=1e-6):
    """Full validity check per task contract."""
    if not np.all(np.isfinite(inner)) or not np.all(np.isfinite(outer)):
        return False
    if side < np.sqrt(11.0) - 1e-12:
        return False
    # pairwise separation
    if not _pairwise_separated(inner, tol):
        return False
    # containment: every inner vertex inside outer hexagon
    ocx, ocy, oa = outer
    Vout = _outer_vertices(ocx, ocy, side, oa)
    # shift inner by -outer center
    inner_shifted = inner.copy()
    inner_shifted[:, 0] -= ocx
    inner_shifted[:, 1] -= ocy
    # For each outer edge p1->p2, cross(p2-p1, p-p1) >= -tol for all inner vertices p
    for i in range(6):
        p1 = Vout[i]
        p2 = Vout[(i + 1) % 6]
        e = p2 - p1
        for row in inner_shifted:
            Vh = _hex_vertices(row[0], row[1], 1.0, row[2])
            for p in Vh:
                cross = e[0] * (p[1] - p1[1]) - e[1] * (p[0] - p1[0])
                if cross < -tol:
                    return False
    # edge length check: shifted inner hexagons keep edge lengths within 1e-8 of 1
    for row in inner_shifted:
        Vh = _hex_vertices(row[0], row[1], 1.0, row[2])
        for i in range(6):
            L = np.linalg.norm(Vh[(i + 1) % 6] - Vh[i])
            if abs(L - 1.0) > 1e-8:
                return False
    return True


def _build_honeycomb():
    """Build the natural honeycomb cluster of 11 unit hexagons.
    Hexagons on a honeycomb lattice: neighbors at distance 2*apothem = sqrt(3) in 6 directions.
    Center at origin. The 6 neighbors at angles 0, 60, 120, 180, 240, 300 degrees, distance sqrt(3).
    Then 4 more: the 'corner' hexagons. In a 7-hex cluster (center + 6), the corners are at
    distance 2*sqrt(3) at angles 30, 90, 150, 210, 270, 330? No, that's 6.
    
    Actually the 11-hex cluster: center + 6 ring + 4 of the 6 second-ring positions.
    The second ring in honeycomb: 12 positions. The 6 'corner' ones (at 30,90,150,210,270,330 from center, distance 2*sqrt(3))
    and 6 'edge' ones (at 0,60,120,180,240,300, distance 2*sqrt(3)).
    
    For 11, we take center + 6 first-ring + 4 second-ring. Which 4? The symmetric choice: 
    take 4 of the 6 corner positions. But that breaks 6-fold symmetry.
    
    Alternatively, the known optimal 11-hex packing in a hexagon: it's the 7-hex cluster plus 4 more.
    Let me just use the 7-hex cluster + 4 corner hexagons at angles 0, 120, 240, 360? No.
    
    Let me use a different approach: place 11 hexagons in a roughly hexagonal shape.
    Row structure (like hex numbers): 
    Row 0 (top): 1 hex
    Row 1: 2 hex
    Row 2: 3 hex
    Row 3: 3 hex
    Row 4: 2 hex
    Row 5: 1 hex  -- that's 12.
    
    For 11: 
    Row 0: 1
    Row 1: 2
    Row 2: 3
    Row 3: 3
    Row 4: 1  -- that's 10.
    
    Let me just use the honeycomb 7 + 4 corners at 0, 120, 240, and one more.
    Actually, let me use a simpler approach: the 11 hexagons in a compact arrangement.
    
    I'll use the 7-hex cluster and add 4 hexagons at the 4 'corner' positions of a larger hexagon.
    """
    d = np.sqrt(3.0)  # distance between centers of touching hexagons
    hexes = []
    # center
    hexes.append([0.0, 0.0, 0.0])
    # 6 neighbors
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        hexes.append([d * np.cos(ang), d * np.sin(ang), 0.0])
    # 4 more: at distance 2*d at angles 30, 90, 150, 210 (4 of 6 corner positions)
    # Actually for symmetry, let me use angles 0, 120, 240, 360? No, 360=0.
    # Let me use the 4 corner positions at 30, 150, 270, 30? That's only 3 distinct.
    # 6 corner positions at 30, 90, 150, 210, 270, 330.
    # Take 4: 30, 90, 210, 270 (opposite pairs) for symmetry.
    for ang_deg in [30.0, 90.0, 210.0, 270.0]:
        ang = np.deg2rad(ang_deg)
        hexes.append([2 * d * np.cos(ang), 2 * d * np.sin(ang), 0.0])
    return np.array(hexes)


def _optimize_outer_rotation(inner, n_angles=360):
    """Try many outer hexagon rotations, return best (angle, side)."""
    best_side = float('inf')
    best_angle = 0.0
    for k in range(n_angles):
        a = 360.0 * k / n_angles
        S = _required_side(inner, a)
        if S < best_side:
            best_side = S
            best_angle = a
    # refine around best
    a0 = best_angle
    for k in range(3600):
        a = a0 - 0.5 + 360.0 * k / 3600
        S = _required_side(inner, a)
        if S < best_side:
            best_side = S
            best_angle = a
    return best_angle, best_side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon.
    Returns
        inner_hex_data: np.ndarray of shape (11,3)
        outer_hex_data: np.ndarray of shape (3,)
        outer_hex_side_length: float
    """
    # Build honeycomb cluster
    inner = _build_honeycomb()
    
    # Optimize outer hexagon rotation
    best_angle, best_side = _optimize_outer_rotation(inner, n_angles=720)
    
    outer = np.array([0.0, 0.0, best_angle])
    
    # Validate
    if not _validate(inner, outer, best_side):
        # Fallback: use a large enough outer hexagon
        best_side = 10.0
        outer = np.array([0.0, 0.0, 0.0])
    
    return inner, outer, float(best_side)


# EVOLVE-BLOCK-END
