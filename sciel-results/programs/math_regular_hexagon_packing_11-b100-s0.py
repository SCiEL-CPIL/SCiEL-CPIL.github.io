# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hex_vertices(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    ang = a + np.deg2rad(60 * np.arange(6))
    return np.stack([cx + s * np.cos(ang), cy + s * np.sin(ang)], axis=1)


def _hex_normals(a_deg):
    a = np.deg2rad(a_deg)
    # 12 edge normals: 6 edge directions, each with + and -
    base = a + np.deg2rad(60 * np.arange(6))
    # edge normal is perpendicular to edge; for a hex with vertex angles base,
    # the edges are between consecutive vertices; normal direction = base + 30deg
    normals = base + np.deg2rad(30)
    dirs = np.stack([np.cos(normals), np.sin(normals)], axis=1)
    # both directions
    return np.vstack([dirs, -dirs])


def _separation_ok(v1, n1, v2, n2):
    """Check if two hexagons (vertices, normals) are separated on at least one normal."""
    # For each of 12 normals of hex1, project both hexes; if intervals are apart by > 1e-6, separated.
    for d in n1:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)  # hex1 entirely below hex2
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)  # hex2 entirely below hex1
        if gap > 1e-6:
            return True
    for d in n2:
        p1 = v1 @ d
        p2 = v2 @ d
        gap = min(p1) - max(p2)
        if gap > 1e-6:
            return True
        gap = min(p2) - max(p1)
        if gap > 1e-6:
            return True
    return False


def _hexagons_separated(inner_data):
    """Check all pairs of inner hexagons are separated. inner_data: (11,3) (x,y,angle_deg)."""
    verts = []
    norms = []
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        n = _hex_normals(a)
        verts.append(v)
        norms.append(n)
    for i in range(len(inner_data)):
        for j in range(i + 1, len(inner_data)):
            if not _separation_ok(verts[i], norms[i], verts[j], norms[j]):
                return False
    return True


def _contains_in_hex(inner_data, outer_data, side):
    """Check all inner hex vertices are inside or on the outer hexagon."""
    ox, oy, oa = outer_data
    outer_v = _hex_vertices(ox, oy, side, oa)
    # For each inner vertex, check it's to the left of every outer edge (CCW)
    # Cross product test: for edge p1->p2, point p should have cross(p2-p1, p-p1) >= -tol
    tol = -1e-6
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            for e in range(6):
                p1 = outer_v[e]
                p2 = outer_v[(e + 1) % 6]
                cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                if cross < tol:
                    return False
    return True


def _valid(inner_data, outer_data, side):
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if side < np.sqrt(11):
        return False
    if not _hexagons_separated(inner_data):
        return False
    if not _contains_in_hex(inner_data, outer_data, side):
        return False
    # Check edge lengths within 1e-8 of 1
    for i in range(len(inner_data)):
        x, y, a = inner_data[i]
        v = _hex_vertices(x, y, 1.0, a)
        for e in range(6):
            p1 = v[e]
            p2 = v[(e + 1) % 6]
            if abs(np.linalg.norm(p2 - p1) - 1.0) > 1e-8:
                return False
    return True


def _objective(params):
    """
    Symmetry-reduced parameterization.
    We use 6-fold symmetry: place hexagons in orbits under the hexagonal symmetry group.
    
    Layout: 1 center + 6 in first ring + 4 in second ring (using symmetry).
    Actually, let's use a more flexible approach:
    - Center hexagon at origin
    - 6 hexagons in a ring at distance r1, angles 0,60,120,180,240,300 (with possible individual angle offsets)
    - 4 more hexagons placed symmetrically
    
    For simplicity and robustness, let's parameterize:
    params = [r1, r2, angle_offset, r3, angle3_offset, ...]
    
    Let me use a different approach: exploit 6-fold rotational symmetry.
    11 = 1 + 6 + 4. The 4 in the outer ring can be placed at angles 30, 150, 210, 330 (i.e., 30 + 60k for k=0,1,3,4) 
    or 0,60,240,300 etc. Let's try 30, 90, 210, 270? No, that's 4-fold.
    
    Better: 1 center, 6 at radius r1 at angles 0,60,...,300, and 4 at radius r2 at angles 30,150,210,330.
    All hexagons can have the same orientation (angle=0) for simplicity, or we allow a global rotation.
    
    params = [r1, r2, global_angle]
    Then the outer hexagon side is computed as the minimum side that contains all.
    """
    r1 = params[0]
    r2 = params[1]
    gangle = params[2]  # global rotation in degrees
    
    # Build inner hexagons
    inner = []
    # Center
    inner.append([0.0, 0.0, gangle])
    # First ring: 6 hexagons
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    # Second ring: 4 hexagons at 30, 150, 210, 330 degrees
    for k in [0, 2, 3, 5]:  # 30, 150, 210, 330
        ang = np.deg2rad(60 * k + 30)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, gangle])
    
    inner = np.array(inner)
    
    # Compute minimum enclosing hexagon side
    # The outer hexagon is centered at origin with angle 0 (or gangle).
    # For a regular hexagon centered at origin with side s and angle a,
    # the apothem (distance from center to edge midpoint) is s * sqrt(3)/2.
    # A point (x,y) is inside if its projection onto each of the 6 edge normals <= apothem.
    # Edge normals are at angles a + 30 + 60k.
    
    # To find minimum s such that all inner vertices are inside:
    # For each inner vertex, compute the maximum over the 6 normals of (dot(vertex, normal) / apothem_per_unit_side)
    # Actually, for a hex with side s, the constraint for point p is:
    #   |p · n_k| <= s * sqrt(3)/2  for each of the 6 normals n_k (unit vectors)
    # So s >= max over all vertices and all k of |p · n_k| / (sqrt(3)/2)
    
    normals = []
    a_rad = np.deg2rad(gangle)
    for k in range(6):
        ang = a_rad + np.deg2rad(30 + 60 * k)
        normals.append([np.cos(ang), np.sin(ang)])
    normals = np.array(normals)
    
    max_proj = 0.0
    for i in range(len(inner)):
        x, y, a = inner[i]
        v = _hex_vertices(x, y, 1.0, a)
        for p in v:
            projs = np.abs(v @ normals.T)  # shape (6,)
            max_proj = max(max_proj, np.max(projs))
    
    side = max_proj / (np.sqrt(3) / 2)
    
    # Check validity
    outer = np.array([0.0, 0.0, gangle])
    if not _valid(inner, outer, side):
        return side + 1000.0  # penalty
    
    return side


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    deadline = time.time() + 600.0
    
    best_side = float('inf')
    best_params = None
    
    # Initial guesses
    initial_guesses = [
        [2.5, 4.5, 0.0],
        [2.5, 5.0, 30.0],
        [3.0, 5.0, 0.0],
        [2.5, 4.0, 15.0],
        [2.0, 4.5, 0.0],
        [2.5, 4.5, 60.0],
    ]
    
    for guess in initial_guesses:
        if time.time() > deadline - 60:
            break
        try:
            res = minimize(
                _objective,
                guess,
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
            )
            if res.fun < best_side:
                best_side = res.fun
                best_params = res.x
        except Exception:
            continue
    
    if best_params is None:
        # Fallback to the seed arrangement
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
        return inner_hex_data, outer_hex_data, outer_hex_side_length
    
    r1, r2, gangle = best_params
    
    # Build final inner hexagons
    inner = []
    inner.append([0.0, 0.0, gangle])
    for k in range(6):
        ang = np.deg2rad(60 * k)
        x = r1 * np.cos(ang)
        y = r1 * np.sin(ang)
        inner.append([x, y, gangle])
    for k in [0, 2, 3, 5]:
        ang = np.deg2rad(60 * k + 30)
        x = r2 * np.cos(ang)
        y = r2 * np.sin(ang)
        inner.append([x, y, gangle])
    
    inner_hex_data = np.array(inner)
    outer_hex_data = np.array([0.0, 0.0, gangle])
    outer_hex_side_length = best_side
    
    # Final validation
    if not _valid(inner_hex_data, outer_hex_data, outer_hex_side_length):
        # Fallback
        inner_hex_data = np.array(
            [
                [0, 0, 0],
                [-2.5, 0, 0],
                [2.5, 0, 0],
                [-1.25, 2.17, 0],
                [1.25, 2.17, 0],
                [-1.25, -2.17, 0],
                [1.25, -2.17, 0],
                [-3.75, 2.17, 0],
                [3.75, 2.17, 0],
                [-3.75, -2.17, 0],
                [3.75, -2.17, 0],
            ]
        )
        outer_hex_data = np.array([0, 0, 0])
        outer_hex_side_length = 8.0
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END
