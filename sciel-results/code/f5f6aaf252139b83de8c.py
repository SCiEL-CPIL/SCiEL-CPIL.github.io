# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hexagon_vertices(cx, cy, d, angle_deg):
    """Compute vertices of a regular hexagon with center (cx,cy), side d, angle in degrees."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3
    return np.column_stack([cx + d * np.cos(angles), cy + d * np.sin(angles)])


def _separating_axis_overlap(v1, v2):
    """Check if two convex polygons overlap using separating axis theorem.
    Returns True if they overlap (bad), False if separated (good)."""
    n1 = len(v1)
    n2 = len(v2)
    # Test all edges of both polygons
    for i in range(n1):
        p1 = v1[i]
        p2 = v1[(i + 1) % n1]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        if proj1.max() < proj2.min() - 1e-8 or proj2.max() < proj1.min() - 1e-8:
            return False
    for i in range(n2):
        p1 = v2[i]
        p2 = v2[(i + 1) % n2]
        edge = p2 - p1
        normal = np.array([edge[1], -edge[0]])
        norm = np.linalg.norm(normal)
        if norm == 0:
            continue
        normal /= norm
        proj1 = np.dot(v1, normal)
        proj2 = np.dot(v2, normal)
        if proj1.max() < proj2.min() - 1e-8 or proj2.max() < proj1.min() - 1e-8:
            return False
    return True


def _point_in_hexagon(v, outer_verts):
    """Check if point v is inside or on boundary of convex polygon outer_verts (CCW)."""
    n = len(outer_verts)
    for i in range(n):
        p1 = outer_verts[i]
        p2 = outer_verts[(i + 1) % n]
        edge = p2 - p1
        cross = edge[0] * (v[1] - p1[1]) - edge[1] * (v[0] - p1[0])
        if cross < -1e-6:
            return False
    return True


def _validate_result(inner_data, outer_data, s):
    """Validate the packing result. Returns True if valid."""
    if not np.all(np.isfinite(inner_data)) or not np.all(np.isfinite(outer_data)):
        return False
    if s < np.sqrt(12) - 1e-9:
        return False

    outer_verts = _hexagon_vertices(outer_data[0], outer_data[1], s, outer_data[2])

    for i in range(6):
        for j in range(i + 1, 6):
            if np.linalg.norm(outer_verts[i] - outer_verts[j]) < 1e-6:
                return False

    inner_verts_list = []
    for i in range(12):
        x, y, a = inner_data[i]
        verts = _hexagon_vertices(x, y, 1.0, a)
        inner_verts_list.append(verts)

    for i in range(12):
        verts = inner_verts_list[i]
        for k in range(6):
            d = np.linalg.norm(verts[(k + 1) % 6] - verts[k])
            if abs(d - 1.0) > 1e-8:
                return False

    for i in range(12):
        for j in range(i + 1, 12):
            if _separating_axis_overlap(inner_verts_list[i], inner_verts_list[j]):
                return False

    for i in range(12):
        for v in inner_verts_list[i]:
            if not _point_in_hexagon(v, outer_verts):
                return False

    return True


def _packing_cost(params, deadline):
    """
    Cost function: minimize s + penalties.
    params[0] = s
    params[1:] = 12*(x, y) for inner centers (angles fixed at 0)
    """
    if time.time() > deadline:
        return 1e10
    s = params[0]
    inner_xy = params[1:].reshape(12, 2)
    penalty = 0.0

    # Outer hexagon centered at origin, angle 0
    outer_verts = _hexagon_vertices(0, 0, s, 0)

    # Inner hexagons (all angle 0)
    inner_verts_list = []
    for i in range(12):
        x, y = inner_xy[i]
        verts = _hexagon_vertices(x, y, 1.0, 0)
        inner_verts_list.append(verts)

    # Check overlap between inner hexagons
    for i in range(12):
        for j in range(i + 1, 12):
            if _separating_axis_overlap(inner_verts_list[i], inner_verts_list[j]):
                penalty += 5.0

    # Check inner vertices inside outer
    for i in range(12):
        for v in inner_verts_list[i]:
            if not _point_in_hexagon(v, outer_verts):
                penalty += 5.0

    return s + penalty


def _honeycomb_initial():
    """Create a 3-4-3 honeycomb arrangement of 12 hexagons."""
    dy = 1.5 * np.sqrt(3)  # vertical spacing
    inner = np.zeros((12, 3))
    idx = 0
    # Row 0 (y=0): 3 hexagons
    for i in range(3):
        x = (i - 1) * 2.0
        inner[idx] = [x, 0, 0]
        idx += 1
    # Row 1 (y=dy): 4 hexagons
    for i in range(4):
        x = (i - 1.5) * 2.0
        inner[idx] = [x, dy, 0]
        idx += 1
    # Row 2 (y=2*dy): 3 hexagons
    for i in range(3):
        x = (i - 1) * 2.0
        inner[idx] = [x, 2 * dy, 0]
        idx += 1
    return inner


def hexagon_packing_12():
    """
    Pack 12 unit regular hexagons in a regular hexagon, minimizing outer side length.
    Returns (inner, outer, s) where inner is (12,3), outer is (3,), s is float.
    """
    start_time = time.time()
    deadline = start_time + 580  # safety margin

    # Start with honeycomb arrangement
    inner_init = _honeycomb_initial()

    # Compute initial s: find minimum s that contains all inner hexagons
    def _min_s_for_inner(inner_xy):
        """Find minimum s such that all inner hexagons fit in outer hexagon centered at origin."""
        # For a regular hexagon with side s centered at origin, angle 0:
        # The apothem (distance from center to edge) is s * sqrt(3)/2
        # Vertices are at distance s from center
        # Check each inner vertex against the outer hexagon
        # Binary search on s
        lo = np.sqrt(12)
        hi = 20.0
        for _ in range(60):
            mid = (lo + hi) / 2
            outer_verts = _hexagon_vertices(0, 0, mid, 0)
            ok = True
            for i in range(12):
                x, y = inner_xy[i]
                verts = _hexagon_vertices(x, y, 1.0, 0)
                for v in verts:
                    if not _point_in_hexagon(v, outer_verts):
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                hi = mid
            else:
                lo = mid
        return hi

    s_init = _min_s_for_inner(inner_init[:, :2])

    # Optimize: minimize s + penalties
    # Parameters: [s, x0, y0, x1, y1, ..., x11, y11]
    x0 = np.concatenate([[s_init], inner_init[:, :2].flatten()])

    def cost(p):
        return _packing_cost(p, deadline)

    result = None
    try:
        result = minimize(cost, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8,
                                   'adaptive': True})
    except Exception:
        pass

    best_s = s_init
    best_inner = inner_init.copy()

    if result is not None and result.fun < best_s:
        best_s = result.x[0]
        best_inner_xy = result.x[1:].reshape(12, 2)
        best_inner = np.column_stack([best_inner_xy, np.zeros(12)])

    # Validate and if invalid, fall back to a known-good large hexagon
    outer_data = np.array([0.0, 0.0, 0.0])
    if _validate_result(best_inner, outer_data, best_s):
        return best_inner, outer_data, best_s

    # Fallback: use a large enough hexagon
    # Find a valid s by expanding
    s = s_init
    for _ in range(50):
        outer_data = np.array([0.0, 0.0, 0.0])
        if _validate_result(inner_init, outer_data, s):
            return inner_init, outer_data, s
        s *= 1.05

    # Last resort: very large hexagon
    s = 15.0
    outer_data = np.array([0.0, 0.0, 0.0])
    return inner_init, outer_data, s


# EVOLVE-BLOCK-END
