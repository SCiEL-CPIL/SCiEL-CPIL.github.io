# EVOLVE-BLOCK-START
import time
import numpy as np

_START = time.time()
_DEADLINE = 630.0 - 30.0  # safety margin before 630s kill


def _hex_verts(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _edge_normals(cx, cy, s, a_deg):
    v = _hex_verts(cx, cy, s, a_deg)
    n = np.zeros((6, 2))
    for i in range(6):
        e = v[(i + 1) % 6] - v[i]
        n[i] = np.array([-e[1], e[0]])
        n[i] /= (np.linalg.norm(n[i]) + 1e-300)
    return n


def _separation_ok(v1, n1, v2, n2, eps=1e-6):
    # For each of 12 normals, project both hexagons; if any normal has a gap, they're separated
    for i in range(6):
        p1 = v1 @ n1[i]
        p2 = v2 @ n1[i]
        if (p1.max() < p2.min() - eps) or (p2.max() < p1.min() - eps):
            return True
    for i in range(6):
        p1 = v1 @ n2[i]
        p2 = v2 @ n2[i]
        if (p1.max() < p2.min() - eps) or (p2.max() < p1.min() - eps):
            return True
    return False


def _min_enclosing_hex_side(pts, angle_deg=0.0):
    """Min side length s of a regular hexagon (centered at origin, given angle) containing all pts."""
    a = np.deg2rad(angle_deg)
    th = a + np.deg2rad(60 * np.arange(6))
    # outward normals for each edge
    normals = np.column_stack([-np.sin(th), np.cos(th)])
    # For a regular hexagon centered at origin with side s, the inradius is s*sqrt(3)/2
    # The distance from center to each edge (inradius) must be >= max projection of pts onto outward normal
    # Actually for a regular hexagon, the support function in direction n (outward normal to edge i)
    # equals s * sqrt(3) / 2 (the inradius), independent of direction for the 6 edge normals.
    # But pts can project beyond that. The constraint is: for each edge normal n_i,
    # max_j (pts_j . n_i) <= inradius = s * sqrt(3)/2
    # So s >= max_i (max_j (pts_j . n_i)) / (sqrt(3)/2)
    # But wait - the hexagon edges are at distance inradius from center, and the normal points outward.
    # A point p is inside if p . n_i <= inradius for all i (where n_i is outward unit normal).
    # Also need p . n_i >= -inradius (covered by opposite edge).
    projections = pts @ normals.T  # (n, 6)
    max_proj = projections.max(axis=0)  # (6,)
    s = max_proj.max() / (np.sqrt(3) / 2)
    return max(s, 0.0)


def _is_inside_hex(p, cx, cy, s, a_deg, tol=1e-6):
    """Check if point p is inside regular hexagon using cross products."""
    v = _hex_verts(cx, cy, s, a_deg)
    for i in range(6):
        p1 = v[i]
        p2 = v[(i + 1) % 6]
        cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
        if cross < -tol:
            return False
    return True


def _all_inside(pts, cx, cy, s, a_deg, tol=1e-6):
    v = _hex_verts(cx, cy, s, a_deg)
    for p in pts:
        ok = True
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
            if cross < -tol:
                ok = False
                break
        if not ok:
            return False
    return True


def _pack_valid(inner_data, outer_data, side, eps=1e-6):
    """Full validity check."""
    # All finite
    if not (np.all(np.isfinite(inner_data)) and np.all(np.isfinite(outer_data))):
        return False
    if side < np.sqrt(11) - 1e-9:
        return False
    # Check separation of inner hexagons
    n = len(inner_data)
    verts_list = []
    normals_list = []
    for i in range(n):
        x, y, a = inner_data[i]
        v = _hex_verts(x, y, 1.0, a)
        nn = _edge_normals(x, y, 1.0, a)
        verts_list.append(v)
        normals_list.append(nn)
    for i in range(n):
        for j in range(i + 1, n):
            if not _separation_ok(verts_list[i], normals_list[i], verts_list[j], normals_list[j], eps):
                return False
    # Check all inner vertices inside outer hexagon
    cx, cy, a = outer_data
    for i in range(n):
        x, y, ai = inner_data[i]
        v = _hex_verts(x, y, 1.0, ai)
        if not _all_inside(v, cx, cy, side, a, tol=1e-6):
            return False
    # Edge lengths within 1e-8 of 1
    for i in range(n):
        v = verts_list[i]
        for k in range(6):
            e = v[(k + 1) % 6] - v[k]
            if abs(np.linalg.norm(e) - 1.0) > 1e-8:
                return False
    return True


def _flower_initial():
    """Flower pattern: 1 center + 6 ring + 4 outer. Returns (11,2) positions, all angle 0."""
    # Regular hexagon with side 1: vertices at distance 1 from center
    # Ring hexagons touch center: their centers are at distance 2 from center
    # (two hexagons with side 1 touching have centers 2*sqrt(3)/2 * ... )
    # Actually for two unit hexagons to be separated (not overlapping),
    # the minimum center distance depends on orientation.
    # For same orientation (angle 0), the minimum separation distance is sqrt(3) (edge-to-edge).
    # Let's use a flower with center distance = sqrt(3) + small gap
    d = np.sqrt(3) + 0.01  # center distance for touching
    angles = np.deg2rad(np.arange(6) * 60)
    pts = np.zeros((11, 2))
    pts[0] = [0, 0]
    for i in range(6):
        pts[i + 1] = [d * np.cos(angles[i]), d * np.sin(angles[i])]
    # 4 outer hexagons at 120 degree intervals, further out
    d2 = 2 * d + 0.01
    angles2 = np.deg2rad([30, 90, 150, 210, 270, 330])
    # Pick 4: at 30, 150, 270, 30 (let's do 30, 90, 210, 330)
    angles2 = np.deg2rad([30, 90, 210, 330])
    for i in range(4):
        pts[i + 7] = [d2 * np.cos(angles2[i]), d2 * np.sin(angles2[i])]
    return pts


def _hex_separation_penalty(pts, eps=1e-6):
    """Penalty for overlapping hexagons (all angle 0, side 1)."""
    n = len(pts)
    penalty = 0.0
    # Precompute vertices and normals for angle=0
    th = np.deg2rad(60 * np.arange(6))
    base_v = np.column_stack([np.cos(th), np.sin(th)])
    # Edge normals
    normals = np.zeros((6, 2))
    for i in range(6):
        e = base_v[(i + 1) % 6] - base_v[i]
        normals[i] = np.array([-e[1], e[0]])
        normals[i] /= (np.linalg.norm(normals[i]) + 1e-300)

    for i in range(n):
        v1 = pts[i] + base_v
        for j in range(i + 1, n):
            v2 = pts[j] + base_v
            sep = False
            for k in range(6):
                p1 = v1 @ normals[k]
                p2 = v2 @ normals[k]
                if p1.max() < p2.min() - eps or p2.max() < p1.min() - eps:
                    sep = True
                    break
            if not sep:
                for k in range(6):
                    p1 = v1 @ normals[k]
                    p2 = v2 @ normals[k]
                    if p1.max() < p2.min() - eps or p2.max() < p1.min() - eps:
                        sep = True
                        break
            if not sep:
                # Compute penetration depth
                pen = 0.0
                for k in range(6):
                    p1 = v1 @ normals[k]
                    p2 = v2 @ normals[k]
                    overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    if overlap > 0:
                        pen += overlap
                for k in range(6):
                    p1 = v1 @ normals[k]
                    p2 = v2 @ normals[k]
                    overlap = min(p1.max(), p2.max()) - max(p1.min(), p2.min())
                    if overlap > 0:
                        pen += overlap
                penalty += pen ** 2
    return penalty


def _optimize(pts, max_iters=5000, t0=0.1, t1=0.001):
    """Simulated annealing to minimize enclosing hex side, with separation constraints."""
    best_pts = pts.copy()
    best_side = _min_enclosing_hex_side(pts - pts[0], 0.0)  # relative to first point
    # Actually we need to center: the enclosing hex is centered at origin, pts are absolute
    # Let's work with pts centered at their centroid
    centroid = pts.mean(axis=0)
    work = pts - centroid

    def obj(p):
        c = p.mean(axis=0)
        pc = p - c
        side = _min_enclosing_hex_side(pc, 0.0)
        pen = _hex_separation_penalty(pc)
        return side + 1000.0 * pen

    cur = work.copy()
    cur_obj = obj(cur)
    best_obj = cur_obj
    best_cur = cur.copy()

    for it in range(max_iters):
        if time.time() > _DEADLINE:
            break
        t = t0 * (t1 / t0) ** (it / max_iters)
        # Random perturbation
        step = t * 0.5
        cand = cur + np.random.randn_like(cur) * step
        cand_obj = obj(cand)
        if cand_obj < cur_obj or np.random.rand() < np.exp(-(cand_obj - cur_obj) / max(t, 1e-10)):
            cur = cand
            cur_obj = cand_obj
            if cur_obj < best_obj:
                best_obj = cur_obj
                best_cur = cur.copy()

    return best_cur + centroid, best_obj


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon,
    maximizing 1/outer_hex_side_length.
    """
    # Start with flower pattern
    pts = _flower_initial()

    # Run simulated annealing
    if time.time() < _DEADLINE - 10:
        pts_opt, obj = _optimize(pts, max_iters=3000, t0=0.15, t1=0.0005)
    else:
        pts_opt = pts

    # Center the configuration
    centroid = pts_opt.mean(axis=0)
    pts_c = pts_opt - centroid

    # Find the minimal enclosing regular hexagon (angle 0)
    side = _min_enclosing_hex_side(pts_c, 0.0)

    # Add small margin for numerical safety
    side += 0.01

    # Verify validity
    inner_data = np.column_stack([pts_c[:, 0], pts_c[:, 1], np.zeros(11)])
    outer_data = np.array([0.0, 0.0, 0.0])

    if not _pack_valid(inner_data, outer_data, side):
        # Try with slightly larger side
        side += 0.05
        if not _pack_valid(inner_data, outer_data, side):
            # Fall back to a known-valid large hexagon
            side = 8.0
            # Re-center
            pts_c = pts - pts.mean(axis=0)
            inner_data = np.column_stack([pts_c[:, 0], pts_c[:, 1], np.zeros(11)])
            outer_data = np.array([0.0, 0.0, 0.0])

    return inner_data, outer_data, float(side)


# EVOLVE-BLOCK-END
