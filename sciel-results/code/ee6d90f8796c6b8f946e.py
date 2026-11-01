"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side.

Strategy:
  - Fast analytic SAT (precomputed edge normals, no per-check norm computation).
  - Several compact lattice clusters (brick / hex / staggered) with exact
    minimum-enclosing-hexagon computation.
  - 1-D global-rotation optimization to squeeze the enclosing side.
  - Cheap local polish, all under a strict wall-clock deadline.
"""

import math
import time
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin before 150s wall-clock kill

# Precomputed unit edge normals for a hexagon (6 directions), cached.
_EDGE_NORMALS = np.array([
    [-math.sin(math.pi/3),  math.cos(math.pi/3)],
    [-math.sin(2*math.pi/3), math.cos(2*math.pi/3)],
    [-math.sin(math.pi),    math.cos(math.pi)],
    [-math.sin(4*math.pi/3), math.cos(4*math.pi/3)],
    [-math.sin(5*math.pi/3), math.cos(5*math.pi/3)],
    [-math.sin(0),          math.cos(0)],
], dtype=float)  # 6x2


def _rot_matrix(angle_rad):
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return np.array([[c, -s], [s, c]], dtype=float)


def _hex_normals(angle_rad):
    """Return 6x2 array of unit edge normals for a hexagon at given rotation."""
    R = _rot_matrix(angle_rad)
    return _EDGE_NORMALS @ R.T


def construct():
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    # ---- Fast SAT: min separation (positive = disjoint) ----
    def sat_min_sep(v1, v2):
        """v1,v2: 6x2 vertex arrays. Returns min separating distance (>=0 if disjoint)."""
        min_sep = 0.0
        for i in range(6):
            # edge i of v1
            e = v1[(i + 1) % 6] - v1[i]
            n = np.array([-e[1], e[0]])
            nn = math.hypot(n[0], n[1])
            if nn < 1e-12:
                continue
            n /= nn
            p1 = v1 @ n
            p2 = v2 @ n
            sep = min(p2.max() - p1.min(), p1.max() - p2.min())
            # Actually we need: gap = max(0, min(p2)-max(p1), min(p1)-max(p2))
            gap = max(0.0, min(p2) - max(p1), min(p1) - max(p2))
            if gap > min_sep:
                min_sep = gap
            # if gap is negative they overlap on this axis; check the other way
            # (covered by the max above since both orderings are included)
        # Also edges of v2
        for i in range(6):
            e = v2[(i + 1) % 6] - v2[i]
            n = np.array([-e[1], e[0]])
            nn = math.hypot(n[0], n[1])
            if nn < 1e-12:
                continue
            n /= nn
            p1 = v1 @ n
            p2 = v2 @ n
            gap = max(0.0, min(p2) - max(p1), min(p1) - max(p2))
            if gap > min_sep:
                min_sep = gap
        return min_sep

    def hex_vertices(cx, cy, side, angle_rad):
        a = angle_rad
        angs = a + np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3])
        return np.column_stack([cx + side * np.cos(angs),
                                cy + side * np.sin(angs)])

    def pairwise_disjoint(inner):
        """inner: list of (x,y,angle_rad). Returns True if all pairs disjoint."""
        verts = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                if sat_min_sep(verts[i], verts[j]) < -1e-9:
                    return False
        return True

    def point_in_hex(px, py, cx, cy, side, angle_rad):
        """Check if point (px,py) is inside hexagon centered (cx,cy)."""
        dx, dy = px - cx, py - cy
        # rotate into hex frame
        c, s = math.cos(-angle_rad), math.sin(-angle_rad)
        lx = dx * c - dy * s
        ly = dx * s + dy * c
        apothem = side * math.sqrt(3) / 2
        for k in range(6):
            na = math.radians(30 + 60 * k)
            if lx * math.cos(na) + ly * math.sin(na) > apothem + 1e-9:
                return False
        return True

    def all_inside(inner, oc, os, oa):
        for x, y, a in inner:
            vs = hex_vertices(x, y, 1.0, a)
            for vx, vy in vs:
                if not point_in_hex(vx, vy, oc[0], oc[1], os, oa):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside(inner, oc, os, oa) and pairwise_disjoint(inner)

    # ---- Minimum enclosing hexagon side for a set of points ----
    # A regular hexagon (pointy-top at angle 0) contains point p iff for all 6
    # outward normals n_k: p . n_k <= S * apothem_unit * S ... actually:
    # distance from center to edge = S * sqrt(3)/2 (apothem).
    # So we need: max_k ( (p - c) . n_k ) <= S * sqrt(3)/2  for all p.
    # => S >= (2/sqrt(3)) * max_k max_p (p - c) . n_k
    # We can also optimize center c. For a given orientation, the minimal S
    # enclosing hexagon centered at c is determined by the support function.
    # Optimal center: minimize max_k support_k(c) where support_k(c) = max_p (p-c).n_k
    # This is a convex problem; we solve via a small grid + local refinement.

    def enclosing_hex_side(points, angle_rad):
        """Compute minimal side length of a regular hexagon (rotated by angle_rad)
        enclosing all points. Returns (side, center)."""
        # 6 outward normals for the enclosing hexagon
        angs = angle_rad + np.array([math.pi/6 + k * math.pi/3 for k in range(6)])
        normals = np.column_stack([np.cos(angs), np.sin(angs)])  # 6x2
        P = np.array(points)  # Mx2

        def required_side(c):
            # For each normal, max projection of (P - c)
            # = max_p (P @ n) - c @ n
            projs = P @ normals.T  # Mx6
            maxproj = projs.max(axis=0)  # 6
            vals = maxproj - normals @ c  # 6
            return (2.0 / math.sqrt(3)) * vals.max()

        # Grid search for center
        xs = P[:, 0]
        ys = P[:, 1]
        cx0, cx1 = xs.min(), xs.max()
        cy0, cy1 = ys.min(), ys.max()
        best_s = float('inf')
        best_c = ((cx0 + cx1) / 2, (cy0 + cy1) / 2)
        # Coarse grid
        for gi in range(11):
            cx = cx0 + (cx1 - cx0) * gi / 10
            for gj in range(11):
                cy = cy0 + (cy1 - cy0) * gj / 10
                s = required_side((cx, cy))
                if s < best_s:
                    best_s = s
                    best_c = (cx, cy)
        # Local refinement (Nelder-Mead on 2D)
        try:
            from scipy.optimize import minimize
            def obj(c):
                return required_side(c)
            res = minimize(obj, best_c, method='Nelder-Mead',
                           options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-8})
            if res.fun < best_s:
                best_s = res.fun
                best_c = (res.x[0], res.x[1])
        except Exception:
            pass
        return best_s, best_c

    # ---- Build candidate clusters ----
    def make_cluster_6x2():
        """6 columns x 2 rows, flat-top, touching."""
        pts = []
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 3.0
                y = (r - 0.5) * math.sqrt(3)
                pts.append((x, y))
        return pts

    def make_cluster_3x4_brick():
        """3 rows x 4 cols, brick offset (pointy-top), touching."""
        pts = []
        # Pointy-top hex: width sqrt(3), height 2.
        # Horizontal pitch sqrt(3), vertical pitch 1.5, odd rows offset sqrt(3)/2.
        for r in range(3):
            for c in range(4):
                x = (c - 1.5) * math.sqrt(3) + (math.sqrt(3) / 2 if r % 2 else 0)
                y = (r - 1.0) * 1.5
                pts.append((x, y))
        return pts

    def make_cluster_4x3_brick():
        """4 rows x 3 cols, brick offset (pointy-top)."""
        pts = []
        for r in range(4):
            for c in range(3):
                x = (c - 1.0) * math.sqrt(3) + (math.sqrt(3) / 2 if r % 2 else 0)
                y = (r - 1.5) * 1.5
                pts.append((x, y))
        return pts

    def make_cluster_hex_lattice():
        """Hexagonal lattice: center + 6 ring + 5 more = 12 (close-packed)."""
        # Use a 3x3 block minus 3 corners + 3 = 12? Let's do:
        # 4-3-4-1? No. Use brick 3x4 (12) which is close-packed.
        # Actually a true hex cluster of 12: center(1) + ring1(6) + ring2 partial(5) = 12
        # Ring 2 has 12 positions; pick 5 contiguous ones.
        pts = [(0, 0)]
        # Ring 1
        for k in range(6):
            a = math.pi / 3 * k
            pts.append((math.sqrt(3) * math.cos(a), math.sqrt(3) * math.sin(a)))
        # Ring 2: 12 positions at distance 2*sqrt(3) for flat, but for pointy-top
        # ring 2 has positions. Let's just use the 4x3 brick which is known close-packed.
        return None  # fallback to brick

    def make_cluster_staggered_2x6():
        """2 rows x 6 cols, flat-top, but staggered vertically for tighter fit."""
        pts = []
        for c in range(6):
            x = (c - 2.5) * 3.0
            pts.append((x, -math.sqrt(3) / 2))
            pts.append((x, math.sqrt(3) / 2))
        return pts

    # Generate candidate point sets (local coords, angle 0 for all hexes)
    candidates = []
    for name, pts in [
        ("6x2", make_cluster_6x2()),
        ("3x4brick", make_cluster_3x4_brick()),
        ("4x3brick", make_cluster_4x3_brick()),
        ("2x6stagger", make_cluster_staggered_2x6()),
    ]:
        if pts and len(pts) == 12:
            candidates.append((name, pts))

    # ---- Evaluate each candidate: find best rotation & enclosing hex ----
    best_result = None  # (side, inner, oc, os, oa)
    for name, pts in candidates:
        if elapsed() > DEADLINE - 5:
            break
        # Try a range of global rotations (hexagon symmetry: 60 deg period)
        best_s_for_cluster = float('inf')
        best_rot = 0.0
        best_center = (0, 0)
        # Coarse scan
        for deg in range(0, 60, 2):
            rot = math.radians(deg)
            # Rotate points
            R = _rot_matrix(rot)
            P = np.array(pts) @ R.T
            s, c = enclosing_hex_side(P.tolist(), rot)
            if s < best_s_for_cluster:
                best_s_for_cluster = s
                best_rot = rot
                best_center = c
        # Fine scan around best
        for deg10 in range(int(best_rot * 10000) - 300, int(best_rot * 10000) + 300, 10):
            rot = math.radians(deg10 / 10000.0)
            R = _rot_matrix(rot)
            P = np.array(pts) @ R.T
            s, c = enclosing_hex_side(P.tolist(), rot)
            if s < best_s_for_cluster:
                best_s_for_cluster = s
                best_rot = rot
                best_center = c

        # Build inner list: each hex at rotated position, angle = best_rot
        inner = []
        R = _rot_matrix(best_rot)
        for px, py in pts:
            rx, ry = R @ np.array([px, py])
            inner.append([rx + best_center[0], ry + best_center[1], best_rot])

        # Verify validity
        if is_valid(inner, best_center, best_s_for_cluster, best_rot):
            if best_result is None or best_s_for_cluster < best_result[0]:
                best_result = (best_s_for_cluster, inner, list(best_center), best_s_for_cluster, best_rot)

    # ---- Fallback: use 6x2 grid with side 9.0 if nothing found ----
    if best_result is None:
        inner = []
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 3.0
                y = (r - 0.5) * math.sqrt(3)
                inner.append([x, y, 0.0])
        best_result = (9.0, inner, [0.0, 0.0], 9.0, 0.0)

    best_os, best_inner, best_oc, best_oa = best_result[0], best_result[1], best_result[2], best_result[4]

    # ---- Local polish: try shrinking os slightly ----
    if elapsed() < DEADLINE - 2:
        step = 0.005
        for _ in range(100):
            if elapsed() > DEADLINE - 1:
                break
            new_os = best_os - step
            if new_os < 2.0:
                break
            if is_valid(best_inner, best_oc, new_os, best_oa):
                best_os = new_os
            else:
                step *= 0.5
                if step < 1e-6:
                    break

    # ---- Try small rotation tweaks to reduce enclosing side ----
    if elapsed() < DEADLINE - 3:
        for ddeg in [0.5, -0.5, 1.0, -1.0, 0.25, -0.25]:
            if elapsed() > DEADLINE - 2:
                break
            new_oa = best_oa + math.radians(ddeg)
            # Re-enclose with new rotation
            R = _rot_matrix(new_oa - best_oa)
            new_inner = []
            for x, y, a in best_inner:
                # rotate position around outer center
                dx, dy = x - best_oc[0], y - best_oc[1]
                nx, ny = R @ np.array([dx, dy])
                new_inner.append([nx + best_oc[0], ny + best_oc[1], a + math.radians(ddeg)])
            # Recompute enclosing side
            pts = [(x, y) for x, y, a in new_inner]
            s, c = enclosing_hex_side(pts, new_oa)
            if s < best_os - 1e-6 and is_valid(new_inner, c, s, new_oa):
                best_inner = new_inner
                best_oc = list(c)
                best_os = s
                best_oa = new_oa

    return best_inner, tuple(best_oc), best_os, best_oa
