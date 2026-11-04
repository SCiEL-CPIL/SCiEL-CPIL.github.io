import math
import time
import random
import numpy as np

N = 11
DEADLINE = 150.0
T0 = time.time()

def _time_left():
    return DEADLINE - time.time()

def _hex_verts(cx, cy, s, ang_deg):
    a0 = math.radians(ang_deg)
    return [(cx + s * math.cos(a0 + math.radians(60 * k)),
             cy + s * math.sin(a0 + math.radians(60 * k))) for k in range(6)]

def _min_side(inner, ocx, ocy, rot_deg):
    """Minimum outer side length so all inner hex vertices are inside."""
    r = math.radians(rot_deg)
    # Outer hexagon vertices at angles r + 60*k, vertices are at s*cos/sin
    # The "radius" (center to vertex) = s.
    # For a point (px, py) relative to center (ox, oy), the maximum of
    #   proj = (px-ox)*cos(theta) + (py-oy)*sin(theta)
    # over the 6 face-normal directions must be <= s.
    # Face normals of regular hexagon are at angles r + 30 + 60*k.
    # Actually, the support function of a regular hexagon with side s:
    # h(n) = s * cos(30°) if n is a face normal direction... 
    # Let me think more carefully.
    # A regular hexagon with side length s has inradius (apothem) = s*sqrt(3)/2.
    # The support function in direction n (unit vector) is:
    # h(n) = s*sqrt(3)/2 * (1 / cos(angle between n and nearest face normal))
    # But simpler: the hexagon is the intersection of 6 half-planes.
    # Vertex at angle r+60k has distance s from center.
    # Edge between vertices k and k+1: the midpoint is at distance s*sqrt(3)/2.
    # The outward normal of edge k (between vertex k and k+1) points at angle r + 30 + 60k.
    # The half-plane: (p - c) · n_k <= s*sqrt(3)/2
    # So for all vertices p of inner hexagons, and all 6 normals n_k:
    #   (p - c) · n_k <= s*sqrt(3)/2
    # Thus s >= max_k max_p (p-c)·n_k / (sqrt(3)/2) = 2/sqrt(3) * max(...)
    # Wait, that's not right either. Let me re-derive.
    # Regular hexagon centered at origin, side length s, rotated by r.
    # Vertices: v_k = s * (cos(r + 60k°), sin(r + 60k°))
    # Edge from v_k to v_{k+1}: midpoint m_k = (v_k + v_{k+1})/2
    # |m_k| = s * cos(30°) = s * sqrt(3)/2
    # Outward normal direction: angle r + 30° + 60k°
    # Half-plane: (x - c) · n_k <= |m_k| = s * sqrt(3)/2
    # So the constraint is: s >= 2/(sqrt(3)) * max over all inner vertices and all k of (p - c) · n_k
    mx = 0.0
    for k in range(6):
        alpha = math.radians(rot_deg + 30 + 60 * k)
        nx, ny = math.cos(alpha), math.sin(alpha)
        for (x, y, a) in inner:
            pts = _hex_verts(x, y, 1.0, a)
            for (px, py) in pts:
                proj = (px - ocx) * nx + (py - ocy) * ny
                if proj > mx:
                    mx = proj
    return 2.0 * mx / math.sqrt(3.0)

def _point_in_hex(px, py, ocx, ocy, s, rot_deg):
    r = math.radians(rot_deg)
    for k in range(6):
        alpha = math.radians(rot_deg + 30 + 60 * k)
        nx, ny = math.cos(alpha), math.sin(alpha)
        if (px - ocx) * nx + (py - ocy) * ny > s * math.sqrt(3.0) / 2.0 + 1e-9:
            return False
    return True

def _seg_dist(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    len2 = dx * dx + dy * dy
    if len2 < 1e-18:
        return math.hypot(px - ax, py - ay)
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / len2))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))

def _min_dist_poly(A, B):
    """Minimum distance between two convex polygons (lists of (x,y) tuples)."""
    # Check if any vertex of A is inside B or vice versa
    for (px, py) in A:
        # Point in polygon via winding/cross products
        sign = 0
        inside = True
        n = len(B)
        for i in range(n):
            cx1, cy1 = B[i]
            cx2, cy2 = B[(i + 1) % n]
            c = (cx2 - cx1) * (py - cy1) - (cy2 - cy1) * (px - cx1)
            if abs(c) < 1e-12:
                continue
            s = 1 if c > 0 else -1
            if sign == 0:
                sign = s
            elif sign != s:
                inside = False
                break
        if inside:
            return 0.0
    for (px, py) in B:
        sign = 0
        inside = True
        n = len(A)
        for i in range(n):
            cx1, cy1 = A[i]
            cx2, cy2 = A[(i + 1) % n]
            c = (cx2 - cx1) * (py - cy1) - (cy2 - cy1) * (px - cx1)
            if abs(c) < 1e-12:
                continue
            s = 1 if c > 0 else -1
            if sign == 0:
                sign = s
            elif sign != s:
                inside = False
                break
        if inside:
            return 0.0
    # Otherwise compute min distance between edges
    md = float('inf')
    for i in range(len(A)):
        ax1, ay1 = A[i]
        ax2, ay2 = A[(i + 1) % len(A)]
        for j in range(len(B)):
            bx1, by1 = B[j]
            bx2, by2 = B[(j + 1) % len(B)]
            d1 = _seg_dist(ax1, ay1, bx1, by1, bx2, by2)
            d2 = _seg_dist(bx1, by1, ax1, ay1, ax2, ay2)
            if d1 < md:
                md = d1
            if d2 < md:
                md = d2
    return md

def _overlap_penalty(inner, ocx, ocy, s, rot_deg):
    """Penalty: sum of squared overlap distances for inner pairs + outer violations."""
    polys = [_hex_verts(x, y, 1.0, a) for (x, y, a) in inner]
    pen = 0.0
    # Inner-inner
    for i in range(N):
        for j in range(i + 1, N):
            d = _min_dist_poly(polys[i], polys[j])
            if d < 1e-9:
                # Overlapping
                # Estimate penetration by checking vertex distances
                # Use a simple penalty
                pen += 1.0
            elif d < 0.01:
                pen += (0.01 - d) ** 2 * 100.0
    # Inner-outer
    for i in range(N):
        for (px, py) in polys[i]:
            if not _point_in_hex(px, py, ocx, ocy, s, rot_deg):
                # Compute how far outside
                r = math.radians(rot_deg)
                max_viol = 0.0
                for k in range(6):
                    alpha = math.radians(rot_deg + 30 + 60 * k)
                    nx, ny = math.cos(alpha), math.sin(alpha)
                    proj = (px - ocx) * nx + (py - ocy) * ny
                    limit = s * math.sqrt(3.0) / 2.0
                    if proj > limit:
                        viol = proj - limit
                        if viol > max_viol:
                            max_viol = viol
                pen += max_viol ** 2 * 100.0
    return pen

def _construct():
    best = None
    best_score = float('inf')
    budget_end = T0 + DEADLINE - 5.0
    
    # Start with the known flower packing: 1 center + 6 at distance sqrt(3) + 4 at 2*sqrt(3)
    d = math.sqrt(3.0)
    inner0 = [[0.0, 0.0, 0.0]]
    for k in range(6):
        a = math.radians(30 + 60 * k)
        inner0.append([d * math.cos(a), d * math.sin(a), 0.0])
    for k in range(4):
        a = math.radians(30 + 60 * k)
        inner0.append([2 * d * math.cos(a), 2 * d * math.sin(a), 0.0])
    # That's 11 hexagons
    assert len(inner0) == N
    
    s0 = _min_side(inner0, 0.0, 0.0, 0.0)
    pen0 = _overlap_penalty(inner0, 0.0, 0.0, s0, 0.0)
    
    if pen0 < 1e-9:
        best_score = s0
        best = (inner0, (0.0, 0.0), s0, 0.0)
    
    # Try to improve via local search
    rng = random.Random(12345)
    cur_inner = [list(v) for v in inner0]
    cur_oc = (0.0, 0.0)
    cur_rot = 0.0
    cur_s = s0
    cur_pen = pen0
    
    # Local search: perturb positions, check validity
    step = 0.15
    n_iter = 0
    while time.time() < budget_end and n_iter < 50000:
        n_iter += 1
        # Pick a random hexagon and perturb
        idx = rng.randrange(N)
        dx = rng.gauss(0, step)
        dy = rng.gauss(0, step)
        da = rng.gauss(0, step * 10)  # degrees
        old = [cur_inner[idx][0], cur_inner[idx][1], cur_inner[idx][2]]
        cur_inner[idx][0] += dx
        cur_inner[idx][1] += dy
        cur_inner[idx][2] += da
        
        new_s = _min_side(cur_inner, cur_oc[0], cur_oc[1], cur_rot)
        new_pen = _overlap_penalty(cur_inner, cur_oc[0], cur_oc[1], new_s, cur_rot)
        
        # Accept if no penalty and s improves, or small penalty
        if new_pen < 1e-6 and new_s < best_score - 1e-9:
            best_score = new_s
            best = ([list(v) for v in cur_inner], (cur_oc[0], cur_oc[1]), new_s, cur_rot)
            cur_s = new_s
            cur_pen = new_pen
        elif new_pen < cur_pen and new_pen < 0.1:
            # Accept if penalty decreases
            cur_s = new_s
            cur_pen = new_pen
        else:
            # Reject
            cur_inner[idx][0] = old[0]
            cur_inner[idx][1] = old[1]
            cur_inner[idx][2] = old[2]
        
        # Occasionally perturb outer center
        if n_iter % 500 == 0:
            old_oc = (cur_oc[0], cur_oc[1])
            cur_oc = (old_oc[0] + rng.gauss(0, 0.05), old_oc[1] + rng.gauss(0, 0.05))
            new_s = _min_side(cur_inner, cur_oc[0], cur_oc[1], cur_rot)
            new_pen = _overlap_penalty(cur_inner, cur_oc[0], cur_oc[1], new_s, cur_rot)
            if new_pen > cur_pen * 2 + 0.1:
                cur_oc = old_oc
                cur_s = s0
                cur_pen = pen0
    
    # Multi-start random search
    for trial in range(30):
        if time.time() > budget_end:
            break
        # Random seed
        ri = [[rng.uniform(-3, 3), rng.uniform(-3, 3), rng.uniform(0, 60)] for _ in range(N)]
        roc = (rng.uniform(-1, 1), rng.uniform(-1, 1))
        rrot = rng.uniform(0, 60)
        rs = _min_side(ri, roc[0], roc[1], rrot)
        rpen = _overlap_penalty(ri, roc[0], roc[1], rs, rrot)
        
        # Local refine
        for it in range(5000):
            if time.time() > budget_end:
                break
            idx = rng.randrange(N)
            dx = rng.gauss(0, 0.2)
            dy = rng.gauss(0, 0.2)
            da = rng.gauss(0, 2.0)
            old = [ri[idx][0], ri[idx][1], ri[idx][2]]
            ri[idx][0] += dx
            ri[idx][1] += dy
            ri[idx][2] += da
            ns = _min_side(ri, roc[0], roc[1], rrot)
            np_ = _overlap_penalty(ri, roc[0], roc[1], ns, rrot)
            if np_ < rpen:
                rpen = np_
                rs = ns
                if rpen < 1e-9 and ns < best_score:
                    best_score = ns
                    best = ([list(v) for v in ri], (roc[0], roc[1]), ns, rrot)
            else:
                ri[idx][0] = old[0]
                ri[idx][1] = old[1]
                ri[idx][2] = old[2]
    
    if best is None:
        return inner0, (0.0, 0.0), s0, 0.0
    
    # Final: recompute tight s
    inner_f, oc_f, s_f, rot_f = best
    s_f = _min_side(inner_f, oc_f[0], oc_f[1], rot_f)
    return inner_f, (oc_f[0], oc_f[1]), s_f, rot_f

def construct():
    return _construct()
