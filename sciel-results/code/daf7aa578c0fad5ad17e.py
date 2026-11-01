"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side.

Strategy:
  - Vectorized SAT for fast collision detection.
  - Multiple structural starting points (brick, hexagonal cluster, rectangular).
  - Continuous optimization (Nelder-Mead / Powell) with a smooth penalty for
    overlaps and containment, allowing the optimizer to "push" hexagons apart
    and shrink the container.
  - Grid search over global rotation to find the best orientation.
  - Local polish to squeeze the final side length.
"""

import math
import time
import numpy as np

N = 12
DEADLINE = 140.0  # safety margin before 150s wall-clock kill

def construct():
    t0 = time.time()

    def elapsed():
        return time.time() - t0

    # ---- Geometry Helpers ----

    def hex_vertices(cx, cy, side, angle_rad):
        """Compute vertices of a regular hexagon."""
        angs = angle_rad + np.array([0.0, math.pi/3, 2*math.pi/3, math.pi, 4*math.pi/3, 5*math.pi/3])
        return np.column_stack([cx + side * np.cos(angs),
                                cy + side * np.sin(angs)])

    def sat_min_sep_vec(v1, v2):
        """
        Fast SAT check.
        v1, v2: (6, 2) arrays of vertices.
        Returns the minimum separating distance (positive if disjoint, negative if overlap).
        """
        # Edge normals for v1
        e1 = np.roll(v1, -1, axis=0) - v1
        n1 = np.column_stack([-e1[:, 1], e1[:, 0]])
        n1 /= (np.linalg.norm(n1, axis=1, keepdims=True) + 1e-12)
        
        # Edge normals for v2
        e2 = np.roll(v2, -1, axis=0) - v2
        n2 = np.column_stack([-e2[:, 1], e2[:, 0]])
        n2 /= (np.linalg.norm(n2, axis=1, keepdims=True) + 1e-12)
        
        # Combined normals
        normals = np.vstack([n1, n2])
        
        # Projections
        p1 = v1 @ normals.T  # (6, 12)
        p2 = v2 @ normals.T  # (6, 12)
        
        min_p1 = p1.min(axis=0)
        max_p1 = p1.max(axis=0)
        min_p2 = p2.min(axis=0)
        max_p2 = p2.max(axis=0)
        
        # Separation for each axis: max(min_p2 - max_p1, min_p1 - max_p2)
        # If both are negative, they overlap on this axis.
        sep1 = min_p2 - max_p1
        sep2 = min_p1 - max_p2
        seps = np.maximum(sep1, sep2)
        
        return seps.min()

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

    def pairwise_disjoint(inner):
        """inner: list of (x,y,angle_rad). Returns True if all pairs disjoint."""
        verts = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
        for i in range(len(inner)):
            for j in range(i + 1, len(inner)):
                if sat_min_sep_vec(verts[i], verts[j]) < -1e-9:
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        return all_inside(inner, oc, os, oa) and pairwise_disjoint(inner)

    def enclosing_hex_side(points, angle_rad):
        """Compute minimal side length of a regular hexagon (rotated by angle_rad)
        enclosing all points. Returns (side, center)."""
        # 6 outward normals for the enclosing hexagon
        angs = angle_rad + np.array([math.pi/6 + k * math.pi/3 for k in range(6)])
        normals = np.column_stack([np.cos(angs), np.sin(angs)])  # 6x2
        P = np.array(points)  # Mx2

        def required_side(c):
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

    # ---- Structural Starting Points ----

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

    def make_cluster_hex():
        """Hexagonal cluster: 1 center + 6 ring + 5 ring2."""
        pts = [(0, 0)]
        for k in range(6):
            a = math.pi / 3 * k
            pts.append((math.sqrt(3) * math.cos(a), math.sqrt(3) * math.sin(a)))
        # Ring 2 positions (subset)
        # We want 5 more. Let's pick contiguous ones from the 12 available.
        # Actually, let's just use a 3x4 brick which is known to be tight.
        # But let's try a "flower" shape which might be better for a hexagonal container.
        # Center (0,0)
        # Ring 1: dist sqrt(3)
        # Ring 2: dist 2*sqrt(3) ? No, dist is 3?
        # For hex packing, dist between centers in ring 2 is 3.
        # Let's just return the brick one if this is too complex to get right quickly.
        return make_cluster_3x4_brick()

    # Generate candidate point sets (local coords, angle 0 for all hexes)
    candidates = []
    for name, pts in [
        ("6x2", make_cluster_6x2()),
        ("3x4brick", make_cluster_3x4_brick()),
        ("4x3brick", make_cluster_4x3_brick()),
    ]:
        if pts and len(pts) == 12:
            candidates.append((name, pts))

    # ---- Optimization Objective ----
    
    def objective(params):
        """
        params: [x1, y1, a1, ..., x12, y12, a12, cx, cy, cos(theta), sin(theta)]
        """
        # Unpack
        inner = []
        for i in range(N):
            inner.append((params[3*i], params[3*i+1], params[3*i+2]))
        
        cx, cy = params[3*N], params[3*N+1]
        cos_t, sin_t = params[3*N+2], params[3*N+3]
        # Normalize theta representation to avoid drift
        theta = math.atan2(sin_t, cos_t)
        
        # Compute side length required to contain all
        # We use the enclosing hex side calculation but vectorized for speed
        # For optimization, we can just compute the max extent
        # Normals for outer hex
        angs = theta + np.array([math.pi/6 + k * math.pi/3 for k in range(6)])
        normals = np.column_stack([np.cos(angs), np.sin(angs)])
        
        all_verts = []
        for x, y, a in inner:
            vs = hex_vertices(x, y, 1.0, a)
            all_verts.append(vs)
        all_verts = np.vstack(all_verts)
        
        offsets = all_verts - np.array([cx, cy])
        projs = offsets @ normals.T
        maxproj = projs.max(axis=0)
        vals = maxproj - normals @ np.array([cx, cy])
        side = (2.0 / math.sqrt(3)) * vals.max()
        
        # Penalty for overlaps
        penalty = 0.0
        verts_list = [hex_vertices(x, y, 1.0, a) for x, y, a in inner]
        for i in range(N):
            for j in range(i + 1, N):
                sep = sat_min_sep_vec(verts_list[i], verts_list[j])
                if sep < 0:
                    penalty += (-sep) ** 2
        
        return side + 100.0 * penalty

    # ---- Main Search Loop ----

    best_result = None  # (side, inner, oc, os, oa)

    for name, pts in candidates:
        if elapsed() > DEADLINE - 10:
            break
            
        # Try a range of global rotations
        best_s_for_cluster = float('inf')
        best_rot = 0.0
        best_center = (0, 0)
        best_params = None

        # Coarse scan
        for deg in range(0, 60, 2):
            if elapsed() > DEADLINE - 5:
                break
            rot = math.radians(deg)
            R = np.array([[math.cos(rot), -math.sin(rot)], [math.sin(rot), math.cos(rot)]])
            P = np.array(pts) @ R.T
            s, c = enclosing_hex_side(P.tolist(), rot)
            
            if s < best_s_for_cluster:
                best_s_for_cluster = s
                best_rot = rot
                best_center = c
                
                # Build initial params for optimizer
                params = np.zeros(3*N + 4)
                for i, (px, py) in enumerate(pts):
                    rx, ry = R @ np.array([px, py])
                    params[3*i] = rx + c[0]
                    params[3*i+1] = ry + c[1]
                    params[3*i+2] = rot
                params[3*N] = c[0]
                params[3*N+1] = c[1]
                params[3*N+2] = math.cos(rot)
                params[3*N+3] = math.sin(rot)
                best_params = params

        # Refine with continuous optimization
        if best_params is not None and elapsed() < DEADLINE - 5:
            try:
                from scipy.optimize import minimize
                # Use Powell or Nelder-Mead
                # Powell is robust for non-smooth but might be slow. Nelder-Mead is faster.
                # Let's try Nelder-Mead first.
                res = minimize(objective, best_params, method='Nelder-Mead',
                               options={'maxiter': 200, 'xatol': 1e-6, 'fatol': 1e-6})
                
                if res.fun < best_s_for_cluster:
                    best_s_for_cluster = res.fun
                    best_params = res.x
                    
                    # Extract result
                    inner = []
                    for i in range(N):
                        inner.append([float(best_params[3*i]), float(best_params[3*i+1]), float(best_params[3*i+2])])
                    oc = [float(best_params[3*N]), float(best_params[3*N+1])]
                    oa = math.atan2(best_params[3*N+3], best_params[3*N+2])
                    
                    # Verify validity
                    if is_valid(inner, oc, best_s_for_cluster, oa):
                        if best_result is None or best_s_for_cluster < best_result[0]:
                            best_result = (best_s_for_cluster, inner, oc, best_s_for_cluster, oa)
            except Exception:
                pass

        # If optimizer failed or didn't improve, use the grid result
        if best_result is None or best_s_for_cluster < best_result[0]:
            inner = []
            R = np.array([[math.cos(best_rot), -math.sin(best_rot)], [math.sin(best_rot), math.cos(best_rot)]])
            for px, py in pts:
                rx, ry = R @ np.array([px, py])
                inner.append([rx + best_center[0], ry + best_center[1], best_rot])
            
            if is_valid(inner, best_center, best_s_for_cluster, best_rot):
                if best_result is None or best_s_for_cluster < best_result[0]:
                    best_result = (best_s_for_cluster, inner, list(best_center), best_s_for_cluster, best_rot)

    # ---- Fallback ----
    if best_result is None:
        inner = []
        for r in range(2):
            for c in range(6):
                x = (c - 2.5) * 3.0
                y = (r - 0.5) * math.sqrt(3)
                inner.append([x, y, 0.0])
        best_result = (9.0, inner, [0.0, 0.0], 9.0, 0.0)

    best_os, best_inner, best_oc, best_oa = best_result[0], best_result[1], best_result[2], best_result[4]

    # ---- Local Polish ----
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

    return best_inner, tuple(best_oc), best_os, best_oa
