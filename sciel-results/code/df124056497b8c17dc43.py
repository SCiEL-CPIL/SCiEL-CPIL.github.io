"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 115.0
    
    sqrt3 = np.sqrt(3.0)
    
    def hex_vertices(cx, cy, side, angle_deg):
        angle = np.radians(angle_deg)
        base = np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0])
        angles = np.radians(base + angle_deg)
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])
    
    def point_hex_violation(px, py, cx, cy, side, angle_deg):
        """Return max(0, distance by which point violates hexagon boundary)."""
        a = np.radians(angle_deg)
        dx = px - cx
        dy = py - cy
        lx = dx * np.cos(-a) - dy * np.sin(-a)
        ly = dx * np.sin(-a) + dy * np.cos(-a)
        inr = side * sqrt3 / 2.0
        max_v = 0.0
        for k in range(6):
            t = np.radians(30.0 + 60.0 * k)
            d = lx * np.cos(t) + ly * np.sin(t)
            if d > inr:
                max_v = max(max_v, d - inr)
        return max_v
    
    def sat_depth(v1, v2):
        """Min separation between two convex hexagons. <0 means overlap."""
        best = np.inf
        for i in range(6):
            a, b = v1[i], v1[(i+1)%6]
            ex, ey = b[0]-a[0], b[1]-a[1]
            L = np.hypot(ex, ey)
            if L < 1e-12: continue
            nx, ny = -ey/L, ex/L
            p1 = np.array([v1[:,0]*nx + v1[:,1]*ny])
            p2 = np.array([v2[:,0]*nx + v2[:,1]*ny])
            s1 = np.max(p1) - np.max(p2)
            s2 = np.max(p2) - np.max(p1)
            if s1 > 0 or s2 > 0:
                best = min(best, max(s1, s2))
                if best <= 0: return best
        for i in range(6):
            a, b = v2[i], v2[(i+1)%6]
            ex, ey = b[0]-a[0], b[1]-a[1]
            L = np.hypot(ex, ey)
            if L < 1e-12: continue
            nx, ny = -ey/L, ex/L
            p1 = np.array([v1[:,0]*nx + v1[:,1]*ny])
            p2 = np.array([v2[:,0]*nx + v2[:,1]*ny])
            s1 = np.max(p1) - np.max(p2)
            s2 = np.max(p2) - np.max(p1)
            if s1 > 0 or s2 > 0:
                best = min(best, max(s1, s2))
                if best <= 0: return best
        return best
    
    def min_outer_side(pos, ang, oc, oa):
        """Binary search min outer side for given config."""
        def fits(s):
            for i in range(N):
                verts = hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i])
                for v in verts:
                    if point_hex_violation(v[0], v[1], oc[0], oc[1], s, oa) > 1e-9:
                        return False
            return True
        lo, hi = 1.0, 25.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if fits(mid): hi = mid
            else: lo = mid
        return hi
    
    def eval_cfg(pos, ang, oc, oa):
        verts = [hex_vertices(pos[i][0], pos[i][1], 1.0, ang[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_depth(verts[i], verts[j]) < -1e-6:
                    return None
        return min_outer_side(pos, ang, oc, oa)
    
    cands = []
    
    # Lattice configs: rows x cols with hex offset
    for (rows, cols) in [(3,4), (4,3), (2,6), (6,2)]:
        for sp in [sqrt3, sqrt3*1.005, sqrt3*1.01, sqrt3*1.02]:
            pos, ang = [], []
            for r in range(rows):
                for c in range(cols):
                    x = (c - (cols-1)/2.0) * sp
                    y = (r - (rows-1)/2.0) * 1.5
                    if r % 2 == 1:
                        x += sp / 2.0
                    pos.append((x, y))
                    ang.append(0.0)
            s = eval_cfg(pos, ang, (0.0, 0.0), 0.0)
            if s is not None:
                cands.append((s, pos, ang, (0.0,0.0), 0.0))
    
    # With 30-deg rotation
    for (rows, cols) in [(3,4), (4,3)]:
        for sp in [sqrt3, sqrt3*1.01]:
            pos, ang = [], []
            for r in range(rows):
                for c in range(cols):
                    x = (c - (cols-1)/2.0) * sp
                    y = (r - (rows-1)/2.0) * 1.5
                    if r % 2 == 1:
                        x += sp / 2.0
                    pos.append((x, y))
                    ang.append(30.0)
            s = eval_cfg(pos, ang, (0.0, 0.0), 30.0)
            if s is not None:
                cands.append((s, pos, ang, (0.0,0.0), 30.0))
    
    if not cands:
        inner = [[(i - 5.5)*2.0, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 15.0, 0.0
    
    cands.sort(key=lambda x: x[0])
    bs, bp, ba, boc, boa = cands[0]
    
    # Refine with Nelder-Mead
    def obj(p):
        ip = p[:36]
        ocx, ocy, os, oa = p[36], p[37], p[38], p[39]
        if os < 1.0: return 1e6
        vs = [hex_vertices(ip[i*3], ip[i*3+1], 1.0, ip[i*3+2]) for i in range(N)]
        op = 0.0
        for i in range(N):
            for j in range(i+1, N):
                d = sat_depth(vs[i], vs[j])
                if d < 0: op += d*d
        bp2 = 0.0
        for v in vs:
            for pt in v:
                viol = point_hex_violation(pt[0], pt[1], ocx, ocy, os, oa)
                if viol > 0: bp2 += viol*viol
        return os + 50.0*op + 50.0*bp2
    
    p0 = np.zeros(40)
    for i in range(N):
        p0[i*3] = bp[i][0]
        p0[i*3+1] = bp[i][1]
        p0[i*3+2] = ba[i]
    p0[36:40] = [boc[0], boc[1], bs, boa]
    
    bp_best = p0.copy()
    bo_best = obj(p0)
    
    if time.time() - start_time < deadline - 10:
        try:
            res = minimize(obj, p0, method='Nelder-Mead',
                          options={'maxiter': 400, 'xatol': 1e-3, 'fatol': 1e-3, 'adaptive': True})
            if res.fun < bo_best:
                bo_best = res.fun
                bp_best = res.x
        except:
            pass
    
    inner = [[bp_best[i*3], bp_best[i*3+1], bp_best[i*3+2]] for i in range(N)]
    ocx, ocy, os, oa = bp_best[36], bp_best[37], bp_best[38], bp_best[39]
    
    # Validate
    vs = [hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2]) for i in range(N)]
    ok = True
    for i in range(N):
        for j in range(i+1, N):
            if sat_depth(vs[i], vs[j]) < -1e-5:
                ok = False
                break
        if not ok: break
    if ok:
        for v in vs:
            for pt in v:
                if point_hex_violation(pt[0], pt[1], ocx, ocy, os, oa) > 1e-5:
                    ok = False
                    break
            if not ok: break
    
    if not ok:
        inner = [[p[0], p[1], a] for p, a in zip(bp, ba)]
        ocx, ocy = boc
        os, oa = bs, boa
    
    return inner, (ocx, ocy), os, oa
# EVOLVE-BLOCK-END
