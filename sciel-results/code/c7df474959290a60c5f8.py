"""Interval-arithmetic branch-and-bound CSP solver for hexagon packing."""
import math
import time
import numpy as np
import heapq

N = 12

# EVOLVE-BLOCK-START

_VERT_DIRS = np.array([[math.cos(math.pi/6 + i*math.pi/3),
                        math.sin(math.pi/6 + i*math.pi/3)] for i in range(6)])
_AXES0 = []
for i in range(6):
    e = _VERT_DIRS[i] - _VERT_DIRS[(i+1) % 6]
    _AXES0.append(np.array([-e[1], e[0]]))
_AXES0 = np.array(_AXES0)
_WORLD_AXES = np.array([[math.cos(a), math.sin(a)] for a in np.linspace(0, math.pi, 96)])

def _outer_normals(theta_deg):
    theta = math.radians(theta_deg)
    return np.array([[math.cos(theta + k*math.pi/3),
                      math.sin(theta + k*math.pi/3)] for k in range(6)])

def _build_initial():
    def axial(q, r):
        return np.array([math.sqrt(3)*(q + r/2.0), 1.5*r])
    ring1 = [(1,0),(1,-1),(0,-1),(-1,0),(-1,1),(0,1)]
    pts = [(0,0)] + list(ring1)
    ring2 = []
    for (q,r) in ring1:
        for (dq,dr) in ring1:
            nq, nr = q+dq, r+dr
            if (nq,nr) not in pts and (nq,nr) not in ring2:
                ring2.append((nq,nr))
    pts.extend(ring2[:5])
    C = np.array([axial(q,r) for (q,r) in pts])
    C -= C.mean(axis=0)
    return C, 0.0

def _sinusoid_range(A, B, a0, a1):
    R = math.hypot(A, B)
    if R < 1e-12:
        return 0.0, 0.0
    phi = math.atan2(B, A)
    cands = [a0, a1]
    for target in (0.0, math.pi, -math.pi):
        ac = phi + target
        k0 = math.floor((a0 - ac)/(2*math.pi))
        for kk in (k0-1, k0, k0+1, k0+2):
            ac2 = ac + kk*2*math.pi
            if a0 - 1e-12 <= ac2 <= a1 + 1e-12:
                cands.append(ac2)
    vals = [R*math.cos(c - phi) for c in cands]
    return min(vals), max(vals)

def _hex_proj_interval(c_lo, c_hi, a_lo, a_hi, ax):
    a_lo_r, a_hi_r = math.radians(a_lo), math.radians(a_hi)
    axx, axy = ax[0], ax[1]
    rx_lo, rx_hi = _sinusoid_range(axx, axy, a_lo_r, a_hi_r)
    ry_lo, ry_hi = _sinusoid_range(axy, -axx, a_lo_r, a_hi_r)
    proj_lo = 1e18; proj_hi = -1e18
    for k in range(6):
        ex, ey = _VERT_DIRS[k]
        lo_k = min(rx_lo*ex + ry_lo*ey, rx_lo*ex + ry_hi*ey,
                   rx_hi*ex + ry_lo*ey, rx_hi*ex + ry_hi*ey)
        hi_k = max(rx_lo*ex + ry_lo*ey, rx_lo*ex + ry_hi*ey,
                   rx_hi*ex + ry_lo*ey, rx_hi*ex + ry_hi*ey)
        proj_lo = min(proj_lo, lo_k); proj_hi = max(proj_hi, hi_k)
    cproj_lo = axx*c_lo[0] + axy*c_lo[1]
    cproj_hi = axx*c_hi[0] + axy*c_hi[1]
    return cproj_lo + proj_lo, cproj_hi + proj_hi

def _separated_on_world_axes(lo, hi, a_lo, a_hi, i, j):
    for ax in _WORLD_AXES:
        p1_lo, p1_hi = _hex_proj_interval(lo[i], hi[i], a_lo[i], a_hi[i], ax)
        p2_lo, p2_hi = _hex_proj_interval(lo[j], hi[j], a_lo[j], a_hi[j], ax)
        if p1_hi < p2_lo or p2_hi < p1_lo:
            return True
    return False

def _box_feasible(lo, hi, a_lo, a_hi):
    for i in range(N):
        for j in range(i+1, N):
            if not _separated_on_world_axes(lo, hi, a_lo, a_hi, i, j):
                return False
    return True

def _objective_upper(lo, hi, a_lo, a_hi, outer_n):
    best = 0.0
    for nk in outer_n:
        for i in range(N):
            p_lo, p_hi = _hex_proj_interval(lo[i], hi[i], a_lo[i], a_hi[i], nk)
            # absolute value: max(|p|) = max(p_hi, -p_lo)
            m = max(p_hi, -p_lo)
            if m > best:
                best = m
    return best / (math.sqrt(3)/2.0)

def _objective_point(C, a_deg, outer_n):
    best = 0.0
    for nk in outer_n:
        for i in range(N):
            a = math.radians(a_deg[i])
            ca, sa = math.cos(a), math.sin(a)
            R = np.array([[ca, -sa],[sa, ca]])
            verts = C[i] + _VERT_DIRS @ R.T
            proj = verts @ nk
            m = max(proj.max(), -proj.min())
            if m > best:
                best = m
    return best / (math.sqrt(3)/2.0)

def _point_feasible(C, a_deg):
    verts = []
    for i in range(N):
        a = math.radians(a_deg[i])
        ca, sa = math.cos(a), math.sin(a)
        R = np.array([[ca, -sa],[sa, ca]])
        verts.append(C[i] + _VERT_DIRS @ R.T)
    for i in range(N):
        for j in range(i+1, N):
            v1, v2 = verts[i], verts[j]
            sep = False
            axes = []
            for k in range(6):
                axes.append(v1[k]-v1[(k+1)%6])
                axes.append(v2[k]-v2[(k+1)%6])
            for e in axes:
                n = np.array([-e[1], e[0]])
                p1 = v1 @ n; p2 = v2 @ n
                if p1.max() < p2.min() or p2.max() < p1.min():
                    sep = True; break
            if not sep:
                return False
    return True

def construct():
    start = time.time()
    deadline = start + 145.0

    C0, a0 = _build_initial()
    outer_n = _outer_normals(0.0)

    slack_pos = 0.4
    slack_ang = 10.0
    lo = C0 - slack_pos
    hi = C0 + slack_pos
    a_lo = np.full(N, a0 - slack_ang)
    a_hi = np.full(N, a0 + slack_ang)

    if not _point_feasible(C0, np.full(N, a0)):
        lo = C0 - 1.0; hi = C0 + 1.0
        a_lo = np.full(N, a0 - 30.0); a_hi = np.full(N, a0 + 30.0)

    incumbent = _objective_point(C0, np.full(N, a0), outer_n)
    best_C = C0.copy()
    best_a = np.full(N, a0)

    root_ub = _objective_upper(lo, hi, a_lo, a_hi, outer_n)
    heap = [(-root_ub, 0, lo.copy(), hi.copy(), a_lo.copy(), a_hi.copy())]
    iter_count = 0

    def elapsed_ok():
        return time.time() < deadline

    while heap and elapsed_ok():
        neg_ub, _, lo_c, hi_c, a_lo_c, a_hi_c = heapq.heappop(heap)
        ub = -neg_ub
        if ub >= incumbent:
            continue
        w_pos = (hi_c - lo_c).max()
        w_ang = (a_hi_c - a_lo_c).max()
        if w_pos < 1e-3 and w_ang < 1e-2:
            Cc = (lo_c + hi_c)/2.0
            ac = (a_lo_c + a_hi_c)/2.0
            if _point_feasible(Cc, ac):
                obj = _objective_point(Cc, ac, outer_n)
                if obj < incumbent:
                    incumbent = obj
                    best_C = Cc.copy()
                    best_a = ac.copy()
            continue
        if not _box_feasible(lo_c, hi_c, a_lo_c, a_hi_c):
            continue
        pos_widths = hi_c - lo_c
        ang_widths = a_hi_c - a_lo_c
        max_pw = pos_widths.max()
        max_aw = ang_widths.max()
        if max_pw >= max_aw and max_pw > 1e-4:
            idx = np.unravel_index(pos_widths.argmax(), pos_widths.shape)
            i, coord = idx
            mid = (lo_c[i, coord] + hi_c[i, coord]) / 2.0
            lo1 = lo_c.copy(); hi1 = hi_c.copy()
            lo2 = lo_c.copy(); hi2 = hi_c.copy()
            hi1[i, coord] = mid
            lo2[i, coord] = mid
            children = [(lo1, hi1, a_lo_c, a_hi_c), (lo2, hi2, a_lo_c, a_hi_c)]
        elif max_aw > 1e-3:
            i = int(ang_widths.argmax())
            mid = (a_lo_c[i] + a_hi_c[i]) / 2.0
            a_lo1 = a_lo_c.copy(); a_hi1 = a_hi_c.copy()
            a_lo2 = a_lo_c.copy(); a_hi2 = a_hi_c.copy()
            a_hi1[i] = mid
            a_lo2[i] = mid
            children = [(lo_c, hi_c, a_lo1, a_hi1), (lo_c, hi_c, a_lo2, a_hi2)]
        else:
            continue
        for (llo, lhi, la_lo, la_hi) in children:
            cub = _objective_upper(llo, lhi, la_lo, la_hi, outer_n)
            if cub < incumbent:
                heapq.heappush(heap, (-cub, 0, llo.copy(), lhi.copy(), la_lo.copy(), la_hi.copy()))
        iter_count += 1

    if not _point_feasible(best_C, best_a):
        best_C = C0.copy()
        best_a = np.full(N, a0)
    final_side = _objective_point(best_C, best_a, outer_n)

    inner = [[float(best_C[i,0]), float(best_C[i,1]), float(best_a[i])] for i in range(N)]
    return inner, (0.0, 0.0), float(final_side), 0.0
# EVOLVE-BLOCK-END
