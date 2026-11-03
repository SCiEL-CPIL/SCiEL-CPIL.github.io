"""Rigid-body dynamics packing of 12 unit hexagons in a regular hexagon."""
import numpy as np
import time

N = 12
_DEADLINE = 110.0  # safety margin under the 150s wall-clock kill

# --- Geometry helpers -------------------------------------------------------
def _hex_verts_local():
    # unit regular hexagon, flat-top, centered at origin, circumradius 1
    return np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)])

_LOCAL = _hex_verts_local()
_LOCAL2 = _LOCAL * _LOCAL          # squared radius = 1 for each vertex
_HALF = 0.5 * np.sqrt(3.0)          # apothem of unit hexagon

def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])

def _world_verts(p, a):
    return p + _LOCAL @ _rot(a).T

def _outer_verts(cx, cy, side, rot):
    # regular hexagon, circumradius = side, flat-top, rotated by rot
    return np.array([[cx + side * np.cos(rot + np.radians(k)),
                      cy + side * np.sin(rot + np.radians(k))] for k in range(0, 360, 60)])

def _signed_dist_point_poly(px, py, verts):
    # signed distance from point (px,py) to convex polygon verts (CCW).
    # >0 inside, <0 outside (magnitude = distance to nearest edge).
    n = len(verts)
    best = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        ex, ey = x2 - x1, y2 - y1
        L2 = ex * ex + ey * ey
        if L2 < 1e-14:
            continue
        t = ((px - x1) * ex + (py - y1) * ey) / L2
        t = max(0.0, min(1.0, t))
        qx, qy = x1 + t * ex, y1 + t * ey
        d = px - qx
        f = py - qy
        # inside if point is left of each directed edge (CCW)
        cross = ex * f - ey * d
        if cross < 0:
            dist = np.hypot(d, f)
            if dist < best:
                best = dist
    return -best  # negative if outside

def _hex_overlap(a, pa, b, pb):
    """Squared separation margin between two unit hexagons.
    Positive => separated by that much (margin), negative => overlap (penetration).
    We return the *penetration* depth (positive when overlapping)."""
    # SAT over all 12 edge normals (6 from each)
    verts_a = _world_verts(pa, a)
    verts_b = _world_verts(pb, b)
    all_verts = [verts_a, verts_b]
    min_margin = np.inf
    for verts in all_verts:
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            ex, ey = x2 - x1, y2 - y1
            L = np.hypot(ex, ey)
            if L < 1e-12:
                continue
            nx, ny = ey / L, -ex / L  # outward normal (CCW polygon)
            # project both polys
            proj_a = verts_a @ np.array([nx, ny])
            proj_b = verts_b @ np.array([nx, ny])
            lo_a, hi_a = proj_a.min(), proj_a.max()
            lo_b, hi_b = proj_b.min(), proj_b.max()
            # gap between intervals; margin = how far apart (negative if overlap)
            gap = max(lo_a, lo_b) - min(hi_a, hi_b)
            # penetration depth contribution = -gap (positive when overlapping)
            if -gap < min_margin:
                min_margin = -gap
    # min_margin is the penetration depth (positive overlap, negative separation)
    return min_margin

def _outer_penetration(p, a, cx, cy, side, rot):
    """Penetration of hexagon (p,a) into the outside of outer hexagon.
    Positive when any vertex is outside the outer hexagon."""
    ov = _outer_verts(cx, cy, side, rot)
    verts = _world_verts(p, a)
    pen = 0.0
    for v in verts:
        d = _signed_dist_point_poly(v[0], v[1], ov)
        if d < 0:
            pen = max(pen, -d)
    return pen

def _evaluate(inner, cx, cy, side, rot):
    """Total violation (sum of penetrations). 0 => feasible."""
    v = 0.0
    for i in range(N):
        v += _outer_penetration(inner[i][0:2], inner[i][2], cx, cy, side, rot)
    for i in range(N):
        for j in range(i + 1, N):
            v += _hex_overlap(inner[i][2], inner[i][0:2], inner[j][2], inner[j][0:2])
    return v

# --- Dynamics ---------------------------------------------------------------
def _dynamics_step(pos, ang, vel, angvel, cx, cy, side, rot, dt, k_rep, k_att, damping,
                   k_outer, k_overlap):
    """One velocity-Verlet step. Returns new pos, ang, vel, angvel."""
    n = len(pos)
    acc = np.zeros((n, 2))
    angacc = np.zeros(n)
    # attractive force toward origin
    for i in range(n):
        acc[i] += -k_att * pos[i]
    # repulsive forces between hexagons + containment
    for i in range(n):
        for j in range(i + 1, n):
            d = pos[i] - pos[j]
            dist = np.hypot(d[0], d[1])
            if dist < 1e-6:
                continue
            # soft repulsion
            f = k_rep / (dist * dist + 1e-4)
            acc[i] += f * d / dist
            acc[j] -= f * d / dist
        # containment: push toward center if outside
        op = _outer_penetration(pos[i], ang[i], cx, cy, side, rot)
        if op > 0:
            d = pos[i] - np.array([cx, cy])
            dist = np.hypot(d[0], d[1])
            if dist > 1e-6:
                acc[i] += k_outer * op * d / dist
            # also nudge rotation to reduce penetration (small)
            angacc[i] += k_outer * 0.01 * op * (1 if np.sin(ang[i]) > 0 else -1)
    # integrate
    pos_new = pos + vel * dt + 0.5 * acc * dt * dt
    vel_new = (vel + 0.5 * acc * dt) * damping
    ang_new = ang + angvel * dt + 0.5 * angacc * dt * dt
    angvel_new = (angvel + 0.5 * angacc * dt) * damping
    return pos_new, ang_new, vel_new, angvel_new

def _dynamics(inner0, cx, cy, side, rot, t_end):
    """Run dynamics; returns (pos, ang, best_inner, best_viol)."""
    n = N
    pos = np.array([inner0[i][0:2] for i in range(n)], dtype=float)
    ang = np.array([inner0[i][2] for i in range(n)], dtype=float)
    vel = np.zeros((n, 2))
    angvel = np.zeros(n)
    dt = 0.002
    k_rep = 5.0
    k_att = 0.5
    k_outer = 30.0
    damping = 0.98
    best_inner = None
    best_viol = np.inf
    t = 0.0
    while t < t_end and time.time() < _DEADLINE:
        pos, ang, vel, angvel = _dynamics_step(pos, ang, vel, angvel, cx, cy, side, rot,
                                               dt, k_rep, k_att, damping, k_outer, 0)
        t += dt
        # periodic violation check
        if int(t / 0.01) % 50 == 0:
            inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(n)]
            v = _evaluate(inner, cx, cy, side, rot)
            if v < best_viol:
                best_viol = v
                best_inner = inner
    if best_inner is None:
        best_inner = [[pos[i][0], pos[i][1], ang[i]] for i in range(n)]
        best_viol = _evaluate(best_inner, cx, cy, side, rot)
    return pos, ang, best_inner, best_viol

# --- Coordinate-descent polish ----------------------------------------------
def _polish(inner, cx, cy, side, rot, t_end):
    """Coordinate-descent on side (and positions) to reduce outer side length."""
    n = N
    best_side = side
    best_inner = [row[:] for row in inner]
    t = 0.0
    step = 0.02
    while t < t_end and time.time() < _DEADLINE:
        improved = False
        # try shrinking side
        new_side = best_side - step
        if new_side > 0.5:
            v = _evaluate(best_inner, cx, cy, new_side, rot)
            if v <= 1e-9:
                best_side = new_side
                improved = True
        # if not shrinking, try fixing by moving hexagons
        if not improved:
            # coordinate descent on positions to reduce violation
            v = _evaluate(best_inner, cx, cy, best_side, rot)
            if v > 1e-9:
                for i in range(n):
                    for dx in (-step, step):
                        for dy in (-step, step):
                            trial = [row[:] for row in best_inner]
                            trial[i][0] += dx
                            trial[i][1] += dy
                            vt = _evaluate(trial, cx, cy, best_side, rot)
                            if vt < v - 1e-12:
                                v = vt
                                best_inner = trial
                    # rotation
                    for da in (-0.05, 0.05):
                        trial = [row[:] for row in best_inner]
                        trial[i][2] += da
                        vt = _evaluate(trial, cx, cy, best_side, rot)
                        if vt < v - 1e-12:
                            v = vt
                            best_inner = trial
            # now try shrinking again
            new_side = best_side - step
            if new_side > 0.5:
                v = _evaluate(best_inner, cx, cy, new_side, rot)
                if v <= 1e-9:
                    best_side = new_side
                    improved = True
        t += 0.01
    return best_inner, best_side

# --- Main construct ---------------------------------------------------------
def construct():
    t0 = time.time()
    cx, cy = 0.0, 0.0
    rot = 0.0
    # start with a moderate side guess; dynamics will find a feasible cluster
    side = 6.0
    best_side = np.inf
    best_inner = None
    best_rot = rot

    # multiple random restarts
    rng = np.random.default_rng(42)
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() - t0 > _DEADLINE - 20:
            break
        # random cluster near origin
        inner = []
        for i in range(N):
            x = rng.uniform(-2, 2)
            y = rng.uniform(-2, 2)
            a = rng.uniform(0, 2 * np.pi)
            inner.append([x, y, a])
        # dynamics to settle
        pos, ang, dyn_inner, viol = _dynamics(inner, cx, cy, side, rot, 0.5)
        # polish
        pol_inner, pol_side = _polish(dyn_inner, cx, cy, side, rot, 15.0)
        if pol_side < best_side:
            best_side = pol_side
            best_inner = pol_inner
            best_rot = rot

    if best_inner is None:
        # fallback
        best_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        best_side = 40.0
    return best_inner, (cx, cy), best_side, best_rot
