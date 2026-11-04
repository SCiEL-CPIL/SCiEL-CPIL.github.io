"""Packing 12 unit hexagons in a minimal regular hexagon via lattice init + Nelder-Mead polish."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
_START = time.time()
_DEADLINE = 118.0

# ---------- geometry helpers ----------
def _hex_verts(cx, cy, a, s=1.0):
    a = float(a)
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[s*ca - 0.5*s*sa, s*sa + 0.5*s*ca] for _ in [0]]) + np.array([cx, cy])

def _hex_pts(cx, cy, a, s=1.0):
    a = float(a)
    base = np.array([[math.cos(k*math.pi/3.0), math.sin(k*math.pi/3.0)] for k in range(6)])
    ca, sa = math.cos(a), math.sin(a)
    R = np.array([[ca, -sa], [sa, ca]])
    return (base * s) @ R.T + np.array([cx, cy])

def _rot(p, ang):
    ca, sa = math.cos(ang), math.sin(ang)
    return np.array([[ca, -sa], [sa, ca]]) @ p

def _sdf_hexagon(p, a, s=1.0):
    """Signed distance from point p to regular hexagon (center 0, circumradius s, rotation a).
    Returns positive outside, negative inside."""
    p = _rot(p, -a)
    # Hexagon with circumradius s: vertices at angles 0,60,...; edges at distance s*sqrt(3)/2
    # For a regular hexagon, |p| <= s and the 3 "slab" constraints:
    # distance to each of 3 edge-lines (normals at 0,120,240 deg)
    d_edge = s * math.sqrt(3.0) / 2.0
    # normals
    n = np.array([[1.0, 0.0],
                  [math.cos(2*math.pi/3.0), math.sin(2*math.pi/3.0)],
                  [math.cos(4*math.pi/3.0), math.sin(4*math.pi/3.0)]])
    proj = p @ n.T  # 3 values
    # inside if all proj <= d_edge and |p| <= s (the vertex caps)
    max_proj = np.max(proj)
    r = float(np.linalg.norm(p))
    # Approximate SDF: max of (max_proj - d_edge) and (r - s) but that overestimates at corners
    # Better: use the max of the 3 edge distances and the radial cap
    return max(max_proj - d_edge, r - s)

def _sdf_hexagon_to_hexagon(a_pts, b_pts, a_ang, b_ang):
    """Approx SDF between two hexagons using sampled boundary points."""
    # sample points on hexagon A boundary, find min SDF into B
    n = 12
    # build A boundary samples: for each edge, sample points
    verts_a = _hex_pts(0.0, 0.0, a_ang)
    verts_b = _hex_pts(0.0, 0.0, b_ang)
    samples = []
    for i in range(6):
        v1 = verts_a[i]; v2 = verts_a[(i+1) % 6]
        for t in np.linspace(0, 1, n, endpoint=False):
            samples.append(v1 + t*(v2 - v1))
    samples = np.array(samples)
    # SDF of each sample into hexagon B (centered at 0)
    d = np.array([_sdf_hexagon(s, b_ang) for s in samples])
    return float(np.min(d))

# ---------- objective ----------
def _objective(theta, outer_ang, outer_side):
    """theta: [x0,y0,ang0, x1,y1,ang1, ...] (6N). Returns outer side needed."""
    xs = theta[0::3]
    ys = theta[1::3]
    angs = theta[2::3]
    centers = np.column_stack([xs, ys])
    # outer hexagon: center 0, rotation outer_ang, side length s -> circumradius R = s
    R = outer_side
    # For a point to be inside outer hexagon (center 0, rotation outer_ang, circumradius R):
    # 6 constraints: for each edge normal (at outer_ang + k*60), proj <= R*sqrt(3)/2, and radial cap R
    n_out = np.array([[math.cos(outer_ang + k*math.pi/3.0),
                       math.sin(outer_ang + k*math.pi/3.0)] for k in range(6)])
    d_edge = R * math.sqrt(3.0) / 2.0
    # containment violation: max over points, over (proj - d_edge, |p| - R)
    # We'll compute per-hexagon max violation of its 6 vertices into outer
    total_out = 0.0
    for i in range(N):
        verts = _hex_pts(xs[i], ys[i], angs[i])
        proj = verts @ n_out.T  # 6x6
        max_proj = np.max(proj)
        r = np.max(np.linalg.norm(verts, axis=1))
        total_out = max(total_out, max_proj - d_edge, r - R)
    # pairwise collision
    total_col = 0.0
    for i in range(N):
        for j in range(i+1, N):
            rel = centers[j] - centers[i]
            # translate so A at 0
            d = _sdf_hexagon_to_hexagon(_hex_pts(0,0,angs[i]), _hex_pts(0,0,angs[j]), angs[i], angs[j])
            # SDF is for A at 0, B at 0; actual relative distance: we need min over A boundary of SDF into B translated by -rel
            # Approx: use relative offset by shifting B
            # Recompute properly:
            d = _sdf_pair(centers[i], angs[i], centers[j], angs[j])
            total_col = max(total_col, -d)  # negative SDF means overlap
    # We want both total_out <= 0 and total_col <= 0. Return a scalar side.
    # Feasibility via penalty: return R + penalty
    pen = max(0.0, total_out) + 5.0 * max(0.0, total_col)
    return R + pen

def _sdf_pair(ci, ai, cj, aj):
    """SDF between hexagon i (center ci, ang ai) and hexagon j (center cj, ang aj). Positive if separated."""
    rel = ci - cj  # shift so j at 0
    # sample i boundary, SDF into j (center 0)
    verts_i = _hex_pts(0.0, 0.0, ai)
    n = 10
    samples = []
    for k in range(6):
        v1 = verts_i[k]; v2 = verts_i[(k+1) % 6]
        for t in np.linspace(0, 1, n, endpoint=False):
            samples.append(v1 + t*(v2 - v1))
    samples = np.array(samples) + rel
    d = np.array([_sdf_hexagon(s, aj) for s in samples])
    return float(np.min(d))

def _objective_fast(theta, outer_ang, outer_side):
    xs = theta[0::3]
    ys = theta[1::3]
    angs = theta[2::3]
    centers = np.column_stack([xs, ys])
    R = outer_side
    n_out = np.array([[math.cos(outer_ang + k*math.pi/3.0),
                       math.sin(outer_ang + k*math.pi/3.0)] for k in range(6)])
    d_edge = R * math.sqrt(3.0) / 2.0
    total_out = 0.0
    for i in range(N):
        verts = _hex_pts(xs[i], ys[i], angs[i])
        proj = verts @ n_out.T
        max_proj = np.max(proj)
        r = np.max(np.linalg.norm(verts, axis=1))
        total_out = max(total_out, max_proj - d_edge, r - R)
    total_col = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sdf_pair(centers[i], angs[i], centers[j], angs[j])
            total_col = max(total_col, -d)
    pen = max(0.0, total_out) + 5.0 * max(0.0, total_col)
    return R + pen

# ---------- initial lattice ----------
def _make_initial():
    # hex lattice, spacing 2.0 (touching), 12 points in a 3x4 block centered
    # offsets in lattice coords
    offs = [(-1, -1), (0, -1), (1, -1),
            (-1,  0), (0,  0), (1,  0),
            (-1,  1), (0,  1), (1,  1),
            (-2,  0), (2,  0), (0,  2)]
    pts = []
    for (i, j) in offs:
        x = 2.0 * (i + 0.5 * j)
        y = 2.0 * (math.sqrt(3.0)/2.0) * j
        pts.append((x, y))
    # center
    cx = np.mean([p[0] for p in pts])
    cy = np.mean([p[1] for p in pts])
    # a few 30-degree rotations for diversity
    angs = [0.0]*N
    angs[4] = math.radians(30.0)
    angs[8] = math.radians(30.0)
    theta = np.zeros(3*N)
    for k in range(N):
        theta[3*k]   = pts[k][0] - cx
        theta[3*k+1] = pts[k][1] - cy
        theta[3*k+2] = angs[k]
    return theta

def _outer_side_for(theta, outer_ang):
    """Minimal outer side (circumradius) to contain all hexagons given positions/rotations."""
    xs = theta[0::3]; ys = theta[1::3]; angs = theta[2::3]
    # binary search on R
    lo, hi = 1.0, 20.0
    for _ in range(40):
        mid = (lo + hi) / 2.0
        if _fits(theta, outer_ang, mid):
            hi = mid
        else:
            lo = mid
    return hi

def _fits(theta, outer_ang, R):
    xs = theta[0::3]; ys = theta[1::3]; angs = theta[2::3]
    n_out = np.array([[math.cos(outer_ang + k*math.pi/3.0),
                       math.sin(outer_ang + k*math.pi/3.0)] for k in range(6)])
    d_edge = R * math.sqrt(3.0) / 2.0
    for i in range(N):
        verts = _hex_pts(xs[i], ys[i], angs[i])
        proj = verts @ n_out.T
        if np.max(proj) > d_edge + 1e-9:
            return False
        if np.max(np.linalg.norm(verts, axis=1)) > R + 1e-9:
            return False
    # collisions
    centers = np.column_stack([xs, ys])
    for i in range(N):
        for j in range(i+1, N):
            if _sdf_pair(centers[i], angs[i], centers[j], angs[j]) < -1e-7:
                return False
    return True

def construct():
    global _START, _DEADLINE
    theta0 = _make_initial()
    best_theta = theta0.copy()
    best_side = 100.0
    best_ang = 0.0

    # Try a few outer rotations
    for outer_ang in [0.0, math.radians(30.0)]:
        # initial side estimate
        R = _outer_side_for(theta0, outer_ang)
        if R < best_side:
            best_side = R; best_theta = theta0.copy(); best_ang = outer_ang

        # Nelder-Mead polish on theta with objective = R(theta) + collision penalty
        # Use a fixed R for the penalty to keep objective smooth; optimize R via outer loop
        # Simpler: optimize theta with objective = _objective_fast at R=best_side, then recompute R
        def obj(th):
            if time.time() > _DEADLINE:
                return 1e9
            return _objective_fast(th, outer_ang, best_side)
        try:
            res = minimize(obj, theta0.copy(), method='Nelder-Mead',
                           options={'maxiter': 400, 'xatol': 1e-5, 'fatol': 1e-6})
            if time.time() > _DEADLINE:
                break
            R2 = _outer_side_for(res.x, outer_ang)
            if R2 < best_side:
                best_side = R2; best_theta = res.x.copy(); best_ang = outer_ang
        except Exception:
            pass

    # Build inner list
    inner = []
    for i in range(N):
        x = best_theta[3*i]; y = best_theta[3*i+1]; a = best_theta[3*i+2]
        inner.append([float(x), float(y), float(math.degrees(a))])
    return inner, (0.0, 0.0), float(best_side), float(math.degrees(best_ang))
