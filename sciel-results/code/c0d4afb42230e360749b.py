import time
import math
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 150.0
START = time.time()

# Vertices of a unit regular hexagon (side 1), centered at origin, angle 0.
# Flat-top orientation: vertices at angles 0,60,...,300 degrees at radius 1.
_HEX_VERTS = np.array([[math.cos(math.pi/6*i), math.sin(math.pi/6*i)] for i in range(6)])

def _hexagon_vertices(cx, cy, ang_deg):
    a = math.radians(ang_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]])
    return _HEX_VERTS @ R.T + np.array([cx, cy])

def _point_in_hexagon(px, py, S):
    # Regular hexagon centered at origin, side S, flat-top (vertices at 0,60,...)
    # It's the intersection of 6 half-planes. Inradius = S*sqrt(3)/2.
    # For flat-top hexagon, the 6 normal directions are at 30,90,150,210,270,330 degrees.
    # Actually for vertices at 0,60,120,180,240,300 (pointy-top-ish), the edge normals
    # are at 30,90,150,210,270,330. Let's use the standard: a regular hexagon with
    # "radius" (circumradius) R_c = S (since side = circumradius for regular hexagon).
    # The 6 faces have outward normals at angles 30+60k, and the support distance is
    # the apothem = S * sqrt(3)/2.
    apothem = S * math.sqrt(3.0) / 2.0
    # normals at 30,90,150,210,270,330
    for k in range(6):
        ang = math.radians(30 + 60 * k)
        nx, ny = math.cos(ang), math.sin(ang)
        if px * nx + py * ny > apothem + 1e-9:
            return False
    return True

def _hexagon_fits_inside(px, py, ang_deg, S):
    verts = _hexagon_vertices(px, py, ang_deg)
    for (vx, vy) in verts:
        if not _point_in_hexagon(vx, vy, S):
            return False
    return True

def _min_S_for_hex(px, py, ang_deg):
    # Minimum S such that the hexagon at (px,py,ang) fits in centered hexagon side S.
    verts = _hexagon_vertices(px, py, ang_deg)
    S = 0.0
    for (vx, vy) in verts:
        # distance in each of the 6 face-normal directions
        for k in range(6):
            ang = math.radians(30 + 60 * k)
            d = vx * math.cos(ang) + vy * math.sin(ang)
            apothem = d  # need S*sqrt(3)/2 >= d
            S = max(S, 2.0 * d / math.sqrt(3.0))
    return max(S, 0.0)

def _objective(params):
    # params: [x0,y0,a0, x1,y1,a1, ..., x11,y11,a11, S]
    S = params[-1]
    if S <= 0:
        S = 1e-6
    # Penalty if any hexagon doesn't fit
    pen = 0.0
    for i in range(N):
        px, py, ang = params[3*i], params[3*i+1], params[3*i+2]
        if not _hexagon_fits_inside(px, py, ang, S):
            # compute how much S would need to be
            needed = _min_S_for_hex(px, py, ang)
            pen += max(0.0, needed - S)
    # Also penalize overlaps softly to encourage separation (helps optimizer)
    # We'll add a small pairwise repulsion to avoid degenerate overlaps
    rep = 0.0
    for i in range(N):
        for j in range(i+1, N):
            dx = params[3*i] - params[3*j]
            dy = params[3*i+1] - params[3*j+1]
            d = math.sqrt(dx*dx + dy*dy)
            # minimum distance for two unit hexagons not to overlap is roughly 2 (if aligned)
            # use a soft lower bound of ~1.7 to push apart
            if d < 1.7:
                rep += (1.7 - d) ** 2
    return S + 10.0 * pen + 0.5 * rep

def _initial_guess():
    # 3 rows x 4 columns grid, hexagon "radius" 1, so horizontal spacing 2, vertical spacing sqrt(3)
    xs = [-3.0, -1.0, 1.0, 3.0]
    ys = [-math.sqrt(3), 0.0, math.sqrt(3)]
    pts = []
    for y in ys:
        for x in xs:
            pts.append([x, y, 0.0])
    # initial S estimate
    S = 10.0
    return np.array([v for p in pts for v in p] + [S])

def _refine(x0, time_budget):
    best = x0.copy()
    best_val = _objective(x0)
    t_end = time.time() + time_budget
    # Try a few optimization methods / restarts
    x = x0.copy()
    for method in ['Nelder-Mead', 'Powell', 'Nelder-Mead']:
        if time.time() > t_end - 5:
            break
        try:
            res = minimize(_objective, x, method=method,
                           options={'maxiter': 2000, 'xatol': 1e-5, 'fatol': 1e-7})
            if res.success or res.fun < best_val:
                x = res.x
                best_val = res.fun
                best = x.copy()
        except Exception:
            pass
    return best, best_val

def _validate_and_build(params):
    # Return a valid packing: adjust S to be the max needed
    S = params[-1]
    for i in range(N):
        px, py, ang = params[3*i], params[3*i+1], params[3*i+2]
        S = max(S, _min_S_for_hex(px, py, ang))
    inner = [[float(params[3*i]), float(params[3*i+1]), float(params[3*i+2])] for i in range(N)]
    return inner, (0.0, 0.0), float(S), 0.0

def construct():
    t_end = START + DEADLINE - 10.0  # safety margin
    # Initial guess
    x0 = _initial_guess()
    # Evaluate
    val = _objective(x0)
    best_params = x0.copy()
    best_val = val

    # Refine with a time budget
    remaining = t_end - time.time()
    if remaining > 5:
        p, v = _refine(x0, remaining - 5)
        if v < best_val:
            best_params = p
            best_val = v

    # Final safety: ensure feasibility by recomputing S
    inner, center, S, ang = _validate_and_build(best_params)
    return inner, center, S, ang
