import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 20.0

def _hex_verts(cx, cy, ang, side=1.0):
    a = ang + np.arange(6) * np.pi / 3.0
    return np.column_stack((cx + side * np.cos(a), cy + side * np.sin(a)))

def _container_side_and_center(pts, beta):
    """Return (S, cx, cy) for minimal enclosing regular hexagon of orientation beta."""
    if len(pts) == 0:
        return 0.0, 0.0, 0.0
    angles = beta + np.arange(6) * np.pi / 3.0
    normals = np.column_stack((np.cos(angles), np.sin(angles)))
    proj = pts @ normals.T
    mx = proj.max(axis=0)
    mn = proj.min(axis=0)
    width = mx - mn
    S = width.max() / math.sqrt(3)
    # Center: least-squares solve normals^T @ [cx,cy]^T = mid
    mid = (mx + mn) / 2.0
    A = normals.T
    try:
        c = np.linalg.lstsq(A, mid, rcond=None)[0]
    except Exception:
        c = pts.mean(axis=0)
    return S, float(c[0]), float(c[1])

def _overlap_penalty(positions, rotations, thresh=2.0):
    pen = 0.0
    n = len(positions)
    for i in range(n):
        for j in range(i + 1, n):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            d = math.hypot(dx, dy)
            if d < thresh:
                pen += (thresh - d) ** 2
    return pen

def _build_positions(params):
    pos = np.empty((N, 2))
    rot = np.empty(N)
    for i in range(N):
        pos[i, 0] = params[3 * i]
        pos[i, 1] = params[3 * i + 1]
        rot[i] = params[3 * i + 2]
    return pos, rot

def _objective(params):
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return 1e6
    pos, rot = _build_positions(params)
    beta = params[3 * N]
    verts = np.vstack([_hex_verts(pos[i, 0], pos[i, 1], rot[i]) for i in range(N)])
    S, _, _ = _container_side_and_center(verts, beta)
    pen = _overlap_penalty(pos, rot)
    return S + 20.0 * pen

def _axial_to_cart(q, r):
    return (math.sqrt(3) * (q + r / 2.0), 1.5 * r)

def _initial_guesses():
    guesses = []
    # Guess A: 3x3 block + 2 extras on the ring
    axial = [(0, 0), (1, 0), (2, 0),
             (0, 1), (1, 1), (2, 1),
             (0, 2), (1, 2), (2, 2),
             (-1, 1), (3, 1)]
    p = []
    for q, r in axial:
        x, y = _axial_to_cart(q, r)
        p.extend([x, y, 0.0])
    p.append(0.0)  # beta
    guesses.append(np.array(p))

    # Guess B: centered flower + 4 outer
    p = [0.0, 0.0, 0.0]
    for k in range(6):
        a = k * np.pi / 3.0
        x = math.sqrt(3) * math.cos(a)
        y = math.sqrt(3) * math.sin(a)
        p.extend([x, y, 0.0])
    # 4 more at distance ~3.0 in 45-deg-ish directions
    for k in range(4):
        a = (k + 0.5) * np.pi / 2.0
        x = 3.0 * math.cos(a)
        y = 3.0 * math.sin(a)
        p.extend([x, y, 0.0])
    p.append(0.0)
    guesses.append(np.array(p))

    # Guess C: linear chain with slight rotation to fill a strip
    p = []
    for i in range(N):
        x = i * math.sqrt(3)
        y = (0 if i % 2 == 0 else 1.5)
        p.extend([x, y, 0.0])
    p.append(0.0)
    guesses.append(np.array(p))

    # Random perturbations
    rng = np.random.default_rng(123)
    for _ in range(6):
        base = guesses[0].copy()
        for i in range(N):
            base[3 * i] += rng.normal(0, 0.15)
            base[3 * i + 1] += rng.normal(0, 0.15)
            base[3 * i + 2] += rng.normal(0, 0.05)
        guesses.append(base)

    return guesses

def _refine_container(pos, rot, beta0):
    verts = np.vstack([_hex_verts(pos[i, 0], pos[i, 1], rot[i]) for i in range(N)])
    def f(p):
        b, cx, cy = p
        angles = b + np.arange(6) * np.pi / 3.0
        nrm = np.column_stack((np.cos(angles), np.sin(angles)))
        pc = verts - np.array([cx, cy])
        proj = pc @ nrm.T
        return (proj.max(axis=0) - proj.min(axis=0)).max() / math.sqrt(3)
    best_S, best_cx, best_cy = _container_side_and_center(verts, beta0)
    best_p = (beta0, best_cx, best_cy)
    for b in (beta0, 0.0, math.pi / 6, math.pi / 3):
        try:
            r = minimize(f, [b, best_cx, best_cy], method='Nelder-Mead',
                         options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10})
            if r.fun < best_S:
                best_S = r.fun
                best_p = r.x
        except Exception:
            pass
    return best_S, float(best_p[1]), float(best_p[2]), float(best_p[0])

def construct():
    guesses = _initial_guesses()
    best_S = float('inf')
    best_params = None

    for guess in guesses:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        try:
            r = minimize(_objective, guess, method='Nelder-Mead',
                         options={'maxiter': 800, 'xatol': 1e-7, 'fatol': 1e-9})
            pos, rot = _build_positions(r.x)
            pen = _overlap_penalty(pos, rot)
            if pen < 1e-4 and r.fun < best_S:
                best_S = r.fun
                best_params = r.x.copy()
        except Exception:
            continue

    if best_params is None:
        best_params = guesses[0].copy()

    pos, rot = _build_positions(best_params)
    beta0 = float(best_params[3 * N])
    S, cx, cy, beta = _refine_container(pos, rot, beta0)

    inner = [[float(pos[i, 0]), float(pos[i, 1]), float(math.degrees(rot[i]))] for i in range(N)]
    center = [float(cx), float(cy)]
    side = float(S)
    angle_deg = float(math.degrees(beta))
    return inner, center, side, angle_deg
