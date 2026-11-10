# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

_T0 = time.time()
_DEADLINE = 600.0
_EPS = 1e-6
_SQ3 = np.sqrt(3.0)


def _hex_verts(cx, cy, s, a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(60.0 * np.arange(6))
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _face_normals(a_deg):
    a = np.deg2rad(a_deg)
    th = a + np.deg2rad(30.0 + 60.0 * np.arange(6))
    return np.column_stack([np.cos(th), np.sin(th)])


def _pair_gap(va, vb, na, nb):
    """Max gap over the 12 edge normals (6 of A, 6 of B). >0 => separated."""
    best = -np.inf
    for n in (na, nb):
        pa = va @ n
        pb = vb @ n
        g = max(pb.min() - pa.max(), pa.min() - pb.max())
        if g > best:
            best = g
    return best


def _outer_side(shifted, a_deg):
    """Minimal side of a regular hexagon (orientation a_deg, centered at origin)
    containing the point set `shifted`."""
    n = _face_normals(a_deg)
    dots = shifted @ n.T
    r = dots.max()
    return 2.0 * r / _SQ3


def _best_orientation(shifted, n_ang=121):
    best_s = np.inf
    best_a = 0.0
    for a_deg in np.linspace(0.0, 60.0, n_ang):
        s = _outer_side(shifted, a_deg)
        if s < best_s:
            best_s, best_a = s, a_deg
    return best_s, best_a


def _compute_metrics(params):
    """Returns (side, min_gap, verts, norms, cent, shifted, a_deg)."""
    verts = np.empty((11, 6, 2))
    norms = np.empty((11, 6, 2))
    for i in range(11):
        cx, cy, ad = params[3 * i], params[3 * i + 1], params[3 * i + 2]
        verts[i] = _hex_verts(cx, cy, 1.0, ad)
        norms[i] = _face_normals(ad)
    
    min_gap = np.inf
    for i in range(11):
        for j in range(i + 1, 11):
            g = _pair_gap(verts[i], verts[j], norms[i], norms[j])
            if g < min_gap:
                min_gap = g
                
    allv = verts.reshape(-1, 2)
    cent = allv.mean(axis=0)
    shifted = allv - cent
    side, a_deg = _best_orientation(shifted, n_ang=61) # Reduced resolution for speed during search
    return side, min_gap, verts, norms, cent, shifted, a_deg


def _objective_scalar(params):
    side, min_gap, _, _, _, _, _ = _compute_metrics(params)
    if min_gap < _EPS:
        return 1e6 * (_EPS - min_gap) + side
    return side


def _valid(params):
    """Contract validity. Returns (side, outer) if valid else None."""
    side, min_gap, verts, norms, cent, shifted, a_deg = _compute_metrics(params)
    
    # Re-check gap with full precision if it looks close
    if min_gap <= _EPS:
        # It's invalid if any gap is <= EPS
        # The _compute_metrics uses min over all pairs.
        # We need to be sure.
        return None

    if side < np.sqrt(11.0) - 1e-9:
        return None
    
    # Exact containment check at the centroid (contract shifts by outer center).
    ov = _hex_verts(0.0, 0.0, side, a_deg)
    for i in range(6):
        p1 = ov[i]
        p2 = ov[(i + 1) % 6]
        e = p2 - p1
        c = np.cross(e, shifted - p1)
        if c.min() < -1e-6:
            return None
            
    return side, (cent[0], cent[1], a_deg)


def _flower_seed():
    p = [[0.0, 0.0, 0.0]]
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        p.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in [0, 2, 4, 5]:
        ang = np.deg2rad(60.0 * k + 30.0)
        p.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    return np.array(p).flatten()


def _ring_seed():
    p = []
    for k in range(6):
        ang = np.deg2rad(60.0 * k)
        p.append([2.0 * np.cos(ang), 2.0 * np.sin(ang), 0.0])
    for k in [0, 2, 4]:
        ang = np.deg2rad(60.0 * k)
        p.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    for k in [1, 3, 5]:
        ang = np.deg2rad(60.0 * k)
        p.append([4.0 * np.cos(ang), 4.0 * np.sin(ang), 0.0])
    return np.array(p).flatten()


def _band_seed():
    # 3-2-3-2-3 brick-wall band (honeycomb), rows along x.
    # This creates 13 hexagons. We need 11.
    # Let's try 3-2-3-2-1
    h = _SQ3 * 2.0
    rows = [
        (0.0, 3), (h, 2), (2 * h, 3), (3 * h, 2), (4 * h, 1)
    ]
    p = []
    for y, cnt in rows:
        for i in range(cnt):
            p.append([(i - (cnt - 1) / 2.0) * 2.0, y, 0.0])
    return np.array(p).flatten()


def _random_seed():
    # Random positions within a large hexagon
    p = []
    for _ in range(11):
        r = 6.0 * np.sqrt(np.random.rand())
        a = 2.0 * np.pi * np.random.rand()
        p.append([r * np.cos(a), r * np.sin(a), np.random.rand() * 60.0])
    return np.array(p).flatten()


def hexagon_packing_11():
    seeds = [_flower_seed(), _ring_seed(), _band_seed()]
    # Add a few random seeds
    for _ in range(3):
        seeds.append(_random_seed())

    best = None          # (side, inner(11,3), outer(3,))
    best_side = np.inf

    def consider(params):
        nonlocal best, best_side
        v = _valid(params)
        if v is None:
            return
        side, outer = v
        if side < best_side:
            best_side = side
            inner = np.empty((11, 3))
            for i in range(11):
                inner[i] = [params[3 * i] - outer[0],
                            params[3 * i + 1] - outer[1],
                            params[3 * i + 2]]
            best = (side, inner, np.array(outer))

    # Strategy 1: Global Search with Differential Evolution
    # Bounds: x, y in [-10, 10], angle in [0, 60]
    bounds = []
    for _ in range(11):
        bounds.append((-10.0, 10.0))
        bounds.append((-10.0, 10.0))
        bounds.append((0.0, 60.0))

    if time.time() < _DEADLINE - 120.0:
        try:
            res_de = differential_evolution(_objective_scalar, bounds,
                                             maxiter=200, tol=1e-8,
                                             popsize=20, seed=42,
                                             polish=False,
                                             update_interval=10,
                                             polish=True)
            consider(res_de.x)
        except Exception:
            pass

    # Strategy 2: Local Refinement from Seeds
    for seed in seeds:
        if time.time() > _DEADLINE - 30.0:
            break
        try:
            res = minimize(_objective_scalar, seed, method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-8,
                                    'fatol': 1e-8, 'adaptive': True})
            consider(res.x)
        except Exception:
            pass

    if best is None:
        # Fallback: a guaranteed-valid (large) grid packing.
        inner = np.array([
            [0, 0, 0], [-2.5, 0, 0], [2.5, 0, 0],
            [-1.25, 2.17, 0], [1.25, 2.17, 0],
            [-1.25, -2.17, 0], [1.25, -2.17, 0],
            [-3.75, 2.17, 0], [3.75, 2.17, 0],
            [-3.75, -2.17, 0], [3.75, -2.17, 0],
        ], dtype=float)
        return inner, np.array([0.0, 0.0, 0.0]), 8.0

    side, inner, outer = best
    return inner, outer, float(side)


# EVOLVE-BLOCK-END
