# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import linprog, minimize

_T0 = time.time()
_DEADLINE = 570.0  # safety margin before 600s


def _hex_vertices(cx, cy, s, a):
    """Vertices of a regular hexagon (center cx,cy, side s, angle a in radians)."""
    th = a + np.arange(6) * (np.pi / 3.0)
    return np.column_stack([cx + s * np.cos(th), cy + s * np.sin(th)])


def _separation_ok(hexes):
    """Return True if every pair of unit hexagons is strictly separated (>1e-6 gap
    on at least one of the 12 edge normals)."""
    n = len(hexes)
    for i in range(n):
        vi = _hex_vertices(hexes[i, 0], hexes[i, 1], 1.0, np.deg2rad(hexes[i, 2]))
        for j in range(i + 1, n):
            vj = _hex_vertices(hexes[j, 0], hexes[j, 1], 1.0, np.deg2rad(hexes[j, 2]))
            ok = False
            for k in range(6):
                p1, p2 = vi[k], vi[(k + 1) % 6]
                edge = p2 - p1
                nrm = np.array([-edge[1], edge[0]])
                nrm /= (np.linalg.norm(nrm) + 1e-15)
                a1 = np.dot(vi, nrm)
                a2 = np.dot(vj, nrm)
                if (a1.max() < a2.min() - 1e-6) or (a2.max() < a1.min() - 1e-6):
                    ok = True
                    break
            if not ok:
                return False
    return True


def _outer_side(hexes):
    """Minimal side length of a regular hexagon (center 0, angle 0) that contains
    all inner hexagons."""
    vs = []
    for h in hexes:
        vs.append(_hex_vertices(h[0], h[1], 1.0, np.deg2rad(h[2])))
    vs = np.vstack(vs)
    lo = 0.0
    hi = 200.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        oev = _hex_vertices(0.0, 0.0, mid, 0.0)
        ok = True
        for k in range(6):
            p1, p2 = oev[k], oev[(k + 1) % 6]
            e = p2 - p1
            rel = vs - p1
            c = e[0] * rel[:, 1] - e[1] * rel[:, 0]
            if np.any(c < -1e-6):
                ok = False
                break
        if ok:
            hi = mid
        else:
            lo = mid
    return hi


def _build_result(hexes):
    hexes = np.asarray(hexes, dtype=float).reshape(11, 3)
    side = _outer_side(hexes)
    inner = hexes.copy()
    outer = np.array([0.0, 0.0, 0.0])
    return inner, outer, float(side)


def _fallback():
    inner = np.array([
        [0.0, 0.0, 0.0],
        [-2.5, 0.0, 0.0], [2.5, 0.0, 0.0],
        [-1.25, 2.17, 0.0], [1.25, 2.17, 0.0],
        [-1.25, -2.17, 0.0], [1.25, -2.17, 0.0],
        [-3.75, 2.17, 0.0], [3.75, 2.17, 0.0],
        [-3.75, -2.17, 0.0], [3.75, -2.17, 0.0],
    ])
    return _build_result(inner)


def _flower_config():
    """Classic 1+6 flower arrangement of 7 hexagons, plus 4 more."""
    # Center hexagon
    centers = [[0.0, 0.0, 0.0]]
    # 6 surrounding hexagons at distance ~2.5 (for unit hexagons, touching distance is ~2.5)
    for i in range(6):
        angle = np.deg2rad(60.0 * i)
        cx = 2.5 * np.cos(angle)
        cy = 2.5 * np.sin(angle)
        centers.append([cx, cy, 0.0])
    # 4 additional hexagons in a ring
    for i in range(4):
        angle = np.deg2rad(60.0 * i + 30.0)
        cx = 4.33 * np.cos(angle)
        cy = 4.33 * np.sin(angle)
        centers.append([cx, cy, 0.0])
    return np.array(centers)


def _ring_config():
    """Ring of 10 hexagons around a center."""
    centers = [[0.0, 0.0, 0.0]]
    for i in range(10):
        angle = np.deg2rad(36.0 * i)
        cx = 2.5 * np.cos(angle)
        cy = 2.5 * np.sin(angle)
        centers.append([cx, cy, 0.0])
    return np.array(centers)


def _line_config():
    """Linear arrangement."""
    centers = []
    for i in range(11):
        centers.append([(i - 5) * 2.5, 0.0, 0.0])
    return np.array(centers)


def _compact_config():
    """Try to pack more compactly using hexagonal lattice."""
    centers = []
    # Use hexagonal lattice with spacing sqrt(3) ~ 1.732 for centers
    # Actually for unit hexagons, center-to-center distance for touching is 2*sqrt(3)/2 * 2 = 2*sqrt(3) ~ 3.464? 
    # No, for regular hexagons with side 1, the distance between centers of adjacent hexagons is 2*apothem = 2*(sqrt(3)/2) = sqrt(3) ~ 1.732? 
    # Wait, apothem = sqrt(3)/2, so distance between parallel sides is sqrt(3). For touching, center distance = 2*apothem = sqrt(3) ~ 1.732? 
    # Actually no. The distance from center to vertex is 1 (circumradius = side length for regular hexagon). 
    # The distance between centers of two touching hexagons (edge to edge) is 2*apothem = 2*(sqrt(3)/2) = sqrt(3) ~ 1.732.
    # But if they touch vertex to vertex, it could be different.
    # Let's use a standard hexagonal packing.
    dx = 2.5  # approximate
    dy = 2.17  # ~2.5*sqrt(3)/2
    positions = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1), (2, 0), (0, 2)
    ]
    for px, py in positions:
        cx = px * dx + (0.5 * dx if py % 2 else 0)
        cy = py * dy
        centers.append([cx, cy, 0.0])
    return np.array(centers)


def _objective(x):
    """Objective = outer side length (minimized). Invalid configs -> large penalty."""
    hexes = x.reshape(11, 3)
    if not np.all(np.isfinite(hexes)):
        return 1e6
    if not _separation_ok(hexes):
        return 1e6
    return _outer_side(hexes)


def _nm_refine(x, maxiter=500):
    try:
        res = minimize(_objective, x, method='Nelder-Mead',
                       options={'maxiter': maxiter, 'xatol': 1e-7, 'fatol': 1e-8})
        return res.x
    except Exception:
        return x


def hexagon_packing_11():
    best_x = None
    best_f = 1e6
    
    # Try multiple initial configurations
    configs = [
        _flower_config(),
        _ring_config(),
        _line_config(),
        _compact_config(),
    ]
    
    rng = np.random.default_rng(12345)
    
    for cfg in configs:
        if (time.time() - _T0) > _DEADLINE - 10:
            break
        
        # Add small random perturbations to break symmetry
        for trial in range(5):
            if (time.time() - _T0) > _DEADLINE - 10:
                break
            
            perturbed = cfg.copy()
            perturbed[:, :2] += rng.normal(0, 0.1, (11, 2))
            # Keep angles mostly 0, with some variation
            perturbed[:, 2] = rng.choice([0.0, 30.0, 60.0], size=11) * 0.1
            
            init_x = perturbed.ravel()
            cand = _nm_refine(init_x, maxiter=300)
            cf = _objective(cand)
            
            if cf < best_f:
                best_f = cf
                best_x = cand
    
    # If we have time, do more random restarts
    if best_f > 3.5 and (time.time() - _T0) < _DEADLINE - 60:
        for restart in range(10):
            if (time.time() - _T0) > _DEADLINE - 10:
                break
            
            # Random initial configuration
            centers = rng.uniform(-5, 5, (11, 2))
            angles = rng.choice([0.0, 30.0, 60.0], size=11)
            init_x = np.column_stack([centers, angles]).ravel()
            
            cand = _nm_refine(init_x, maxiter=400)
            cf = _objective(cand)
            
            if cf < best_f:
                best_f = cf
                best_x = cand
    
    # Final refinement if we have time
    if best_x is not None and (time.time() - _T0) < _DEADLINE - 30:
        refined = _nm_refine(best_x, maxiter=1000)
        rf = _objective(refined)
        if rf < best_f:
            best_f = rf
            best_x = refined
    
    if best_x is None or not np.all(np.isfinite(best_x)):
        return _fallback()
        
    return _build_result(best_x)


# EVOLVE-BLOCK-END
