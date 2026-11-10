# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_T0 = time.time()
_DEADLINE = 560.0  # safety margin before 600s


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
    all inner hexagons (all inner vertices satisfy the cross-product containment)."""
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
            # Fix: Compute 2D cross product manually (z-component)
            # c = e_x * v_y - e_y * v_x
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


def _objective(x):
    """Objective = outer side length (minimized). Invalid configs -> large penalty."""
    hexes = x.reshape(11, 3)
    if not np.all(np.isfinite(hexes)):
        return 1e6
    if not _separation_ok(hexes):
        return 1e6
    return _outer_side(hexes)


def _nm_refine(x):
    try:
        res = minimize(_objective, x, method='Nelder-Mead',
                       options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-9})
        return res.x
    except Exception:
        return x


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


def hexagon_packing_11():
    # Structural Change: Use lattice-based initialization and multiple random restarts
    # instead of slow differential evolution.
    
    best_x = None
    best_f = 1e6
    
    # Generate a set of initial candidates on a hexagonal lattice
    # This provides a "combinatorial" flavor by discretizing starting positions.
    lattice_points = []
    # Hexagonal lattice spacing approx sqrt(3) for touching hexagons
    # We'll sample a grid of potential centers
    max_radius = 6.0
    step = 0.5
    y = -max_radius
    while y <= max_radius:
        x = -max_radius
        # Offset alternate rows for hex packing
        offset = 0.25 if int(round(y / step)) % 2 else 0.0
        while x <= max_radius:
            if x**2 + y**2 <= max_radius**2:
                lattice_points.append((x + offset, y))
            x += step
        y += step
    
    # Convert to list of (x,y)
    lattice_points = np.array(lattice_points)
    
    # Create 11 initial configurations by picking 11 points from the lattice
    # We'll do this randomly but biased towards dense clusters
    rng = np.random.default_rng(42)
    
    for restart in range(20):
        if (time.time() - _T0) > _DEADLINE - 10:
            break
            
        # Pick 11 points from the lattice
        if len(lattice_points) >= 11:
            indices = rng.choice(len(lattice_points), size=11, replace=False)
            centers = lattice_points[indices]
        else:
            centers = rng.uniform(-5, 5, (11, 2))
            
        # Assign random angles (0, 30, 60 degrees are common symmetries)
        angles = rng.choice([0.0, 30.0, 60.0, 90.0], size=11)
        
        init_x = np.column_stack([centers[:, 0], centers[:, 1], angles]).ravel()
        
        # Refine
        cand = _nm_refine(init_x)
        cf = _objective(cand)
        
        if cf < best_f:
            best_f = cf
            best_x = cand
            
        # If we found a very good solution, we can stop early or tighten search
        if best_f < 4.0: # sqrt(11) is ~3.31, optimal is likely around 3.5-4.0
             # Continue to see if we can improve, but break if time is tight
             pass

    # If no valid solution found, use fallback
    if best_x is None or not np.all(np.isfinite(best_x)):
        return _fallback()
        
    return _build_result(best_x)


# EVOLVE-BLOCK-END
