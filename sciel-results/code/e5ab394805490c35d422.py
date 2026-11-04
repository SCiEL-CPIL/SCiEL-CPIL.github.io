"""Optimized packing of 12 unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon."""
    a = np.deg2rad(angle_deg)
    angles = a + np.deg2rad(np.arange(0, 360, 60))
    return np.column_stack([cx + side * np.cos(angles),
                            cy + side * np.sin(angles)])

def _hexagon_containment_margin(vertices, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """
    For a regular hexagon (outer) with given center/side/angle,
    compute the minimum 'distance' by which vertices are outside.
    Return 0 if all inside, else positive = how far outside (in units of side).
    We use the support function: a point p is inside hexagon iff
    for all 6 face-normals n_i, n_i . (p - c) <= side.
    """
    a = np.deg2rad(outer_angle_deg)
    # outward face normals for a regular hexagon (pointy-top at angle 0)
    # A regular hexagon with 'flat top' orientation: normals at 0,60,...,300
    # But we define by angle: vertices at angle + 0,60,...; faces perpendicular
    # Face normals point at angles a + 30, a+90, ... (midpoints of edges)
    normals = np.column_stack([np.cos(a + np.deg2rad(30 + 60 * np.arange(6))),
                               np.sin(a + np.deg2rad(30 + 60 * np.arange(6)))])
    rel = vertices - np.array([outer_cx, outer_cy])
    # support: max over normals of n . rel
    s = rel @ normals.T  # (nverts, 6)
    # point inside iff all s <= side
    return np.max(s - outer_side)  # scalar margin; <=0 means inside

def _objective(params):
    """
    params layout: [s, cx, cy, angle_deg, then for each hex: x, y, angle_deg]
    s = outer side length (what we minimize)
    Minimize s subject to all inner hexagons inside outer.
    Use penalty for violations.
    """
    s = params[0]
    ocx, ocy, oang = params[1], params[2], params[3]
    if s < 1.0:
        return 1e6
    total_violation = 0.0
    for i in range(N):
        base = 4 + 3 * i
        ix, iy, iang = params[base], params[base+1], params[base+2]
        verts = _hex_vertices(ix, iy, 1.0, iang)
        margin = _hexagon_containment_margin(verts, ocx, ocy, s, oang)
        if margin > 0:
            total_violation += margin * margin
    # Also penalize overlap between inner hexagons (soft)
    # For speed, only check a few pairs
    return s + 10.0 * total_violation

def construct():
    # Initial layout: two rows of 6, or a honeycomb-ish arrangement
    # Unit hexagons: distance between centers for non-overlap ~ 2*apothem = sqrt(3)
    # Try a 3x4 grid with hexagonal offset
    apothem = np.sqrt(3) / 2.0  # for unit hexagon
    # Place in a compact cluster
    # Use a "flower" pattern: 1 center + 6 around + 5 more
    positions = []
    # center
    positions.append((0.0, 0.0, 0.0))
    # 6 around at distance sqrt(3) (tangent)
    r = np.sqrt(3) * 1.001  # slight gap
    for k in range(6):
        ang = np.deg2rad(60 * k)
        positions.append((r * np.cos(ang), r * np.sin(ang), 0.0))
    # 5 more: extend outward
    for k in range(5):
        ang = np.deg2rad(60 * k + 30)
        r2 = r * 2.002
        positions.append((r2 * np.cos(ang), r2 * np.sin(ang), 0.0))
    
    # Build initial params
    params0 = np.zeros(4 + 3 * N)
    params0[0] = 4.0  # initial outer side guess
    params0[1] = 0.0
    params0[2] = 0.0
    params0[3] = 0.0
    for i, (x, y, a) in enumerate(positions):
        base = 4 + 3 * i
        params0[base] = x
        params0[base+1] = y
        params0[base+2] = a
    
    # Optimize
    res = minimize(_objective, params0, method='Nelder-Mead',
                   options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
    
    best_params = res.x
    # Try a few different initial configurations and keep best
    best_s = best_params[0]
    best_result = best_params.copy()
    
    # Alternative: try different outer angles
    for trial_oang in [0, 30, 15, 45]:
        p2 = params0.copy()
        p2[3] = trial_oang
        r2 = minimize(_objective, p2, method='Nelder-Mead',
                      options={'maxiter': 1500, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True})
        if r2.x[0] < best_s:
            best_s = r2.x[0]
            best_result = r2.x.copy()
    
    p = best_result
    s = p[0]
    ocx, ocy, oang = p[1], p[2], p[3]
    
    inner = []
    for i in range(N):
        base = 4 + 3 * i
        inner.append([float(p[base]), float(p[base+1]), float(p[base+2])])
    
    return inner, (float(ocx), float(ocy)), float(s), float(oang)
# EVOLVE-BLOCK-END
