"""Gradient-based packing of unit hexagons in a regular hexagon."""
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, s, rot_deg):
    """Return 6 vertices of a regular hexagon (side s) centered at (cx,cy), rotated."""
    rot = np.radians(rot_deg)
    c, s_ = np.cos(rot), np.sin(rot)
    # vertices of unit hexagon: at angles 0,60,...,300 with radius = s (circumradius = side for regular hex)
    base = s * np.array([[np.cos(np.radians(a)), np.sin(np.radians(a))] for a in range(0, 360, 60)])
    R = np.array([[c, -s_], [s_, c]])
    return cx + R @ base[:, 0], cy + R @ base[:, 1]

def _signed_dist_to_hex(px, py, cx, cy, S, rot_deg):
    """Minimum signed distance from point (px,py) to the boundary of outer hexagon.
    Positive if inside. Uses the 6 side normals."""
    rot = np.radians(rot_deg)
    c, s_ = np.cos(rot), np.sin(rot)
    # Side normals of a regular hexagon: directions at 30, 90, 150, 210, 270, 330 degrees
    # For a regular hexagon with side S, apothem = S * sqrt(3)/2
    apothem = S * np.sqrt(3.0) / 2.0
    angles = np.radians(np.array([30, 90, 150, 210, 270, 330]))
    normals = np.array([[np.cos(a), np.sin(a)] for a in angles])
    # Rotate normals
    R = np.array([[c, -s_], [s_, c]])
    normals = (R @ normals.T).T
    # Vector from center to point
    vx, vy = px - cx, py - cy
    dists = normals[:, 0] * vx + normals[:, 1] * vy
    # The point is inside if all dists <= apothem; clearance = apothem - max(dists)
    return apothem - np.max(dists)

def _objective(params):
    """Negative of minimum clearance (we minimize -clearance = maximize clearance).
    params: [S, cx, cy, rot_deg, x1,y1,r1, x2,y2,r2, ...]"""
    S, cx, cy, rot_deg = params[0], params[1], params[2], params[3]
    min_clear = np.inf
    for i in range(N):
        ox = 4 + 3 * i
        ix, iy, ir = params[ox], params[ox+1], params[ox+2]
        vx, vy = _hex_vertices(ix, iy, 1.0, ir)
        for j in range(6):
            d = _signed_dist_to_hex(vx[j], vy[j], cx, cy, S, rot_deg)
            if d < min_clear:
                min_clear = d
    # Also enforce S > 0
    if S < 1.0:
        return 1e6
    return -min_clear

def _grad_objective(params):
    """Numerical gradient."""
    eps = 1e-5
    g = np.zeros_like(params)
    base = _objective(params)
    for i in range(len(params)):
        p2 = params.copy()
        p2[i] += eps
        g[i] = (_objective(p2) - base) / eps
    return g

def _initial_guess(scale):
    """Generate an initial guess with random perturbations scaled by `scale`."""
    rng = np.random.default_rng()
    # Start with a reasonable layout: hexagonal packing of inner hexagons
    # For N=11, a 3-4-3 or similar arrangement
    # Let's use a compact arrangement
    positions = []
    # Central hexagon
    positions.append((0.0, 0.0))
    # Ring of 6 around it (distance 2*sqrt(3) for non-overlap in hex packing... actually for hexagons side 1,
    # center-to-center distance for touching is 2*sqrt(3)/2 * 2 = 2*sqrt(3)? 
    # Actually for two regular hexagons of side 1 to be tangent, center distance = 2 * apothem = 2 * sqrt(3)/2 * 2? 
    # Apothem of unit hex = sqrt(3)/2. Two hexagons touching side-to-side: center dist = 2*apothem = sqrt(3).
    # But they can also touch vertex-to-vertex. Minimum center distance to avoid overlap = 2 (vertex to vertex) or sqrt(3) (side to side).
    # For safety, use distance > 2.
    for k in range(6):
        a = np.radians(60 * k)
        positions.append((2.2 * np.cos(a), 2.2 * np.sin(a)))
    # 4 more in second ring
    for k in range(4):
        a = np.radians(60 * k + 30)
        positions.append((3.8 * np.cos(a), 3.8 * np.sin(a)))
    
    params = np.zeros(4 + 3 * N)
    # Estimate outer side: need to fit these ~4 unit radius points, so S ~ 4-5
    params[0] = 6.0  # S
    params[1] = 0.0  # cx
    params[2] = 0.0  # cy
    params[3] = 0.0  # rot_deg
    
    for i in range(N):
        ox = 4 + 3 * i
        params[ox] = positions[i][0] + rng.normal(0, scale)
        params[ox+1] = positions[i][1] + rng.normal(0, scale)
        params[ox+2] = rng.uniform(0, 60)
    return params

def construct():
    best_S = np.inf
    best_params = None
    
    # Multi-start optimization
    for trial in range(8):
        scale = 0.5 + trial * 0.3
        x0 = _initial_guess(scale)
        x0[0] = max(3.0, x0[0])  # Ensure S starts reasonably
        
        res = minimize(
            _objective, x0, method='L-BFGS-B',
            jac=_grad_objective,
            bounds=[(3.0, 20.0), (-10, 10), (-10, 10), (-180, 180)] + 
                   [(-10, 10), (-10, 10), (-180, 180)] * N,
            options={'maxiter': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.success or res.fun < best_S:
            S_val = res.x[0]
            clear = -res.fun
            if S_val < best_S and clear > -1.0:  # Some margin
                best_S = S_val
                best_params = res.x
    
    if best_params is None:
        # Fallback to a safe solution
        inner = [[2.5 * i, 0.0, 0.0] for i in range(N)]
        return inner, (0.0, 0.0), 15.0, 0.0
    
    S, cx, cy, rot_deg = best_params[0], best_params[1], best_params[2], best_params[3]
    inner = []
    for i in range(N):
        ox = 4 + 3 * i
        ix, iy, ir = best_params[ox], best_params[ox+1], best_params[ox+2]
        inner.append([float(ix), float(iy), float(ir)])
    
    return inner, (float(cx), float(cy)), float(S), float(rot_deg)
# EVOLVE-BLOCK-END
