import time
import math
import numpy as np
from scipy.optimize import minimize, linprog

N = 11
START = time.time()
DEADLINE = 150.0 - 5.0  # safety margin before 150 s kill


def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx, cy) with given rotation (degrees)."""
    a = math.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts[i] = [cx + side * math.cos(theta), cy + side * math.sin(theta)]
    return verts


def _all_inner_vertices(params):
    """Given flat params (3*N: x, y, angle_deg per hex), return (6N, 2) vertex array."""
    n = len(params) // 3
    allv = np.empty((6 * n, 2))
    for i in range(n):
        allv[6*i:6*i+6] = _hex_vertices(params[3*i], params[3*i+1], params[3*i+2])
    return allv


def _min_enclosing_hex_side(vertices, outer_angle_deg, outer_center=None):
    """
    Find the minimum side length of a regular hexagon (with given rotation) that contains all
    given vertices. If outer_center is None, also optimize the center.
    
    Returns (side, center_x, center_y).
    """
    verts = np.asarray(vertices)
    # A regular hexagon with rotation angle 'a' has 6 edge normals at angles a + i*pi/3,
    # each edge is at distance s*sqrt(3)/2 from center along the inward normal.
    # Containment: for each vertex p, for each edge k: dot(p - c, n_k) <= s * sqrt(3)/2
    # where n_k are the outward normals (unit vectors).
    
    a = math.radians(outer_angle_deg)
    normals = np.empty((6, 2))
    for k in range(6):
        theta = a + k * math.pi / 3.0
        normals[k] = [math.cos(theta), math.sin(theta)]
    
    # For a fixed center c, the required side is:
    # s >= max over all vertices p and all edges k of dot(p - c, n_k) / (sqrt(3)/2)
    # but only where dot(p-c, n_k) > 0; otherwise no constraint.
    # Actually, containment requires dot(p - c, n_k) <= s*sqrt(3)/2 for ALL k.
    # So s >= max_{p,k} dot(p - c, n_k) / (sqrt(3)/2), but we take max over positive values only? 
    # No: if dot(p-c, n_k) is negative, that constraint is automatically satisfied for any s>=0.
    # So s = max(0, max_{p,k} dot(p-c, n_k)) / (sqrt(3)/2)
    
    # But wait, we need ALL 6 half-planes. The hexagon is the intersection of 6 half-planes.
    # For a point to be inside: dot(p - c, n_k) <= s*sqrt(3)/2 for all k = 0..5.
    # So the minimum s for fixed c is: max_k max_p dot(p-c, n_k) / (sqrt(3)/2)
    # But we must ensure this max is positive for each k, otherwise that edge doesn't bind.
    # Actually s must satisfy ALL 6 constraints simultaneously:
    # s >= dot(p-c, n_k)/(sqrt(3)/2) for every p and every k.
    # So s_min(c) = max_{p, k} [dot(p-c, n_k) / (sqrt(3)/2)]
    # But if for some k, all dot(p-c, n_k) < 0, then that constraint gives s >= negative, which is fine (s>=0).
    # So s_min(c) = max(0, max_{p,k} dot(p-c, n_k)) / (sqrt(3)/2)
    
    apothem_factor = math.sqrt(3.0) / 2.0
    
    if outer_center is not None:
        c = np.array(outer_center, dtype=float)
        # Compute max dot product
        # For each normal n_k, compute max_p dot(p - c, n_k) = max_p (dot(p, n_k) - dot(c, n_k))
        # = (max_p dot(p, n_k)) - dot(c, n_k)
        max_dots = []
        for k in range(6):
            dots = verts @ normals[k]  # dot(p, n_k) for all p
            max_dot = dots.max() - c @ normals[k]
            max_dots.append(max_dot)
        s = max(0.0, max(max_dots)) / apothem_factor
        return s, c[0], c[1]
    
    # Optimize center c to minimize s_min(c)
    # s_min(c) = max(0, max_k [M_k - c·n_k]) / apothem_factor
    # where M_k = max_p dot(p, n_k)
    # This is a convex optimization problem (max of affine functions).
    # We can solve it via LP: minimize t subject to M_k - c·n_k <= t*apothem_factor for all k, and t >= 0.
    # Variables: c_x, c_y, t. 3 variables, 6+1=7 constraints.
    
    M = np.empty(6)
    for k in range(6):
        dots = verts @ normals[k]
        M[k] = dots.max()
    
    # Minimize t
    # s.t. -c_x * n_kx - c_y * n_ky + apothem_factor * t >= M_k  =>  -n_kx*c_x - n_ky*c_y + apothem_factor*t - M_k >= 0
    # In standard form for linprog (minimize c^T x s.t. A_ub x <= b_ub):
    # -n_kx*c_x - n_ky*c_y + apothem_factor*t <= -M_k  (NO, this is wrong direction)
    # We need: M_k - c·n_k <= apothem_factor * t
    # => -n_kx*c_x - n_ky*c_y + apothem_factor*t >= M_k
    # => n_kx*c_x + n_ky*c_y - apothem_factor*t <= -M_k
    
    # Also t >= 0 => -t <= 0
    
    c_obj = np.array([0.0, 0.0, 1.0])  # minimize t
    A_ub = np.empty((7, 3))
    b_ub = np.empty(7)
    for k in range(6):
        A_ub[k] = [normals[k][0], normals[k][1], -apothem_factor]
        b_ub[k] = -M[k]
    A_ub[6] = [0.0, 0.0, -1.0]
    b_ub[6] = 0.0  # t >= 0
    
    res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, method='highs')
    if res.success:
        cx, cy, t = res.x
        return t, cx, cy
    else:
        # Fallback: use centroid
        cx, cy = verts[:, 0].mean(), verts[:, 1].mean()
        return _min_enclosing_hex_side(verts, outer_angle_deg, [cx, cy])


def _overlap_penalty(params, weight=50.0):
    """Penalty for overlapping inner hexagons. Uses minimum center-distance approximation."""
    n = len(params) // 3
    penalty = 0.0
    # For two unit regular hexagons, the minimum center distance for non-overlap depends on relative rotation.
    # Conservative lower bound: sqrt(3) (edge-sharing, same orientation).
    # With arbitrary rotation, the true minimum can be smaller, but sqrt(3) is a safe lower bound
    # for the same-orientation case. For safety, use a slightly smaller value.
    min_dist = math.sqrt(3.0) * 0.99  # small tolerance
    for i in range(n):
        xi, yi = params[3*i], params[3*i+1]
        for j in range(i + 1, n):
            xj, yj = params[3*j], params[3*j+1]
            d = math.sqrt((xi - xj)**2 + (yi - yj)**2)
            if d < min_dist:
                penalty += (min_dist - d) ** 2
    return weight * penalty


def _objective(params, outer_angle_deg):
    """Outer side length + overlap penalty."""
    verts = _all_inner_vertices(params)
    s, _, _ = _min_enclosing_hex_side(verts, outer_angle_deg)
    pen = _overlap_penalty(params)
    return s + pen


def _build_positions():
    """Build initial positions: central hex + 6 around + 4 from ring 2."""
    # For flat-top unit hexagons, center-to-center distance for edge-sharing is sqrt(3).
    # Use axial coordinates. Flat-top: x = 1.5 * s * q, y = sqrt(3) * s * (r + q/2), with s=1.
    # But center-to-center for adjacent is 2 * apothem = sqrt(3).
    # Let's just use the direct formula.
    spacing = math.sqrt(3.0)
    
    positions = []
    # Center
    positions.append([0.0, 0.0, 0.0])
    # Ring 1: 6 hexagons at 60-degree intervals, distance = sqrt(3)
    for i in range(6):
        theta = i * math.pi / 3.0
        positions.append([spacing * math.cos(theta), spacing * math.sin(theta), 0.0])
    # Ring 2: 4 hexagons. Use axial coords (2,0), (1,1), (0,2), (-1,1)
    # For flat-top: x = 1.5 * q * side, y = sqrt(3) * (r + q/2) * side, side=1
    # But this gives center-to-center of 1 for (1,0), which is wrong.
    # The correct formula: for unit hexagons (side=1), the distance between adjacent centers is sqrt(3).
    # In axial coords, (1,0) is adjacent to (0,0). So the pixel distance should be sqrt(3).
    # Standard formula: x = sqrt(3) * (q + r/2), y = 3/2 * r  for pointy-top with size=1 (circumradius=1)
    # But for flat-top: x = 3/2 * q, y = sqrt(3) * (r + q/2)
    # With this, distance from (0,0) to (1,0) is 3/2, not sqrt(3).
    # I think the issue is that "size" in hex grid formulas is the circumradius, not the side length.
    # For a regular hexagon with side length s, circumradius = s, inradius = s*sqrt(3)/2.
    # Adjacent center distance = 2 * inradius = s*sqrt(3).
    # So with s=1, adjacent distance = sqrt(3).
    # In the formula x = sqrt(3)*(q+r/2), y = 3*r/2, the distance from (0,0) to (1,0) is sqrt(3), correct!
    # Let me use pointy-top coordinates.
    
    # Redo with pointy-top: x = sqrt(3) * (q + r/2), y = 3/2 * r
    # Ring 1 axial: (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # Ring 2 axial (select 4): (2,0), (1,1), (0,2), (-1,2) or similar
    
    positions = []
    positions.append([0.0, 0.0, 0.0])
    
    ring1_axial = [(1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    for q, r in ring1_axial:
        x = math.sqrt(3.0) * (q + r / 2.0)
        y = 1.5 * r
        positions.append([x, y, 0.0])
    
    # Ring 2: pick 4 from: (2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)
    # Actually ring 2 axial coords are those with max(|q|,|r|,|q+r|) = 2
    # Let me just pick 4 that are well-spread
    ring2_axial = [(2,0), (1,1), (0,2), (-2,0)]
    for q, r in ring2_axial:
        x = math.sqrt(3.0) * (q + r / 2.0)
        y = 1.5 * r
        positions.append([x, y, 0.0])
    
    return positions


def construct():
    t0 = time.time()
    
    positions = _build_positions()
    x0 = np.array(positions).flatten()
    
    best_side = float('inf')
    best_params = None
    
    outer_angles = [0.0, 30.0, 60.0, 15.0, 45.0]
    
    for outer_rot in outer_angles:
        if time.time() > DEADLINE - 10:
            break
        try:
            res = minimize(_objective, x0, args=(outer_rot,), method='L-BFGS-B',
                          options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8})
            if res.fun < best_side:
                # Recompute exact side
                verts = _all_inner_vertices(res.x)
                s, cx, cy = _min_enclosing_hex_side(verts, outer_rot)
                if s < best_side:
                    best_side = s
                    best_params = res.x.copy()
                    best_center = [cx, cy]
                    best_outer_rot = outer_rot
        except Exception:
            continue
    
    if best_params is None:
        # Fallback
        verts = _all_inner_vertices(x0)
        best_side, cx, cy = _min_enclosing_hex_side(verts, 0.0)
        best_params = x0.copy()
        best_center = [cx, cy]
        best_outer_rot = 0.0
    
    inner_list = []
    for i in range(N):
        inner_list.append([best_params[3*i], best_params[3*i+1], best_params[3*i+2]])
    
    return (inner_list, best_center, best_side, best_outer_rot)


# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
