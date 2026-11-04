"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle, s=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with rotation angle (radians)."""
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _compute_S_for_rotation(verts, oangle):
    """
    Compute minimal side length S of outer hexagon at origin with rotation oangle
    containing all vertices in 'verts' (N x 6 x 2).
    Uses vectorized projection onto 6 facet normals.
    """
    # Transform to container local frame (rotate by -oangle)
    cos_a = np.cos(-oangle)
    sin_a = np.sin(-oangle)
    
    rx = verts[..., 0] * cos_a - verts[..., 1] * sin_a
    ry = verts[..., 0] * sin_a + verts[..., 1] * cos_a
    
    # Normals for unit hexagon facets (apothem directions)
    # For hexagon with vertices at 0, 60, 120, ... degrees, facets are at 30, 90, 150, ...
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    # Projections: (N, 6, 6)
    proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
    
    max_proj = np.max(np.abs(proj))
    apothem = np.sqrt(3) / 2
    return max_proj / apothem

def _find_best_outer(verts):
    """
    Given all inner hexagon vertices (N x 6 x 2), find the minimal enclosing regular hexagon.
    Returns (best_S, best_oangle). The outer hexagon is centered at origin.
    We search over rotation in [0, pi/3) due to 6-fold symmetry.
    """
    def S_of_angle(oangle):
        return _compute_S_for_rotation(verts, oangle)
    
    # Coarse search
    best_S = float('inf')
    best_angle = 0.0
    n_coarse = 30
    for i in range(n_coarse):
        ang = i * (np.pi / 3) / n_coarse
        s_val = S_of_angle(ang)
        if s_val < best_S:
            best_S = s_val
            best_angle = ang
    
    # Fine search around best
    result = minimize_scalar(S_of_angle, bounds=(best_angle - 0.1, best_angle + 0.1), 
                             method='bounded', options={'xatol': 1e-8})
    if result.success:
        best_S = result.fun
        best_angle = result.x
    
    # Ensure angle is in [0, pi/3)
    best_angle = best_angle % (np.pi / 3)
    
    return best_S, best_angle

def _sat_overlap(v1, v2, tol=1e-9):
    """Check if two convex hexagons overlap using Separating Axis Theorem."""
    polys = [v1, v2]
    for poly in polys:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-12:
                continue
            normal = normal / norm
            proj1 = v1 @ normal
            proj2 = v2 @ normal
            if proj1.max() + tol < proj2.min() or proj2.max() + tol < proj1.min():
                return False
    return True

def construct():
    start_time = time.time()
    
    def get_all_vertices(params):
        """Get vertices for all hexagons given parameters [cx0, cy0, a0, cx1, ...]"""
        n = len(params) // 3
        verts = np.zeros((n, 6, 2))
        for i in range(n):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            angle = params[3 * i + 2]
            verts[i] = _hex_vertices(cx, cy, angle)
        return verts

    def objective(params):
        if time.time() - start_time > DEADLINE:
            return float('inf')
        
        n = N
        verts = get_all_vertices(params)
        
        # Find best outer hexagon analytically
        S, _ = _find_best_outer(verts)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                c1 = verts[i].mean(axis=0)
                c2 = verts[j].mean(axis=0)
                dist = np.linalg.norm(c1 - c2)
                if dist < 2.0:  # Only check SAT if potentially overlapping
                    if _sat_overlap(verts[i], verts[j]):
                        penalty += 1000.0 * (2.0 - dist)
        
        return S + penalty

    # Initial guess: Hexagonal Lattice
    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    axial_coords = [(0,0)]
    # Ring 1: 6 hexes
    axial_coords += [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    # Ring 2: 4 more
    axial_coords += [(2,0), (2,-1), (2,-2), (1,-2)]
    
    params = np.zeros(3 * N)
    for i, (q, r) in enumerate(axial_coords):
        x, y = axial_to_cart(q, r)
        params[3*i] = x
        params[3*i+1] = y
        params[3*i+2] = 0.0
    
    # Center the configuration
    centers = params[:3*N].reshape(N, 3)[:, :2]
    centroid = centers.mean(axis=0)
    params[:3*N:3] -= centroid[0]
    params[1:3*N:3] -= centroid[1]
    
    best_params = params.copy()
    best_score = objective(best_params)
    
    # Try multiple optimization strategies
    # Strategy 1: Nelder-Mead
    if time.time() - start_time < DEADLINE - 30:
        res = minimize(objective, best_params, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
    
    # Strategy 2: Powell from current best
    if time.time() - start_time < DEADLINE - 20:
        res2 = minimize(objective, best_params, method='Powell',
                       options={'maxiter': 2000, 'maxfev': 15000, 'disp': False})
        if res2.fun < best_score:
            best_score = res2.fun
            best_params = res2.x.copy()
    
    # Strategy 3: Try a different initial configuration (rotated lattice)
    if time.time() - start_time < DEADLINE - 15:
        params_alt = params.copy()
        # Rotate all angles by 30 degrees
        params_alt[2::3] = np.pi / 6
        res3 = minimize(objective, params_alt, method='Nelder-Mead',
                       options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True})
        if res3.fun < best_score:
            best_score = res3.fun
            best_params = res3.x.copy()
    
    # Strategy 4: Final polish with Powell
    if time.time() - start_time < DEADLINE - 5:
        res4 = minimize(objective, best_params, method='Powell',
                       options={'maxiter': 1000, 'maxfev': 8000, 'disp': False})
        if res4.fun < best_score:
            best_score = res4.fun
            best_params = res4.x.copy()
    
    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    # Find final best outer hexagon
    verts = get_all_vertices(best_params)
    S, oangle = _find_best_outer(verts)
    
    # The outer hexagon is centered at origin in our frame
    # Return inner positions, outer center (0,0), side length, and rotation angle in degrees
    return inner, (0.0, 0.0), S, np.degrees(oangle)
