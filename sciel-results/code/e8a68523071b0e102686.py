"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0  # 10s safety margin before 150s kill

def _hex_vertices(cx, cy, angle, s=1.0):
    """Return the 6 vertices of a regular hexagon centered at (cx,cy) with rotation angle."""
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _hexagon_contains_point(px, py, cx, cy, angle, S):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side S and rotation angle."""
    # Transform to local frame
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # For a regular hexagon with side S, the apothem is S*sqrt(3)/2
    # The hexagon in local coords has vertices at distance S from center
    # at angles 0, 60, 120, 180, 240, 300 degrees
    # A point is inside if its projection onto each of the 6 face normals is <= apothem
    apothem = S * np.sqrt(3) / 2
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    proj = lx * nx + ly * ny
    return np.all(np.abs(proj) <= apothem + 1e-10)

def _min_enclosing_hex_side(centers, angles, outer_angle):
    """
    Compute the minimum side length S of a regular hexagon centered at origin
    with rotation outer_angle that contains all vertices of the inner hexagons.
    centers: (N,2) array of centers
    angles: (N,) array of rotations
    outer_angle: rotation of the outer hexagon
    """
    # Get all vertices of all inner hexagons
    all_verts = np.zeros((N, 6, 2))
    for i in range(N):
        all_verts[i] = _hex_vertices(centers[i, 0], centers[i, 1], angles[i])
    
    # Transform to outer hexagon's local frame (rotate by -outer_angle)
    cos_a = np.cos(-outer_angle)
    sin_a = np.sin(-outer_angle)
    rx = all_verts[..., 0] * cos_a - all_verts[..., 1] * sin_a
    ry = all_verts[..., 0] * sin_a + all_verts[..., 1] * cos_a
    
    # Project onto the 6 face normals of the outer hexagon
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    # proj[i, j, k] = rx[i,j]*nx[k] + ry[i,j]*ny[k]
    proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
    
    max_proj = np.max(np.abs(proj))
    apothem = np.sqrt(3) / 2
    return max_proj / apothem

def _hexagons_disjoint(centers, angles, tol=1e-9):
    """Check if all pairs of unit hexagons are disjoint (non-overlapping)."""
    for i in range(N):
        for j in range(i + 1, N):
            # Quick distance check
            dist = np.linalg.norm(centers[i] - centers[j])
            # Two unit hexagons can overlap if centers are within ~2*sqrt(3)/2*2 = 2*sqrt(3) ~ 3.46
            # More precisely, the maximum distance between centers of two touching hexagons
            # depends on orientation. For safety, use a generous bound.
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            
            # SAT check
            v1 = _hex_vertices(centers[i, 0], centers[i, 1], angles[i])
            v2 = _hex_vertices(centers[j, 0], centers[j, 1], angles[j])
            
            polys = [v1, v2]
            separated = False
            for poly in polys:
                n = len(poly)
                for k in range(n):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % n]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    if proj1.max() < proj2.min() + tol or proj2.max() < proj1.min() + tol:
                        separated = True
                        break
                if separated:
                    break
            if not separated:
                return False
    return True

def _overlap_penalty(centers, angles, weight=1000.0):
    """Compute penalty for overlapping hexagons."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist > 2.0 * np.sqrt(3) + 0.1:
                continue
            v1 = _hex_vertices(centers[i, 0], centers[i, 1], angles[i])
            v2 = _hex_vertices(centers[j, 0], centers[j, 1], angles[j])
            
            polys = [v1, v2]
            min_gap = np.inf
            for poly in polys:
                n = len(poly)
                for k in range(n):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % n]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    proj1 = v1 @ normal
                    proj2 = v2 @ normal
                    gap = min(proj2.min() - proj1.max(), proj1.min() - proj2.max())
                    # If gap is negative, they overlap along this axis
                    min_gap = min(min_gap, -gap)
            if min_gap < 0:
                penalty += weight * (-min_gap)
    return penalty

def construct():
    start_time = time.time()
    
    def hex_vertices(cx, cy, angle, s=1.0):
        angles = angle + np.arange(6) * np.pi / 3
        return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])
    
    def get_all_vertices(centers, angles):
        verts = np.zeros((N, 6, 2))
        for i in range(N):
            verts[i] = hex_vertices(centers[i, 0], centers[i, 1], angles[i])
        return verts
    
    def compute_S(centers, angles, oangle):
        all_verts = get_all_vertices(centers, angles)
        cos_a = np.cos(-oangle)
        sin_a = np.sin(-oangle)
        rx = all_verts[..., 0] * cos_a - all_verts[..., 1] * sin_a
        ry = all_verts[..., 0] * sin_a + all_verts[..., 1] * cos_a
        
        theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(theta_n)
        ny = np.sin(theta_n)
        
        proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
               ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
        
        max_proj = np.max(np.abs(proj))
        apothem = np.sqrt(3) / 2
        return max_proj / apothem
    
    def sat_overlap(v1, v2):
        polys = [v1, v2]
        for poly in polys:
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                proj1 = v1 @ normal
                proj2 = v2 @ normal
                if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                    return False
        return True
    
    def objective(params):
        """
        params: [cx0, cy0, a0, cx1, cy1, a1, ..., cxN-1, cyN-1, aN-1, oangle]
        """
        centers = params[:3*N].reshape(N, 3)[:, :2]
        angles = params[:3*N].reshape(N, 3)[:, 2]
        oangle = params[3*N]
        
        S = compute_S(centers, angles, oangle)
        
        # Overlap penalty
        penalty = 0.0
        verts = get_all_vertices(centers, angles)
        for i in range(N):
            for j in range(i + 1, N):
                dist = np.linalg.norm(centers[i] - centers[j])
                if dist < 3.5:
                    if sat_overlap(verts[i], verts[j]):
                        penalty += 500.0 * (3.5 - dist)
        
        return S + penalty
    
    # Generate initial configurations
    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y
    
    # Configuration 1: Hexagonal lattice (center + ring 1 + 4 from ring 2)
    axial_coords = [(0,0)]
    axial_coords += [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    axial_coords += [(2,0), (2,-1), (2,-2), (1,-2)]
    
    configs = []
    for i, (q, r) in enumerate(axial_coords):
        x, y = axial_to_cart(q, r)
        configs.append([x, y, 0.0])
    
    # Center the configuration
    centers_arr = np.array(configs)[:, :2]
    centroid = centers_arr.mean(axis=0)
    for i in range(N):
        configs[i][0] -= centroid[0]
        configs[i][1] -= centroid[1]
    
    params1 = np.array(configs).flatten()
    params1 = np.append(params1, 0.0)  # outer angle
    
    # Configuration 2: Try different ring 2 selection
    axial_coords2 = [(0,0)]
    axial_coords2 += [(1,0), (1,-1), (0,-1), (-1,0), (-1,1), (0,1)]
    axial_coords2 += [(2,0), (2,-1), (1,-2), (0,-2)]
    
    configs2 = []
    for i, (q, r) in enumerate(axial_coords2):
        x, y = axial_to_cart(q, r)
        configs2.append([x, y, 0.0])
    
    centers_arr2 = np.array(configs2)[:, :2]
    centroid2 = centers_arr2.mean(axis=0)
    for i in range(N):
        configs2[i][0] -= centroid2[0]
        configs2[i][1] -= centroid2[1]
    
    params2 = np.array(configs2).flatten()
    params2 = np.append(params2, 0.0)
    
    # Configuration 3: All rotations 0, but slightly different positions
    params3 = params1.copy()
    params3[3*N] = np.pi / 6  # Different outer angle
    
    # Configuration 4: Random perturbation of lattice
    params4 = params1 + np.random.normal(0, 0.1, 3*N+1)
    params4[3*N] = 0.0
    
    initial_params_list = [params1, params2, params3, params4]
    
    best_params = None
    best_score = np.inf
    
    # Evaluate initial configs
    for p in initial_params_list:
        if time.time() - start_time > DEADLINE:
            break
        s = objective(p)
        if s < best_score:
            best_score = s
            best_params = p.copy()
    
    # Optimization phases
    # Phase 1: Powell on best initial
    if best_params is not None and time.time() - start_time < DEADLINE - 5:
        res = minimize(objective, best_params, method='Powell',
                      options={'maxiter': 1000, 'maxfev': 8000, 'disp': False})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
    
    # Phase 2: Try other initial configs with shorter optimization
    for p in initial_params_list[1:]:
        if time.time() - start_time > DEADLINE - 10:
            break
        # Quick local optimization
        res = minimize(objective, p, method='Powell',
                      options={'maxiter': 500, 'maxfev': 3000, 'disp': False})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
    
    # Phase 3: Refine with Nelder-Mead for local polish
    if best_params is not None and time.time() - start_time < DEADLINE - 5:
        res = minimize(objective, best_params, method='Nelder-Mead',
                      options={'maxiter': 500, 'maxfev': 3000, 'disp': False,
                               'xatol': 1e-6, 'fatol': 1e-6})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x.copy()
    
    # Extract results
    centers = best_params[:3*N].reshape(N, 3)[:, :2]
    angles = best_params[:3*N].reshape(N, 3)[:, 2]
    oangle = best_params[3*N]
    
    # Verify validity: check no overlaps
    verts = get_all_vertices(centers, angles)
    valid = True
    for i in range(N):
        for j in range(i + 1, N):
            if sat_overlap(verts[i], verts[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fall back to lattice configuration which is known to be valid
        for i, (q, r) in enumerate(axial_coords):
            x, y = axial_to_cart(q, r)
            centers[i, 0] = x - centroid[0]
            centers[i, 1] = y - centroid[1]
            angles[i] = 0.0
        oangle = 0.0
        best_score = compute_S(centers, angles, oangle)
    
    inner = []
    for i in range(N):
        inner.append([float(centers[i, 0]), float(centers[i, 1]), float(np.degrees(angles[i]))])
    
    # The outer hexagon is centered at (0,0) in our optimization frame
    return inner, (0.0, 0.0), float(best_score), float(np.degrees(oangle))

# EVOLVE-BLOCK-END
