"""Pack 11 unit hexagons inside a minimal regular hexagon."""
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar, differential_evolution

N = 11
DEADLINE = 145.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, angle, s=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with rotation angle (radians)."""
    angles = angle + np.arange(6) * np.pi / 3
    return np.column_stack([cx + s * np.cos(angles), cy + s * np.sin(angles)])

def _compute_S_for_rotation_fast(verts, oangle):
    """
    Compute minimal side length S of outer hexagon at origin with rotation oangle
    containing all vertices in 'verts' (N x 6 x 2).
    Vectorized projection.
    """
    cos_a = np.cos(-oangle)
    sin_a = np.sin(-oangle)
    
    rx = verts[..., 0] * cos_a - verts[..., 1] * sin_a
    ry = verts[..., 0] * sin_a + verts[..., 1] * cos_a
    
    theta_n = np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(theta_n)
    ny = np.sin(theta_n)
    
    proj = rx[..., np.newaxis] * nx[np.newaxis, np.newaxis, :] + \
           ry[..., np.newaxis] * ny[np.newaxis, np.newaxis, :]
    
    max_proj = np.max(np.abs(proj))
    apothem = np.sqrt(3) / 2
    return max_proj / apothem

def _seg_seg_dist(p1, p2, p3, p4):
    """Minimum distance between two line segments p1-p2 and p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3
    a = np.dot(d1, d1)
    e = np.dot(d2, d2)
    f = np.dot(d2, r)
    
    if a < 1e-12 and e < 1e-12:
        return np.linalg.norm(p1 - p3)
    
    if a < 1e-12:
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        s = c / a
        s = max(0, min(1, s))
        d = (p1 + s * d1) - p3
        return np.linalg.norm(d)
    
    b = np.dot(d1, d2)
    denom = a * e - b * b
    s = 0
    t = 0
    
    if denom > 1e-12:
        s = (b * f - c * e) / denom
        s = max(0, min(1, s))
        t = (b * s + f) / e
        if t < 0:
            t = 0
            s = max(0, min(1, c / a))
        elif t > 1:
            t = 1
            s = max(0, min(1, (b + f) / a))
        d = (p1 + s * d1) - (p3 + t * d2)
        return np.linalg.norm(d)
    else:
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap.
    """
    # Check if any vertex of p1 is inside p2
    # Using winding number or cross product check for convex polygon
    def point_in_poly(pt, poly):
        n = len(poly)
        sign = 0
        for i in range(n):
            a = poly[i]
            b = poly[(i+1) % n]
            cross = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
            if cross > 1e-12:
                if sign < 0:
                    return False
                sign = 1
            elif cross < -1e-12:
                if sign > 0:
                    return False
                sign = -1
        return True

    for v in p1:
        if point_in_poly(v, p2):
            return 0.0
    for v in p2:
        if point_in_poly(v, p1):
            return 0.0
    
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _objective_fast(params, time_start):
    """
    Fast objective for DE: Fixed outer angles (0 and 30 deg).
    params: shape (33,)
    """
    if time.time() - time_start > DEADLINE:
        return float('inf')
        
    n = N
    # Reshape to (N, 3)
    p = params.reshape(n, 3)
    # Extract vertices
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(p[i, 0], p[i, 1], p[i, 2])
    
    # Compute S for two candidate outer angles
    s1 = _compute_S_for_rotation_fast(verts, 0.0)
    s2 = _compute_S_for_rotation_fast(verts, np.pi / 6) # 30 degrees
    S = min(s1, s2)
    
    # Overlap Penalty
    penalty = 0.0
    centers = p[:, :2]
    # Precompute distances to skip far pairs
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < 2.5: # Threshold
                sep = _poly_separation(verts[i], verts[j])
                if sep < 1e-4:
                    penalty += 1000.0 * (1e-4 - sep) ** 2
                elif sep < 0.01:
                    penalty += 10.0 * (0.01 - sep) ** 2

    return S + penalty

def _objective_precise(params, time_start):
    """
    Precise objective for local polish: Optimizes outer angle via minimize_scalar.
    """
    if time.time() - time_start > DEADLINE:
        return float('inf')
        
    n = N
    p = params.reshape(n, 3)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        verts[i] = _hex_vertices(p[i, 0], p[i, 1], p[i, 2])
    
    # Find best outer angle
    def S_of_angle(oangle):
        return _compute_S_for_rotation_fast(verts, oangle)
    
    # Coarse search
    best_S = float('inf')
    best_angle = 0.0
    n_coarse = 12
    for i in range(n_coarse):
        ang = i * (np.pi / 3) / n_coarse
        s_val = S_of_angle(ang)
        if s_val < best_S:
            best_S = s_val
            best_angle = ang
            
    # Fine search
    try:
        result = minimize_scalar(S_of_angle, bounds=(best_angle - 0.1, best_angle + 0.1), 
                                 method='bounded', options={'xatol': 1e-8})
        if result.success:
            best_S = result.fun
            best_angle = result.x
    except:
        pass
        
    # Overlap Penalty (stricter)
    penalty = 0.0
    centers = p[:, :2]
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(centers[i] - centers[j])
            if dist < 2.5:
                sep = _poly_separation(verts[i], verts[j])
                if sep < 1e-6:
                    penalty += 10000.0 * (1e-6 - sep) ** 2
                elif sep < 0.01:
                    penalty += 100.0 * (0.01 - sep) ** 2

    return best_S + penalty

def construct():
    start_time = time.time()
    
    # Initial guess: Hexagonal Lattice centered at 0
    # Axial coordinates for 11 hexes: center + ring 1 (6) + part of ring 2 (4)
    # Or simply a dense cluster.
    # Let's use the seed from inspiration: concentric rings
    # Center: (0,0)
    # Ring 1: 6 hexes at distance sqrt(3) (touching)
    # Ring 2: 4 hexes at distance 3?
    
    # Let's generate a reasonable initial guess
    init_params = np.zeros(3 * N)
    
    # Center
    init_params[0:2] = [0.0, 0.0]
    init_params[2] = 0.0
    
    # Ring 1: 6 hexes
    for k in range(6):
        angle = k * np.pi / 3
        dist = np.sqrt(3) + 0.01
        idx = 1 + k
        init_params[idx*3] = dist * np.cos(angle)
        init_params[idx*3+1] = dist * np.sin(angle)
        init_params[idx*3+2] = 0.0
        
    # Ring 2: 4 hexes (filling gaps)
    # Positions roughly at distance 3, angles 30, 90, 150, 210?
    # Actually, for a hex lattice, the next ring has 12 hexes. We only need 4.
    # Let's place them at the "corners" of the ring 1 cluster to minimize bounding box.
    # Or just randomize angles for DE to explore.
    
    # Better initial guess: Randomized hex lattice
    np.random.seed(42)
    init_params = np.zeros(3 * N)
    # Center
    init_params[0:2] = [0.0, 0.0]
    init_params[2] = 0.0
    
    # Ring 1
    for k in range(6):
        angle = k * np.pi / 3
        dist = np.sqrt(3) + 0.05
        idx = 1 + k
        init_params[idx*3] = dist * np.cos(angle)
        init_params[idx*3+1] = dist * np.sin(angle)
        init_params[idx*3+2] = 0.0
        
    # Ring 2 (4 hexes)
    # Place them in the "valleys" between ring 1 hexes?
    # Valley angle: 30, 90, 150, 210, 270, 330.
    # Valley dist: 3.0
    valley_angles = [np.pi/6, np.pi/2, 3*np.pi/2, 5*np.pi/6] # 30, 90, 270, 150
    for k in range(4):
        idx = 7 + k
        init_params[idx*3] = 3.0 * np.cos(valley_angles[k])
        init_params[idx*3+1] = 3.0 * np.sin(valley_angles[k])
        init_params[idx*3+2] = 0.0

    # Bounds for DE
    # x, y in [-5, 5], angle in [0, pi/3]
    bounds = [(-5, 5), (-5, 5), (0, np.pi/3)] * N
    
    best_params = init_params.copy()
    best_score = _objective_fast(best_params, start_time)
    
    # Phase 1: Differential Evolution
    if time.time() - start_time < DEADLINE - 60:
        try:
            res_de = differential_evolution(
                _objective_fast, 
                bounds, 
                args=(start_time,),
                maxiter=50, 
                popsize=10, 
                tol=1e-4, 
                seed=42,
                polish=False, # We will polish later
                updating='immediate'
            )
            if res_de.fun < best_score:
                best_score = res_de.fun
                best_params = res_de.x.copy()
        except Exception as e:
            pass
            
    # Phase 2: Local Polish with Precise Objective
    # Use Powell or Nelder-Mead on the best result from DE
    if time.time() - start_time < DEADLINE - 20:
        try:
            res_polish = minimize(
                _objective_precise, 
                best_params, 
                args=(start_time,),
                method='Powell',
                options={'maxiter': 200, 'maxfev': 5000, 'disp': False}
            )
            if res_polish.fun < best_score:
                best_score = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception as e:
            pass

    # Phase 3: Final Polish with Nelder-Mead if time remains
    if time.time() - start_time < DEADLINE - 5:
        try:
            res_nm = minimize(
                _objective_precise, 
                best_params, 
                args=(start_time,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res_nm.fun < best_score:
                best_score = res_nm.fun
                best_params = res_nm.x.copy()
        except Exception as e:
            pass

    # Extract results
    inner = []
    for i in range(N):
        cx = best_params[3 * i]
        cy = best_params[3 * i + 1]
        angle = best_params[3 * i + 2]
        inner.append([cx, cy, np.degrees(angle)])
    
    # Find final best outer hexagon
    verts = np.zeros((N, 6, 2))
    for i in range(N):
        verts[i] = _hex_vertices(inner[i][0], inner[i][1], np.radians(inner[i][2]))
        
    def S_of_angle(oangle):
        return _compute_S_for_rotation_fast(verts, oangle)
        
    # Coarse search
    best_S = float('inf')
    best_angle = 0.0
    n_coarse = 36
    for i in range(n_coarse):
        ang = i * (np.pi / 3) / n_coarse
        s_val = S_of_angle(ang)
        if s_val < best_S:
            best_S = s_val
            best_angle = ang
            
    # Fine search
    result = minimize_scalar(S_of_angle, bounds=(best_angle - 0.05, best_angle + 0.05), 
                             method='bounded', options={'xatol': 1e-10})
    if result.success:
        best_S = result.fun
        best_angle = result.x
        
    best_angle = best_angle % (np.pi / 3)
    
    return inner, (0.0, 0.0), best_S, np.degrees(best_angle)
