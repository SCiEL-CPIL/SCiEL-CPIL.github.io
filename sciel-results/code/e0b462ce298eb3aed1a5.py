import numpy as np
import time
from scipy.optimize import minimize, differential_evolution

def _hex_vertices(cx, cy, angle_deg, s=1.0):
    """Vertices of a regular hexagon with side s, centered at (cx,cy), rotated angle_deg."""
    angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]) + angle_deg)
    vs = np.column_stack([s * np.cos(angles), s * np.sin(angles)])
    return vs + np.array([cx, cy])

def _point_in_polygon(pt, poly):
    """Check if point is inside convex polygon using cross products."""
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
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)
    
    c = np.dot(d1, r)
    if e < 1e-12:
        t = 0
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
        # Parallel
        s = 0
        t = f / e
        t = max(0, min(1, t))
        d = p1 - (p3 + t * d2)
        return np.linalg.norm(d)

def _poly_separation(p1, p2):
    """
    Compute the minimum distance between two convex polygons.
    Returns 0 if they overlap (interiors intersect or share boundary),
    positive if separated.
    """
    # Check if any vertex of p1 is inside p2
    for v in p1:
        if _point_in_polygon(v, p2):
            return 0.0
    # Check if any vertex of p2 is inside p1
    for v in p2:
        if _point_in_polygon(v, p1):
            return 0.0
    
    # Check edge-edge distances
    min_dist = float('inf')
    n1 = len(p1)
    n2 = len(p2)
    for i in range(n1):
        a1 = p1[i]
        a2 = p1[(i+1) % n1]
        for j in range(n2):
            b1 = p2[j]
            b2 = p2[(j+1) % n2]
            # Compute min distance between two line segments
            d = _seg_seg_dist(a1, a2, b1, b2)
            if d < min_dist:
                min_dist = d
    return min_dist

def _fast_side_and_penalty(params, n, outer_angle_deg=0.0):
    """
    Fast calculation of required outer side length and overlap penalty.
    params: shape (n, 3) [x, y, angle_deg]
    Returns: (side_length, penalty)
    """
    # Outer hexagon normals (for angle 0, vertices at 0, 60.., normals at 30, 90..)
    # Let's stick to the standard: Vertices at 0, 60, ...
    # Normals are at 30, 90, 150, 210, 270, 330
    normal_angles = np.deg2rad(np.array([30, 90, 150, 210, 270, 330]) + outer_angle_deg)
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
    
    # Compute all vertices of all inner hexagons
    # Vectorized vertex computation
    xs = params[:, 0]
    ys = params[:, 1]
    angles_deg = params[:, 2]
    angles_rad = np.deg2rad(angles_deg)
    
    # Base hexagon vertices relative to center
    base_angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300]))
    base_cos = np.cos(base_angles)
    base_sin = np.sin(base_angles)
    
    # Rotate base vertices
    # cos(a+b) = cos(a)cos(b) - sin(a)sin(b)
    # sin(a+b) = sin(a)cos(b) + cos(a)sin(b)
    cos_a = np.cos(angles_rad)[:, np.newaxis]
    sin_a = np.sin(angles_rad)[:, np.newaxis]
    
    vx = base_cos * cos_a - base_sin * sin_a + xs[:, np.newaxis]
    vy = base_sin * cos_a + base_cos * sin_a + ys[:, np.newaxis]
    
    # vx, vy shape: (n, 6)
    
    # Project onto normals
    # proj[i, k] = vx[i, j] * nx[k] + vy[i, j] * ny[k] for all j, take max over j
    # We need max over vertices j for each normal k
    # proj = vx @ normals.T + vy @ normals.T  -> shape (n, 6)
    proj = vx @ normals.T + vy @ normals.T
    
    # Max projection for each normal across all hexagons
    max_proj = np.max(proj, axis=0)
    
    # Required side length S such that S * sqrt(3)/2 >= max_proj
    # S >= 2 * max_proj / sqrt(3)
    side = 2.0 * np.max(max_proj) / np.sqrt(3.0)
    
    # Overlap Penalty
    penalty = 0.0
    # Only check pairs that are close
    # We can approximate distance by center distance
    centers = params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < 2.0: # Potential overlap
                v1 = np.column_stack([vx[i], vy[i]])
                v2 = np.column_stack([vx[j], vy[j]])
                sep = _poly_separation(v1, v2)
                if sep < 0.01:
                    penalty += 100.0 * (0.01 - sep) ** 2
    
    return side, penalty

def _objective(params, n):
    side, penalty = _fast_side_and_penalty(params, n)
    return side + penalty

def _is_valid(params, side, n, min_sep=1e-8, outer_angle_deg=0.0):
    """Check if the configuration is valid."""
    # Re-check side length strictly
    side_check, _ = _fast_side_and_penalty(params, n, outer_angle_deg)
    if side_check > side + 1e-9:
        return False
        
    # Check non-overlap strictly
    centers = params[:, :2]
    dists = np.linalg.norm(centers[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    for i in range(n):
        for j in range(i+1, n):
            if dists[i, j] < 2.0:
                v1 = _hex_vertices(params[i, 0], params[i, 1], params[i, 2])
                v2 = _hex_vertices(params[j, 0], params[j, 1], params[j, 2])
                sep = _poly_separation(v1, v2)
                if sep < min_sep:
                    return False
    return True

def _seed_solution():
    """Return the concentric ring seed solution."""
    eps = 1e-9
    centers = []
    centers.append((0.0, 0.0))
    for k in range(6):
        angle = np.deg2rad(k * 60)
        x = (np.sqrt(3) + eps) * np.cos(angle)
        y = (np.sqrt(3) + eps) * np.sin(angle)
        centers.append((x, y))
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        x = (3.0 + 2*eps) * np.cos(angle)
        y = (3.0 + 2*eps) * np.sin(angle)
        centers.append((x, y))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    max_x = (2 + eps) * np.sqrt(3)
    max_y = 4 + 2 * eps
    s_x = 2 * max_x / np.sqrt(3)
    s_y = max_y
    side = max(s_x, s_y) + 1e-9
    
    return inner, (0.0, 0.0), side, 30.0

def construct():
    start_time = time.time()
    DEADLINE = 140.0
    
    n = 11
    
    # Seed solution
    seed_inner, seed_center, seed_side, seed_angle = _seed_solution()
    best_inner = seed_inner
    best_side = seed_side
    best_valid = True
    
    # Initial guess
    init_params = np.zeros((n, 3))
    for i, (x, y, a) in enumerate(seed_inner):
        init_params[i] = [x, y, a]
    
    # Try Differential Evolution for global search
    # Bounds: x, y in [-5, 5], angle in [0, 180]
    bounds = [(-5, 5), (-5, 5), (0, 180)] * n
    
    try:
        # DE is slow but robust. We limit maxiter and popsize.
        # Since we have a fast objective, we can try it.
        # However, DE with 33 dimensions might be too slow.
        # Let's try Nelder-Mead with the fast objective first, as it converges faster locally.
        # The seed is good, so NM should refine it.
        
        result = minimize(
            _objective,
            init_params.flatten(),
            args=(n,),
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
        )
        
        elapsed = time.time() - start_time
        if elapsed < DEADLINE - 10:
            opt_params = result.x.reshape(n, 3)
            opt_side, opt_pen = _fast_side_and_penalty(opt_params, n)
            
            if opt_pen < 1e-6: # Ensure no overlap
                if _is_valid(opt_params, opt_side, n):
                    if opt_side < best_side:
                        best_side = opt_side
                        best_inner = [[float(opt_params[i,0]), float(opt_params[i,1]), float(opt_params[i,2])] for i in range(n)]
                        best_valid = True
            
            # Try a few more restarts if time permits
            for restart in range(5):
                elapsed = time.time() - start_time
                if elapsed > DEADLINE - 15:
                    break
                
                # Perturb the initial guess
                rng = np.random.default_rng(restart)
                perturbed = opt_params + rng.normal(0, 0.2, (n, 3))
                # Clamp angles
                perturbed[:, 2] = (perturbed[:, 2] + 180) % 180
                
                result_r = minimize(
                    _objective,
                    perturbed.flatten(),
                    args=(n,),
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                )
                
                opt_params_r = result_r.x.reshape(n, 3)
                opt_side_r, opt_pen_r = _fast_side_and_penalty(opt_params_r, n)
                
                if opt_pen_r < 1e-6:
                    if _is_valid(opt_params_r, opt_side_r, n):
                        if opt_side_r < best_side:
                            best_side = opt_side_r
                            best_inner = [[float(opt_params_r[i,0]), float(opt_params_r[i,1]), float(opt_params_r[i,2])] for i in range(n)]
                            best_valid = True
                            
    except Exception as e:
        pass
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE - 5:
        # Emergency return
        return best_inner, (0.0, 0.0), best_side, 30.0
    
    # Final polish
    if best_valid:
        try:
            polish_params = np.zeros((n, 3))
            for i, (x, y, a) in enumerate(best_inner):
                polish_params[i] = [x, y, a]
            
            result_polish = minimize(
                _objective,
                polish_params.flatten(),
                args=(n,),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            
            elapsed = time.time() - start_time
            if elapsed < DEADLINE - 2:
                polish_params = result_polish.x.reshape(n, 3)
                polish_side, polish_pen = _fast_side_and_penalty(polish_params, n)
                
                if polish_pen < 1e-8:
                    if _is_valid(polish_params, polish_side, n):
                        if polish_side < best_side:
                            best_side = polish_side
                            best_inner = [[float(polish_params[i,0]), float(polish_params[i,1]), float(polish_params[i,2])] for i in range(n)]
        except Exception as e:
            pass
    
    return best_inner, (0.0, 0.0), best_side, 30.0
