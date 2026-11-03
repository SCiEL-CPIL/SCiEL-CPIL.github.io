import time
import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull

N = 14

# Precompute indices for all combinations of 3 points
_indices = np.arange(N)
IDX_I, IDX_J, IDX_K = np.meshgrid(_indices, _indices, _indices, indexing='ij')
mask = (IDX_I < IDX_J) & (IDX_J < IDX_K)
IDX_I = IDX_I[mask].flatten()
IDX_J = IDX_J[mask].flatten()
IDX_K = IDX_K[mask].flatten()

def _compute_min_triangle_area(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return np.min(areas)

def _compute_hull_area(points):
    try:
        if np.allclose(points[:, 0], points[0, 0]) or np.allclose(points[:, 1], points[0, 1]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_area = _compute_min_triangle_area(points)
    return min_area / hull_area

def _all_areas(points):
    pi = points[IDX_I]
    pj = points[IDX_J]
    pk = points[IDX_K]
    
    v1x = pj[:, 0] - pi[:, 0]
    v1y = pj[:, 1] - pi[:, 1]
    v2x = pk[:, 0] - pi[:, 0]
    v2y = pk[:, 1] - pi[:, 1]
    
    areas = 0.5 * np.abs(v1x * v2y - v1y * v2x)
    return areas

def _objective(flat_points):
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    return -_score(pts)

def _objective_soft(flat_points, beta=50.0):
    """Soft-min objective: differentiable approximation of -min_area/hull_area."""
    pts = flat_points.reshape(N, 2)
    if np.any(~np.isfinite(pts)):
        return 1e10
    areas = _all_areas(pts)
    if np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min = -log_sum / beta
    if soft_min < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(pts)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min

# --- Convex polygon parameterization via edge lengths and angles ---
# N points as convex polygon vertices. Parameterize by:
#   - N edge lengths (positive)
#   - N-1 internal angles (the last is determined by sum = (N-2)*pi)
# This ensures strict convexity by construction.

def _poly_params_to_points(params):
    """
    Convert edge-length + angle parameters to polygon vertices.
    params: array of length 2*N-1
      params[0:N] = edge lengths (positive)
      params[N:2*N-1] = internal angles at vertices 0..N-2 (radians)
      Internal angle at vertex N-1 = (N-2)*pi - sum(params[N:2*N-1])
    Returns: (N, 2) array of vertices in CCW order, or None if invalid.
    """
    n = N
    edge_lengths = params[:n]
    angles = params[n:2*n-1]
    
    # Ensure positive edge lengths
    if np.any(edge_lengths <= 1e-10):
        return None
    
    # Compute last angle
    total_angle = (n - 2) * np.pi
    last_angle = total_angle - np.sum(angles)
    
    # All internal angles must be in (0, pi) for strict convexity
    all_angles = np.append(angles, last_angle)
    if np.any(all_angles <= 1e-8) or np.any(all_angles >= np.pi - 1e-8):
        return None
    
    # External turning angles: pi - internal_angle
    # For a convex polygon traversed CCW, external angles are positive and sum to 2*pi
    ext_angles = np.pi - all_angles  # each in (0, pi)
    # Sum of external angles should be 2*pi
    # Due to floating point, normalize
    ext_sum = np.sum(ext_angles)
    if abs(ext_sum - 2*np.pi) > 1e-6:
        # Scale to make sum exactly 2*pi (small correction)
        ext_angles = ext_angles * (2*np.pi / ext_sum)
    
    # Build polygon by walking along edges with turning angles
    # Start at origin, first edge along positive x-axis
    vertices = np.zeros((n, 2))
    current_angle = 0.0  # direction of first edge
    
    for i in range(n):
        if i > 0:
            current_angle += ext_angles[i-1]  # turn by external angle at vertex i-1
        # Actually, let's think more carefully:
        # Vertex 0 is at origin. Edge from vertex 0 to vertex 1 has direction 0.
        # At vertex 1, we turn by external angle ext_angles[1] (which is pi - internal_angle_1)
        # Wait, the external angle at vertex i is the angle you turn when going from edge (i-1,i) to edge (i,i+1)
        # For CCW traversal, you turn left by (pi - internal_angle_i)
        
        # Let me redo: 
        # Edge i goes from vertex i to vertex i+1 (mod n)
        # Direction of edge 0: 0 radians
        # At vertex i (for i >= 1), the direction changes by external angle
        # Direction of edge i = sum of external angles at vertices 1..i (for i >= 1)
        # Actually direction of edge i = sum_{j=1}^{i} ext_angles[j] for i >= 1, and 0 for i=0
        
        pass
    
    # Let me use a cleaner approach:
    # direction[i] = angle of edge from vertex i to vertex i+1
    # direction[0] = 0
    # direction[i] = direction[i-1] + ext_angles[i] for i >= 1 (turning at vertex i)
    # Wait, that's not right either. Let me think step by step.
    
    # Walking CCW around a convex polygon:
    # - Start at vertex 0, go along edge 0 to vertex 1
    # - At vertex 1, turn left by external angle ext_angles[1]
    # - Go along edge 1 to vertex 2
    # - At vertex 2, turn left by external angle ext_angles[2]
    # - ...
    # - At vertex n-1, turn left by ext_angles[n-1]
    # - Go along edge n-1 back to vertex 0
    
    # So direction of edge i (from vertex i to vertex i+1 mod n):
    # dir[0] = 0
    # dir[i] = dir[i-1] + ext_angles[i] for i = 1, ..., n-1
    
    directions = np.zeros(n)
    for i in range(1, n):
        directions[i] = directions[i-1] + ext_angles[i]
    
    # Now compute vertices
    vertices = np.zeros((n, 2))
    for i in range(n):
        if i > 0:
            vertices[i] = vertices[i-1] + edge_lengths[i-1] * np.array([np.cos(directions[i-1]), np.sin(directions[i-1])])
    
    # Check closure: vertices[n-1] + edge_lengths[n-1] * (cos(dir[n-1]), sin(dir[n-1])) should equal vertices[0]
    # Due to the angle constraint, this should close up. But let's verify.
    closure_error = np.linalg.norm(vertices[0] - (vertices[-1] + edge_lengths[-1] * np.array([np.cos(directions[-1]), np.sin(directions[-1])])))
    if closure_error > 0.1 * np.mean(edge_lengths):
        # Polygon doesn't close well; this parameterization is inconsistent
        # This can happen if the angles don't correspond to a valid polygon with these edge lengths
        # For now, return None (invalid)
        return None
    
    # Center and normalize
    center = np.mean(vertices, axis=0)
    vertices = vertices - center
    max_norm = np.max(np.linalg.norm(vertices, axis=1))
    if max_norm > 0:
        vertices = vertices / max_norm
    
    return vertices

def _points_to_poly_params(points):
    """
    Convert N convex polygon vertices (CCW order) to edge-length + angle parameters.
    Returns array of length 2*N-1, or None if points are not a valid convex polygon.
    """
    n = N
    pts = points.copy()
    
    # Ensure CCW order
    # Compute signed area
    x = pts[:, 0]
    y = pts[:, 1]
    area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    if area2 < 0:
        pts = pts[::-1]  # reverse to make CCW
        x = pts[:, 0]
        y = pts[:, 1]
        area2 = -area2
    
    if area2 < 1e-10:
        return None
    
    # Compute edge lengths
    edge_lengths = np.zeros(n)
    for i in range(n):
        next_i = (i + 1) % n
        edge_lengths[i] = np.linalg.norm(pts[next_i] - pts[i])
    
    if np.any(edge_lengths < 1e-10):
        return None
    
    # Compute internal angles
    # Internal angle at vertex i: angle between edge (i-1 -> i) and edge (i -> i+1)
    # For CCW polygon, this is the angle you turn through
    angles = np.zeros(n)
    for i in range(n):
        prev_i = (i - 1) % n
        next_i = (i + 1) % n
        v1 = pts[i] - pts[prev_i]  # incoming edge direction
        v2 = pts[next_i] - pts[i]  # outgoing edge direction
        
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        # The internal angle is pi - turn_angle, where turn_angle is the external angle
        # For CCW convex polygon, the turn is to the left
        # The angle between -v1 and v2 is the external angle
        # Internal angle = pi - external_angle
        # Actually: the angle from -v1 (reversed incoming) to v2 (outgoing) is the external turn
        # Let's compute it directly:
        # The internal angle at vertex i is the angle between vectors (prev_i - i) and (next_i - i)
        w1 = pts[prev_i] - pts[i]
        w2 = pts[next_i] - pts[i]
        cos_int = np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2))
        cos_int = np.clip(cos_int, -1.0, 1.0)
        angles[i] = np.arccos(cos_int)
    
    # Verify convexity: all internal angles in (0, pi)
    if np.any(angles <= 1e-8) or np.any(angles >= np.pi - 1e-8):
        return None
    
    # Check that angles sum to (n-2)*pi
    if abs(np.sum(angles) - (n-2)*np.pi) > 1e-6:
        return None
    
    # Return first n-1 angles (the last is determined)
    return np.concatenate([edge_lengths, angles[:n-1]])

def _objective_poly(params):
    """Objective for polygon parameterization."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _poly_params_to_points(params)
    if points is None:
        return 1e10
    if np.any(~np.isfinite(points)):
        return 1e10
    return -_score(points)

def _objective_poly_soft(params, beta=100.0):
    """Soft objective for polygon parameterization."""
    if np.any(~np.isfinite(params)):
        return 1e10
    points = _poly_params_to_points(params)
    if points is None:
        return 1e10
    areas = _all_areas(points)
    if len(areas) == 0 or np.min(areas) < 1e-12:
        return 1e10
    z = -beta * areas
    max_z = np.max(z)
    if max_z < -500:
        log_sum = max_z
    else:
        log_sum = max_z + np.log(np.sum(np.exp(z - max_z)))
    soft_min_val = -log_sum / beta
    if soft_min_val < 1e-12:
        return 1e10
    hull_area = _compute_hull_area(points)
    if hull_area < 1e-10:
        return 1e10
    return hull_area / soft_min_val

def _generate_poly_inits():
    """Generate initial polygon parameter vectors."""
    inits = []
    
    # Regular 14-gon
    n = N
    ext_angle = 2 * np.pi / n
    internal_angle = np.pi - ext_angle
    edge_len = 2 * np.sin(np.pi / n)  # for unit circumradius
    
    base_edges = np.full(n, edge_len)
    base_angles = np.full(n-1, internal_angle)
    inits.append(np.concatenate([base_edges, base_angles]))
    
    # Slightly perturbed regular
    rng = np.random.default_rng(42)
    for seed in range(20):
        rng = np.random.default_rng(seed * 13 + 7)
        edges = edge_len * (1.0 + rng.normal(0, 0.1, n))
        edges = np.maximum(edges, 0.1)
        angles = internal_angle * (1.0 + rng.normal(0, 0.05, n-1))
        # Ensure sum constraint is satisfiable
        angles = np.clip(angles, 0.1, np.pi - 0.1)
        inits.append(np.concatenate([edges, angles]))
    
    # Alternating edge lengths
    for ratio in [0.7, 0.8, 0.85, 0.9]:
        edges = np.ones(n)
        edges[::2] = ratio
        angles = np.full(n-1, internal_angle)
        inits.append(np.concatenate([edges, angles]))
    
    # Non-uniform angles (still convex)
    for pattern in [
        np.linspace(0.9 * np.pi, 1.1 * np.pi, n-1),
        np.concatenate([np.full(n//2, 0.95*np.pi), np.full(n-1-n//2, 1.05*np.pi)]),
    ]:
        edges = np.full(n, edge_len)
        # Adjust edges to try to close the polygon
        inits.append(np.concatenate([edges, pattern]))
    
    # Random convex polygons
    for seed in range(15):
        rng = np.random.default_rng(seed * 23 + 3)
        # Generate random convex polygon by random points on unit circle, sorted by angle
        theta = np.sort(rng.uniform(0, 2*np.pi, n))
        radii = 1.0 + rng.normal(0, 0.1, n)
        radii = np.maximum(radii, 0.5)
        pts = np.column_stack([radii * np.cos(theta), radii * np.sin(theta)])
        params = _points_to_poly_params(pts)
        if params is not None:
            inits.append(params)
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    best_points = None
    best_score = -np.inf

    def update_best(pts, score):
        nonlocal best_points, best_score
        if score > best_score:
            best_score = score
            best_points = pts.copy()

    def refine(points_init, method='Nelder-Mead', maxiter=2000, tol=1e-8):
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'Nelder-Mead':
                options = {'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
                res = minimize(_objective, pts.flatten(), method='Nelder-Mead', options=options)
            elif method == 'Powell':
                res = minimize(_objective, pts.flatten(), method='Powell', 
                               options={'maxiter': maxiter, 'xatol': 1e-8, 'fatol': 1e-8})
            elif method == 'L-BFGS-B':
                res = minimize(_objective, pts.flatten(), method='L-BFGS-B', 
                               options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8})
            else:
                return None, -np.inf
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    def refine_soft(points_init, method='L-BFGS-B', maxiter=3000, beta=50.0):
        """Refine using soft-min objective for smoother landscape."""
        pts = points_init.copy()
        center = np.mean(pts, axis=0)
        pts = pts - center
        max_norm = np.max(np.linalg.norm(pts, axis=1))
        if max_norm > 0:
            pts = pts / max_norm
            
        try:
            if method == 'L-BFGS-B':
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='L-BFGS-B', options={'maxiter': maxiter, 'ftol': 1e-14, 'gtol': 1e-10})
            elif method == 'Powell':
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='Powell', options={'maxiter': maxiter, 'xtol': 1e-11, 'ftol': 1e-13})
            else:
                res = minimize(lambda x: _objective_soft(x, beta=beta), pts.flatten(), 
                               method='Nelder-Mead', options={'maxiter': maxiter, 'xatol': 1e-9, 'fatol': 1e-11, 'adaptive': True})
                
            cand_pts = res.x.reshape(N, 2)
            cand_score = _score(cand_pts)
            return cand_pts, cand_score
        except Exception:
            return None, -np.inf

    # 1. Initializations
    initializations = []
    
    # Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x_reg = np.column_stack((np.cos(angles), np.sin(angles)))
    initializations.append(x_reg)
    
    # Two concentric rings (7+7) with various radii and offsets
    for r1 in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85]:
        a1 = np.linspace(0, 2*np.pi, 7, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 7, endpoint=False) + np.pi/7
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            np.column_stack([np.cos(a2), np.sin(a2)])
        ])
        initializations.append(pts)
    
    # Three rings (5+4+5)
    for r1, r2 in [(0.3, 0.6), (0.4, 0.75), (0.5, 0.85)]:
        a1 = np.linspace(0, 2*np.pi, 5, endpoint=False)
        a2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + np.pi/4
        a3 = np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5
        pts = np.vstack([
            r1 * np.column_stack([np.cos(a1), np.sin(a1)]),
            r2 * np.column_stack([np.cos(a2), np.sin(a2)]),
            np.column_stack([np.cos(a3), np.sin(a3)])
        ])
        initializations.append(pts)
    
    # Elliptical configurations
    for rx, ry in [(1.0, 0.6), (1.0, 0.8), (1.2, 0.7), (1.1, 0.9)]:
        a = np.linspace(0, 2*np.pi, N, endpoint=False)
        initializations.append(np.column_stack([rx*np.cos(a), ry*np.sin(a)]))
    
    # Halton sequence
    def halton(index, base):
        f = 1
        r = 0
        while index:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    halton_pts = []
    for i in range(1, N+1):
        x = halton(i, 2)
        y = halton(i, 3)
        halton_pts.append([x, y])
    x_halton = np.array(halton_pts)
    initializations.append(x_halton)
    
    # Random points
    rng = np.random.default_rng(123)
    for _ in range(3):
        r = np.sqrt(rng.uniform(0, 1, N))
        theta = rng.uniform(0, 2*np.pi, N)
        x_rand = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        initializations.append(x_rand)

    # 2. Refine initializations
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 25.0:
            break
            
        cand_pts, cand_score = refine(init_pts, 'Nelder-Mead', maxiter=1500)
        if cand_pts is not None:
            update_best(cand_pts, cand_score)
            
        if time.time() < deadline - 25.0:
            cand_pts2, cand_score2 = refine(cand_pts, 'Powell', maxiter=500)
            if cand_pts2 is not None:
                update_best(cand_pts2, cand_score2)

    # 3. Polygon parameterization optimization (KEY STRUCTURAL CHANGE)
    # Optimize edge lengths and angles of a convex polygon
    if time.time() < deadline - 30.0:
        poly_inits = _generate_poly_inits()
        
        for i, flat_init in enumerate(poly_inits):
            if time.time() > deadline - 25.0:
                break
            
            # Check if initial params produce valid points
            test_pts = _poly_params_to_points(flat_init)
            if test_pts is None:
                continue
            
            # Nelder-Mead on polygon parameters
            try:
                res_poly = minimize(_objective_poly, flat_init, method='Nelder-Mead',
                                    options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True})
                if res_poly.fun < 0:
                    pts14 = _poly_params_to_points(res_poly.x)
                    if pts14 is not None:
                        cand_score = -res_poly.fun
                        update_best(pts14, cand_score)
            except Exception:
                pass
            
            # Powell on polygon parameters
            if time.time() < deadline - 20.0:
                try:
                    res_poly_pow = minimize(_objective_poly, flat_init, method='Powell',
                                            options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8})
                    if res_poly_pow.fun < 0:
                        pts14 = _poly_params_to_points(res_poly_pow.x)
                        if pts14 is not None:
                            cand_score = -res_poly_pow.fun
                            update_best(pts14, cand_score)
                except Exception:
                    pass

    # 4. Differential Evolution on polygon space (27D)
    if time.time() < deadline - 20.0:
        def de_poly_objective(flat):
            if np.any(~np.isfinite(flat)):
                return 1.0
            pts = _poly_params_to_points(flat)
            if pts is None:
                return 1.0
            if np.allclose(pts, pts[0]):
                return 1.0
            return -_score(pts)
            
        try:
            # Edge lengths: (0.1, 2.0)
            # Angles: (0.1, pi-0.1)
            bounds = [(0.1, 2.0)] * N + [(0.1, np.pi - 0.1)] * (N-1)
            res_de = differential_evolution(
                de_poly_objective, 
                bounds, 
                seed=42, 
                maxiter=30, 
                popsize=8, 
                tol=1e-4, 
                mutation=(0.5, 1.0), 
                recombination=0.9,
                polish=True,
                init='latinhypercube'
            )
            if res_de.success or res_de.fun < 0:
                pts14 = _poly_params_to_points(res_de.x)
                if pts14 is not None:
                    cand_score = -res_de.fun
                    update_best(pts14, cand_score)
        except Exception:
            pass

    # 5. Final polish on the global best
    if best_points is not None and time.time() < deadline - 10.0:
        for method in ['Nelder-Mead', 'Powell']:
            if time.time() > deadline - 10.0:
                break
            cand_pts, cand_score = refine(best_points, method, maxiter=1000)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)
        
        # Soft refinement
        if time.time() < deadline - 8.0:
            cand_pts, cand_score = refine_soft(best_points, 'L-BFGS-B', maxiter=2000, beta=50.0)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    # 6. Random restarts if time allows
    if best_points is not None and time.time() < deadline - 15.0:
        rng2 = np.random.default_rng(42)
        for _ in range(5):
            if time.time() > deadline - 10.0:
                break
            noise = rng2.normal(0, 0.1, size=(N, 2))
            perturbed = best_points + noise
            cand_pts, cand_score = refine(perturbed, 'Nelder-Mead', maxiter=500)
            if cand_pts is not None:
                update_best(cand_pts, cand_score)

    if best_points is None:
        best_points = initializations[0]
        
    return best_points
