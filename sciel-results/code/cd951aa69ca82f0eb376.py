import numpy as np
import time
from scipy.optimize import minimize

N = 11

# Vertices of the equilateral triangle
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])

# Precompute indices for all triples (i, j, k) with i < j < k
_IDX_TRIPLES = np.array([(i, j, k) for i in range(N) for j in range(i+1, N) for k in range(j+1, N)])
_IDX_I = _IDX_TRIPLES[:, 0]
_IDX_J = _IDX_TRIPLES[:, 1]
_IDX_K = _IDX_TRIPLES[:, 2]

def _min_triangle_area_vectorized(points):
    """
    Compute the smallest area among all triangles formed by triples of points, vectorized.
    points: (N, 2) array
    Returns: float min area
    """
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    areas = 0.5 * np.abs(cross)
    if len(areas) == 0:
        return 0.0
    return np.min(areas)

def _min_area_and_grad(points):
    """Returns (min_area, gradient of -min_area w.r.t. point coordinates)."""
    pi = points[_IDX_I]
    pj = points[_IDX_J]
    pk = points[_IDX_K]
    dx_j = pj[:, 0] - pi[:, 0]
    dy_j = pj[:, 1] - pi[:, 1]
    dx_k = pk[:, 0] - pi[:, 0]
    dy_k = pk[:, 1] - pi[:, 1]
    cross = dx_j * dy_k - dy_j * dx_k
    areas = 0.5 * np.abs(cross)
    min_idx = np.argmin(areas)
    min_area = areas[min_idx]
    grad = np.zeros((N, 2))
    s = np.sign(cross[min_idx])
    if s == 0:
        s = 1.0
    i = _IDX_I[min_idx]
    j = _IDX_J[min_idx]
    k = _IDX_K[min_idx]
    grad[i, 0] += -0.5 * s * (points[j,1] - points[k,1])
    grad[i, 1] += -0.5 * s * (points[k,0] - points[j,0])
    grad[j, 0] += -0.5 * s * (points[k,1] - points[i,1])
    grad[j, 1] += -0.5 * s * (-(points[k,0] - points[i,0]))
    grad[k, 0] += -0.5 * s * (-(points[j,1] - points[i,1]))
    grad[k, 1] += -0.5 * s * (points[j,0] - points[i,0])
    return min_area, grad

def _objective_and_grad(params):
    points = _param_to_points(params)
    min_area, grad_cart = _min_area_and_grad(points)
    s = params[0::2]
    t = params[1::2]
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    grad_params = np.zeros(2 * N)
    for i in range(N):
        dxd_s = 1.0 - 0.5 * t[i]
        dxd_t = -0.5 * s[i]
        dyd_s = t[i] * np.sqrt(3) / 2
        dyd_t = s[i] * np.sqrt(3) / 2
        grad_params[2*i] = grad_cart[i,0]*dxd_s + grad_cart[i,1]*dyd_s
        grad_params[2*i+1] = grad_cart[i,0]*dxd_t + grad_cart[i,1]*dyd_t
    return -min_area, grad_params

def _param_to_points(params):
    """
    Map parameters in [0, 1]^{2N} to points inside the triangle.
    Barycentric coordinates:
    lambda_0 = 1 - s_i
    lambda_1 = s_i * (1 - t_i)
    lambda_2 = s_i * t_i
    """
    s = params[0::2]
    t = params[1::2]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    lam1 = s * (1.0 - t)
    lam2 = s * t
    
    x = lam1 * V1[0] + lam2 * V2[0]
    y = lam1 * V1[1] + lam2 * V2[1]
    
    return np.column_stack([x, y])

def _points_to_param(points):
    """
    Map points in the triangle to parameters (s, t).
    """
    x = points[:, 0]
    y = points[:, 1]
    
    lam2 = 2.0 * y / np.sqrt(3)
    lam1 = x - 0.5 * lam2
    
    s = lam1 + lam2
    t = np.zeros_like(s)
    mask = s > 1e-12
    t[mask] = lam2[mask] / s[mask]
    
    s = np.clip(s, 0.0, 1.0)
    t = np.clip(t, 0.0, 1.0)
    
    params = np.zeros(2 * N)
    params[0::2] = s
    params[1::2] = t
    return params

def _objective(params):
    """Objective: negative min triangle area."""
    points = _param_to_points(params)
    return -_min_triangle_area_vectorized(points)

def _generate_fps_initialization(rng, n_points=N, n_candidates=500):
    """
    Generate an initial configuration using Greedy Farthest Point Sampling.
    """
    points = np.zeros((n_points, 2))
    
    # Start with centroid
    points[0] = (V0 + V1 + V2) / 3.0
    
    for k in range(1, n_points):
        r1 = rng.uniform(0, 1, n_candidates)
        r2 = rng.uniform(0, 1, n_candidates)
        mask = r1 + r2 > 1
        r1[mask] = 1 - r1[mask]
        r2[mask] = 1 - r2[mask]
        
        candidates = np.column_stack([
            r1 * V1[0] + r2 * V2[0],
            r1 * V1[1] + r2 * V2[1]
        ])
        
        diffs = candidates[:, np.newaxis, :] - points[np.newaxis, :k, :]
        dists = np.linalg.norm(diffs, axis=2)
        min_dists = np.min(dists, axis=1)
        
        best_idx = np.argmax(min_dists)
        points[k] = candidates[best_idx]
        
    return points

def _structured_inits(rng):
    """
    Generate structured initializations based on grid or symmetric patterns.
    """
    inits = []
    
    # Grid-like patterns
    for grid_n in [3, 4, 5]:
        # Generate points on a triangular grid
        points = []
        for i in range(grid_n):
            for j in range(grid_n - i):
                # Barycentric coordinates
                l0 = 1 - (i + j) / (grid_n - 1) if grid_n > 1 else 1
                l1 = i / (grid_n - 1) if grid_n > 1 else 0
                l2 = j / (grid_n - 1) if grid_n > 1 else 0
                
                # Scale to fit exactly N points by sampling
                # This is a rough approach; we'll just add points and then sample
                pt = l0 * V0 + l1 * V1 + l2 * V2
                points.append(pt)
        
        # If we have more than N points, randomly select N
        points = np.array(points)
        if len(points) >= N:
            idx = rng.choice(len(points), N, replace=False)
            inits.append(points[idx])
        elif len(points) > 0:
            # Pad with random points if needed
            while len(points) < N:
                r1 = rng.uniform(0, 1)
                r2 = rng.uniform(0, 1)
                if r1 + r2 > 1:
                    r1 = 1 - r1
                    r2 = 1 - r2
                pt = r1 * V1 + r2 * V2
                points = np.vstack([points, pt])
            inits.append(points)
    
    # Symmetric patterns: 1 center + 10 around
    # 1 center
    center = (V0 + V1 + V2) / 3.0
    # 5 at radius 0.3, 5 at radius 0.6
    pts = [center]
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False):
        r = 0.3
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        # Check if inside triangle, if not, project or skip
        # Simple check: barycentric
        # For now, just add and rely on clipping later
        pts.append(pt)
    for angle in np.linspace(0, 2*np.pi, 5, endpoint=False) + np.pi/5:
        r = 0.6
        pt = center + r * np.array([np.cos(angle), np.sin(angle)])
        pts.append(pt)
    
    inits.append(np.array(pts))
    
    return inits

def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    def time_left():
        return deadline - time.time()

    best_params = None
    best_score = -np.inf
    
    rng = np.random.RandomState(42)
    
    initial_params = []
    
    # 1. Generate diverse initializations
    for i in range(15):
        if time_left() < 40.0:
            break
        pts = _generate_fps_initialization(rng)
        params = _points_to_param(pts)
        initial_params.append(params)
        
    # Add structured inits
    for pts in _structured_inits(rng):
        if time_left() < 40.0:
            break
        params = _points_to_param(pts)
        initial_params.append(params)

    # Add random initializations
    for i in range(10):
        if time_left() < 40.0:
            break
        s = rng.uniform(0.1, 0.9, size=N)
        t = rng.uniform(0.1, 0.9, size=N)
        params = np.zeros(2 * N)
        params[0::2] = s
        params[1::2] = t
        initial_params.append(params)

    bounds = [(0.0, 1.0)] * (2 * N)
    
    # 2. Optimize each initialization using SLSQP with gradients
    for start_p in initial_params:
        if time_left() < 25.0:
            break
            
        try:
            res = minimize(
                _objective_and_grad, start_p, method='SLSQP', bounds=bounds,
                jac=True,
                options={'maxiter': 300, 'ftol': 1e-10, 'iprint': 0}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    # 3. Iterative refinement with decreasing perturbation
    if best_params is not None:
        for scale in [0.05, 0.02, 0.01, 0.005]:
            if time_left() < 20.0:
                break
            for i in range(8):
                if time_left() < 15.0:
                    break
                try:
                    noise = rng.normal(0, scale, size=2*N)
                    perturbed = np.clip(best_params + noise, 0.0, 1.0)
                    res = minimize(
                        _objective_and_grad, perturbed, method='SLSQP', bounds=bounds,
                        jac=True,
                        options={'maxiter': 200, 'ftol': 1e-10, 'iprint': 0}
                    )
                    score_res = -res.fun
                    if score_res > best_score:
                        best_score = score_res
                        best_params = res.x.copy()
                except Exception:
                    pass

    # 4. Final Nelder-Mead polish in parameter space
    if best_params is not None and time_left() > 25.0:
        try:
            res = minimize(
                _objective, best_params, method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            score_res = -res.fun
            if score_res > best_score:
                best_score = score_res
                best_params = res.x.copy()
        except Exception:
            pass

    if best_params is None:
        best_params = np.zeros(2 * N)
        best_params[0::2] = 0.5
        best_params[1::2] = 0.5

    points = _param_to_points(best_params)
    
    # Final safety check: ensure all points are strictly inside or on the boundary
    # Clip to triangle
    def _clip_to_triangle(pts):
        # Barycentric check
        # For each point, compute barycentric coords
        # If any < 0, project to nearest edge
        # This is a simple clipping
        for i in range(len(pts)):
            x, y = pts[i]
            # Barycentric
            # l0 = 1 - x - y/sqrt(3)  ? No.
            # Using V0=(0,0), V1=(1,0), V2=(0.5, sqrt(3)/2)
            # Area of triangle = sqrt(3)/4
            # l0 = area(P, V1, V2) / Area(T)
            # l1 = area(V0, P, V2) / Area(T)
            # l2 = area(V0, V1, P) / Area(T)
            
            # Cross products
            # area(V0, V1, V2) = 0.5 * | (1,0) x (0.5, sqrt(3)/2) | = 0.5 * sqrt(3)/2 = sqrt(3)/4
            
            # l2 = area(V0, V1, P) = 0.5 * | (1,0) x (x,y) | = 0.5 * y
            # l2 = y / (sqrt(3)/2) = 2y/sqrt(3)
            # l1 = area(V0, P, V2) = 0.5 * | (x,y) x (0.5, sqrt(3)/2) | = 0.5 * (x*sqrt(3)/2 - y*0.5)
            # l1 = (x*sqrt(3)/2 - y/2) / (sqrt(3)/2) = x - y/sqrt(3)
            # l0 = 1 - l1 - l2
            
            l2 = 2 * y / np.sqrt(3)
            l1 = x - y / np.sqrt(3)
            l0 = 1 - l1 - l2
            
            if l0 < 0 or l1 < 0 or l2 < 0:
                # Project to nearest edge
                # This is complex. For now, just clip coordinates roughly.
                # If l0 < 0, point is beyond edge V1-V2
                # If l1 < 0, point is beyond edge V0-V2
                # If l2 < 0, point is beyond edge V0-V1
                
                # Simple fix: set negative barycentric to 0 and renormalize
                l0 = max(0, l0)
                l1 = max(0, l1)
                l2 = max(0, l2)
                s = l0 + l1 + l2
                if s > 0:
                    l0 /= s
                    l1 /= s
                    l2 /= s
                else:
                    l0 = 1
                    l1 = 0
                    l2 = 0
                
                pts[i] = l0 * V0 + l1 * V1 + l2 * V2
        
        return pts

    points = _clip_to_triangle(points)
    
    return np.array(points)
