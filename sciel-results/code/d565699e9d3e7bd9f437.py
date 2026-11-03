"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull


def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _precise_circumradius_angles(points):
    """
    Compute the precise spherical circumradius angle for each facet of the convex hull.
    Uses the circumsphere center via perpendicular bisector plane intersection.
    Returns array of facet angles in radians.
    """
    try:
        hull = ConvexHull(points)
    except:
        return np.array([np.pi])
    
    simplices = hull.simplices
    p1 = points[simplices[:, 0]]
    p2 = points[simplices[:, 1]]
    p3 = points[simplices[:, 2]]
    
    n_facets = len(simplices)
    angles = np.zeros(n_facets)
    
    for i in range(n_facets):
        a = p1[i]
        b = p2[i]
        c = p3[i]
        
        # Edge vectors
        e1 = b - a
        e2 = c - a
        e3 = c - b
        
        # Facet normal (outward)
        n = np.cross(e1, e2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-12:
            angles[i] = np.pi
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
        
        # Perpendicular bisector planes:
        # (x - (a+b)/2) . (b-a) = 0  =>  x . e1 = (a+b) . e1 / 2
        # (x - (a+c)/2) . (c-a) = 0  =>  x . e2 = (a+c) . e2 / 2
        # (x - (b+c)/2) . (c-b) = 0  =>  x . e3 = (b+c) . e3 / 2
        
        # Solve: A @ center = rhs, where rows of A are e1, e2, e3
        A = np.array([e1, e2, e3])
        rhs = np.array([
            np.dot(a + b, e1) / 2.0,
            np.dot(a + c, e2) / 2.0,
            np.dot(b + c, e3) / 2.0
        ])
        
        try:
            center = np.linalg.solve(A, rhs)
        except:
            angles[i] = np.pi
            continue
        
        center_norm = np.linalg.norm(center)
        if center_norm < 1e-12:
            angles[i] = np.pi
            continue
        
        # The circumcenter direction is center / |center|
        # The angular distance from the facet's circumcenter to any vertex
        cos_angle = np.clip(np.dot(center / center_norm, a), -1.0, 1.0)
        angles[i] = np.arccos(cos_angle)
    
    return angles


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of unit points.
    Uses precise convex hull facet circumradii.
    """
    if len(points) < 3:
        return 180.0
    angles = _precise_circumradius_angles(points)
    if len(angles) == 0:
        return 180.0
    return float(np.max(angles)) * 180.0 / np.pi


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _generate_test_grid(n_theta=40, n_phi=80):
    """Generate a grid of test points on the sphere."""
    thetas = np.linspace(0, np.pi, n_theta)
    phis = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    theta_grid, phi_grid = np.meshgrid(thetas, phis, indexing='ij')
    x = np.sin(theta_grid) * np.cos(phi_grid)
    y = np.sin(theta_grid) * np.sin(phi_grid)
    z = np.cos(theta_grid)
    grid = np.stack([x, y, z], axis=-1)
    return grid.reshape(-1, 3)


def _smooth_surrogate(params, beta=200.0, grid=None):
    """
    Vectorized smooth differentiable surrogate for the covering radius.
    Uses a grid of test points and log-sum-exp for soft min/max.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    if grid is None:
        grid = _generate_test_grid()
        
    dots = np.clip(grid @ pts.T, -1.0, 1.0)
    angles = np.arccos(dots)
    
    neg_beta_angles = -beta * angles
    max_neg = np.max(neg_beta_angles, axis=1, keepdims=True)
    exp_terms = np.exp(neg_beta_angles - max_neg)
    sum_exp = np.sum(exp_terms, axis=1)
    logsumexp_neg = max_neg.squeeze() + np.log(sum_exp)
    soft_min_angles = -logsumexp_neg / beta
    
    beta_soft_min = beta * soft_min_angles
    max_val = np.max(beta_soft_min)
    logsumexp_max = max_val + np.log(np.sum(np.exp(beta_soft_min - max_val)))
    soft_max = logsumexp_max / beta
    
    return soft_max


def _facet_smooth_surrogate(params, beta=100.0):
    """
    Smooth surrogate based on precise convex hull facet circumradii.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
    
    facet_angles = _precise_circumradius_angles(pts)
    
    if len(facet_angles) == 0:
        return np.pi
    
    beta_facet = beta * facet_angles
    max_val = np.max(beta_facet)
    logsumexp = max_val + np.log(np.sum(np.exp(beta_facet - max_val)))
    soft_max = logsumexp / beta
    
    return soft_max


def _smooth_gradient_optimize(points, time_limit=30, beta=200.0, use_facet=False, grid=None):
    """
    Gradient-based optimization using smooth surrogate with L-BFGS-B.
    """
    start_time = time.time()
    
    if use_facet:
        grid = None
    elif grid is None:
        grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        if use_facet:
            return _facet_smooth_surrogate(params, beta=beta)
        else:
            return _smooth_surrogate(params, beta=beta, grid=grid)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': 1500, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 8000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _fibonacci_points():
    """Generate Fibonacci spiral points on sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _jittered_fibonacci_points(seed=123, jitter=0.05):
    """Generate Fibonacci points with small random jitter to break symmetries."""
    pts = _fibonacci_points()
    rng = np.random.RandomState(seed)
    noise = rng.randn(N, 3) * jitter
    pts = pts + noise
    return _normalize(pts)


def _random_points(seed=42):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    points = rng.randn(N, 3)
    return _normalize(points)


def _icosahedron_based_points():
    """Generate points based on icosahedron vertices with perturbations."""
    phi = (1 + np.sqrt(5)) / 2
    verts = np.array([
        [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
        [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
        [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1]
    ], dtype=float)
    verts = _normalize(verts)
    
    rng = np.random.RandomState(123)
    points = [verts[i] for i in range(12)]
    
    while len(points) < N:
        base = verts[rng.randint(12)]
        noise = rng.randn(3) * 0.3
        new_pt = base + noise
        new_pt = new_pt / np.linalg.norm(new_pt)
        min_dist = min(np.linalg.norm(new_pt - p) for p in points)
        if min_dist > 0.1:
            points.append(new_pt)
    
    return np.array(points[:N])


def _spherical_lloyd_iteration(points, max_iter=100, time_limit=30, damping=0.3):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation) with adaptive damping.
    Uses precise circumradius angles for facet weighting.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    current_damping = damping
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        simplices = hull.simplices
        p1 = points[simplices[:, 0]]
        p2 = points[simplices[:, 1]]
        p3 = points[simplices[:, 2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n, axis=1, keepdims=True)
        mask = norm_n[:, 0] > 1e-10
        if not np.any(mask):
            break
        
        n_valid = n[mask] / norm_n[mask]
        p1_valid = p1[mask]
        
        dots = np.sum(n_valid * p1_valid, axis=1)
        n_valid = n_valid.copy()
        n_valid[dots < 0] *= -1
        
        # Use precise circumradius angles
        all_angles = _precise_circumradius_angles(points)
        facet_angles = all_angles[mask]
        
        # Accumulate targets for each point using arrays
        acc = np.zeros((N, 3))
        weights = np.zeros(N)
        
        simplices_valid = simplices[mask]
        for fi in range(len(simplices_valid)):
            angle = facet_angles[fi]
            n_vec = n_valid[fi]
            idxs = simplices_valid[fi]
            acc[idxs] += angle * n_vec
            weights[idxs] += angle
        
        new_points = points.copy()
        for pi in range(N):
            if weights[pi] > 1e-10:
                target = acc[pi] / np.linalg.norm(acc[pi])
                candidate = points[pi] + current_damping * (target - points[pi])
                candidate = candidate / np.linalg.norm(candidate)
                new_points[pi] = candidate
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
            current_damping = min(0.8, current_damping * 1.1)
        else:
            current_damping *= 0.5
            if current_damping < 0.01:
                break
    
    return best_points


def _nelder_mead_optimize(points, time_limit=20, maxiter=1000):
    """
    Local optimization using Nelder-Mead on the true covering radius.
    """
    start_time = time.time()
    
    def objective(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pts = params.reshape(N, 3)
        pts = _normalize(pts)
        return _covering_radius_radians(pts)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': maxiter, 'xatol': 1e-10, 'fatol': 1e-10})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _block_coordinate_descent(points, time_limit=30, max_rounds=10):
    """
    Block coordinate descent: optimize each point individually while keeping others fixed.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for round_num in range(max_rounds):
        if time.time() - start_time > time_limit:
            break
        
        for idx in range(N):
            if time.time() - start_time > time_limit:
                break
            
            other_points = np.delete(points, idx, axis=0)
            
            def objective_single(params):
                pt = params / np.linalg.norm(params)
                all_pts = np.vstack([other_points, pt.reshape(1, 3)])
                return _covering_radius_radians(all_pts)
            
            x0 = points[idx]
            result = minimize(objective_single, x0, method='Nelder-Mead',
                              options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8})
            new_pt = result.x / np.linalg.norm(result.x)
            points[idx] = new_pt
        
        new_cost = _covering_radius(points)
        if new_cost < best_cost:
            best_cost = new_cost
            best_points = points.copy()
    
    return best_points


def _sa_optimize_covering(points, T_start=1.0, T_end=1e-4, n_iter=5000, time_limit=100):
    """
    Simulated Annealing to minimize covering radius directly.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        if time.time() - start_time > time_limit:
            break
        T = T_start * (cooling ** i)
        
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        step_size = 0.05 * T / T_start
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        points[idx] = new_point
        
        new_cost = _covering_radius(points)
        delta_C = new_cost - current_cost
        
        if delta_C < 0 or np.random.rand() < np.exp(-delta_C / T):
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            points[idx] = original_point
            
    return best_points


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Precompute grid for smooth surrogate
    grid = _generate_test_grid(n_theta=40, n_phi=80)
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Jittered Fibonacci (breaks symmetry, often better start)
    initializations.append(_jittered_fibonacci_points(seed=123, jitter=0.05))
    initializations.append(_jittered_fibonacci_points(seed=456, jitter=0.1))
    initializations.append(_jittered_fibonacci_points(seed=789, jitter=0.08))
    
    # Random with different seeds
    for seed in [42, 123, 777, 999, 1234]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Stage 1: Global Structure via Lloyd's
    for init_pts in initializations:
        if _time_left() < 40:
            break
        
        pts = _spherical_lloyd_iteration(init_pts, max_iter=60, time_limit=min(12, _time_left() - 35), damping=0.3)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Stage 2: Smooth Gradient Optimization (Moderate Beta)
    if _time_left() > 35:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(12, _time_left() - 30), beta=100.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Stage 3: Block Coordinate Descent
    if _time_left() > 30:
        pts = _block_coordinate_descent(best_points, time_limit=min(15, _time_left() - 25), max_rounds=5)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Stage 4: High-Precision Polish (High Beta)
    if _time_left() > 25:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(10, _time_left() - 20), beta=500.0, use_facet=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Stage 5: Nelder-Mead on True Objective
    if _time_left() > 20:
        pts = _nelder_mead_optimize(best_points, time_limit=min(10, _time_left() - 15))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Stage 6: Simulated Annealing (Escape Local Minima)
    if _time_left() > 30:
        pts = _sa_optimize_covering(best_points, T_start=0.3, T_end=1e-4, n_iter=5000, time_limit=min(15, _time_left() - 25))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
        
        if cost < best_cost:
            pts = _spherical_lloyd_iteration(best_points, max_iter=30, time_limit=min(5, _time_left() - 15), damping=0.2)
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # Stage 7: Final Block Coordinate Descent
    if _time_left() > 15:
        pts = _block_coordinate_descent(best_points, time_limit=min(8, _time_left() - 10), max_rounds=3)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Stage 8: Final High-Precision Nelder-Mead
    if _time_left() > 8:
        pts = _nelder_mead_optimize(best_points, time_limit=min(5, _time_left() - 5), maxiter=500)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
