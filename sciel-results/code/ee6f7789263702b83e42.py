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


def _covering_radius(points):
    """
    Compute the covering radius in degrees for a set of unit points.
    Uses convex hull facet circumradii (vectorized).
    """
    if len(points) < 3:
        return 180.0
    try:
        hull = ConvexHull(points)
    except:
        return 180.0
    
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
        return 180.0
    
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    # Ensure normals point outward
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    h = np.sum(n * p1m, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    
    return float(np.max(angles)) * 180.0 / np.pi


def _covering_radius_radians(points):
    """Covering radius in radians (for optimization)."""
    return _covering_radius(points) * np.pi / 180.0


def _facet_angles_vectorized(pts):
    """Compute all facet angles in radians, vectorized."""
    try:
        hull = ConvexHull(pts)
    except:
        return np.array([10.0])
    
    simplices = hull.simplices
    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norms = np.linalg.norm(n, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    n = n / norms
    
    # Ensure outward normal
    dots = np.sum(n * p1, axis=1)
    n = np.where(dots[:, None] < 0, -n, n)
    
    h = np.sum(n * p1, axis=1)
    h = np.clip(h, -1.0, 1.0)
    angles = np.arccos(h)
    return angles


def _smooth_surrogate(params, beta=80.0):
    """
    Smooth differentiable surrogate using log-sum-exp over hull facet heights.
    Minimizes a smooth upper bound on the covering radius (in radians).
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return 10.0
    
    angles = []
    for simplex in hull.simplices:
        p1 = pts[simplex[0]]
        p2 = pts[simplex[1]]
        p3 = pts[simplex[2]]
        
        v1 = p2 - p1
        v2 = p3 - p1
        n = np.cross(v1, v2)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            continue
        n = n / norm_n
        
        if np.dot(n, p1) < 0:
            n = -n
        
        h = np.dot(n, p1)
        h = np.clip(h, -1.0, 1.0)
        angle = np.arccos(h)
        angles.append(angle)
    
    if not angles:
        return 10.0
    
    angles_arr = np.array(angles)
    
    # log-sum-exp approximation of max(angles)
    max_val = np.max(angles_arr)
    if max_val > 1e-10:
        logsumexp = max_val + np.log(np.sum(np.exp(angles_arr - max_val)))
    else:
        logsumexp = max_val
    
    return logsumexp / beta


def _facet_smooth_surrogate(params, beta=50.0):
    """
    Smooth surrogate based on convex hull facet circumradii (vectorized).
    This provides a gradient for L-BFGS-B.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    try:
        hull = ConvexHull(pts)
    except:
        return np.pi
        
    simplices = hull.simplices
    if len(simplices) == 0:
        return np.pi

    p1 = pts[simplices[:, 0]]
    p2 = pts[simplices[:, 1]]
    p3 = pts[simplices[:, 2]]
    
    # Normal to the plane
    v1 = p2 - p1
    v2 = p3 - p1
    n = np.cross(v1, v2)
    norm_n = np.linalg.norm(n, axis=1, keepdims=True)
    
    # Handle degenerate triangles
    mask = norm_n[:, 0] > 1e-10
    if not np.all(mask):
        return np.pi
        
    n = n[mask] / norm_n[mask]
    p1m = p1[mask]
    
    # Ensure normals point towards the interior of the sphere relative to the triangle?
    # The circumcenter c is parallel to n.
    # We need c.p1 > 0.
    dots = np.sum(n * p1m, axis=1)
    n[dots < 0] *= -1
    
    # Calculate circumradius
    # cos(R) = c.p1
    cos_r = np.sum(n * p1m, axis=1)
    cos_r = np.clip(cos_r, -1.0, 1.0)
    radii = np.arccos(cos_r)
    
    # Soft-max (log-sum-exp)
    # We want to minimize the max radius, so we approximate max(radii) with logsumexp(beta * radii) / beta
    max_val = np.max(beta * radii)
    logsumexp = max_val + np.log(np.sum(np.exp(beta * radii - max_val)))
    return logsumexp / beta


def _epigraph_surrogate(augmented, beta=80.0, mu=5.0):
    """
    Epigraph formulation: minimize t + mu * logsumexp(beta*(angles - t)) / beta
    augmented = [pts (N*3), t]
    """
    params = augmented[:N*3]
    t = augmented[N*3]
    
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    angles = _facet_angles_vectorized(pts)
    
    if len(angles) == 0:
        return 10.0
    
    # Soft constraint: t + mu * (1/beta) * log(sum(exp(beta*(angles - t))))
    shifted = beta * (angles - t)
    max_shifted = np.max(shifted)
    if max_shifted > 500:
        logsumexp = max_shifted + np.log(len(shifted))
    elif max_shifted < -500:
        logsumexp = max_shifted
    else:
        logsumexp = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    
    return t + mu * logsumexp / beta


def _epigraph_optimize(points, time_limit=20, beta=80.0, mu=5.0):
    """Optimize using epigraph formulation with L-BFGS-B."""
    start_time = time.time()
    
    x0 = np.concatenate([points.flatten(), [0.3]])  # initial t guess
    
    def objective(augmented):
        if time.time() - start_time > time_limit:
            return 1e6
        return _epigraph_surrogate(augmented, beta=beta, mu=mu)
    
    result = minimize(objective, x0, method='L-BFGS-B',
                      options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000})
    best_points = result.x[:N*3].reshape(N, 3)
    return _normalize(best_points)


def _gram_smooth_surrogate(params, beta=50.0):
    """
    Gram-matrix-based smooth surrogate.
    Uses pairwise angular distances with a soft-max over test directions.
    """
    pts = params.reshape(N, 3)
    pts = _normalize(pts)
    
    # Compute Gram matrix of pairwise dot products
    G = pts @ pts.T
    G = np.clip(G, -1.0, 1.0)
    
    # Angular distances (radians)
    D = np.arccos(G)
    np.fill_diagonal(D, 0.0)
    
    # For each point, find max distance to nearest other point
    # Actually we want: for each point i, min_j D[i,j] (nearest neighbor distance)
    # Then the covering radius is related to max_i min_j D[i,j]
    
    # For each row, compute soft-min (log-sum-exp with negative)
    # soft_min_j D[i,j] = -logsumexp(-D[i,j]) / beta
    neg_D = -D
    neg_max = np.max(neg_D, axis=1, keepdims=True)
    neg_max = np.where(neg_max > 1e-10, neg_max, 0.0)
    logsumexp_neg = neg_max + np.log(np.sum(np.exp(neg_D - neg_max), axis=1, keepdims=True))
    soft_min_dist = -logsumexp_neg / beta
    
    # Now soft-max over all points
    sm_max = np.max(soft_min_dist)
    if sm_max > 1e-10:
        logsumexp_sm = sm_max + np.log(np.sum(np.exp(soft_min_dist - sm_max)))
    else:
        logsumexp_sm = sm_max
    
    return logsumexp_sm / beta


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


def _spherical_lloyd_iteration(points, max_iter=80, time_limit=15, damping=0.5):
    """
    Spherical Lloyd's algorithm (Voronoi relaxation).
    Moves each point toward the centroid of its adjacent facet normals.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    current_cost = _covering_radius(points)
    best_points = points.copy()
    best_cost = current_cost
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        
        try:
            hull = ConvexHull(points)
        except:
            break
        
        new_points = points.copy()
        
        point_to_facets = {i: [] for i in range(N)}
        for fi, simplex in enumerate(hull.simplices):
            for pi in simplex:
                point_to_facets[pi].append(fi)
        
        for pi in range(N):
            facets = point_to_facets[pi]
            if not facets:
                continue
            
            normal_sum = np.zeros(3)
            for fi in facets:
                simplex = hull.simplices[fi]
                p1 = points[simplex[0]]
                p2 = points[simplex[1]]
                p3 = points[simplex[2]]
                
                v1 = p2 - p1
                v2 = p3 - p1
                n = np.cross(v1, v2)
                norm_n = np.linalg.norm(n)
                if norm_n < 1e-10:
                    continue
                n = n / norm_n
                
                if np.dot(n, p1) < 0:
                    n = -n
                
                normal_sum += n
            
            if np.linalg.norm(normal_sum) > 1e-10:
                target = normal_sum / np.linalg.norm(normal_sum)
                new_points[pi] = points[pi] + damping * (target - points[pi])
                new_points[pi] = new_points[pi] / np.linalg.norm(new_points[pi])
        
        new_points = _normalize(new_points)
        new_cost = _covering_radius(new_points)
        
        if new_cost < current_cost:
            current_cost = new_cost
            points = new_points
            if current_cost < best_cost:
                best_cost = current_cost
                best_points = points.copy()
        else:
            break
    
    return best_points


def _smooth_gradient_optimize(points, time_limit=20, beta=80.0, use_gram=False, use_facet_smooth=False):
    """
    Gradient-based optimization using smooth log-sum-exp surrogate with L-BFGS-B.
    Can use hull-facet or Gram-matrix based surrogate.
    """
    start_time = time.time()
    
    if use_gram:
        def objective(params):
            if time.time() - start_time > time_limit:
                return 1e6
            return _gram_smooth_surrogate(params, beta=beta)
    elif use_facet_smooth:
        def objective(params):
            if time.time() - start_time > time_limit:
                return 1e6
            return _facet_smooth_surrogate(params, beta=beta)
    else:
        def objective(params):
            if time.time() - start_time > time_limit:
                return 1e6
            return _smooth_surrogate(params, beta=beta)
    
    x0 = points.flatten()
    result = minimize(objective, x0, method='L-BFGS-B', 
                      options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 3000})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _facet_targeted_refinement(points, time_limit=10, n_rounds=3):
    """
    Identify the worst facet(s) and move points to reduce the largest circumradius.
    Uses adaptive step size and considers multiple worst facets per round.
    """
    start_time = time.time()
    pts = _normalize(points).copy()
    best_pts = pts.copy()
    best_cost = _covering_radius(pts)
    
    for round_i in range(n_rounds):
        if time.time() - start_time > time_limit:
            break
        
        angles = _facet_angles_vectorized(pts)
        if len(angles) == 0:
            break
        
        # Find top 3 worst facets
        sorted_indices = np.argsort(angles)[::-1]
        worst_indices = sorted_indices[:min(3, len(sorted_indices))]
        
        try:
            hull = ConvexHull(pts)
        except:
            break
        
        step = 0.02
        for worst_idx in worst_indices:
            worst_facet = hull.simplices[worst_idx]
            i, j, k = worst_facet
            p1, p2, p3 = pts[i], pts[j], pts[k]
            
            # Compute facet normal
            v1 = p2 - p1
            v2 = p3 - p1
            n = np.cross(v1, v2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            n = n / norm_n
            if np.dot(n, p1) < 0:
                n = -n
            
            # Move each vertex toward the centroid of the other two to shrink the facet
            for idx in [i, j, k]:
                if idx == i:
                    target = (p2 + p3) / 2.0
                elif idx == j:
                    target = (p1 + p3) / 2.0
                else:
                    target = (p1 + p2) / 2.0
                
                target = target / np.linalg.norm(target)
                new_pt = pts[idx] + step * (target - pts[idx])
                new_pt = new_pt / np.linalg.norm(new_pt)
                pts[idx] = new_pt
        
        new_cost = _covering_radius(pts)
        if new_cost < best_cost:
            best_cost = new_cost
            best_pts = pts.copy()
        else:
            pts = _normalize(best_pts).copy()
            break
    
    return best_pts


def _nelder_mead_optimize(points, time_limit=15, maxiter=3000):
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
                      options={'maxiter': maxiter, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True})
    best_points = result.x.reshape(N, 3)
    return _normalize(best_points)


def _sa_optimize_covering(points, T_start=0.15, T_end=1e-5, n_iter=2000, time_limit=20):
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
        
        step_size = 0.02 * T / T_start
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


def _refine_point(points, idx, time_limit=3):
    """
    Refine a single point by optimizing its position given others fixed.
    Uses L-BFGS-B on a smooth surrogate for speed.
    """
    start_time = time.time()
    points = _normalize(points).copy()
    other_points = np.delete(points, idx, axis=0)
    
    def objective_single(params):
        if time.time() - start_time > time_limit:
            return 1e6
        pt = params / np.linalg.norm(params)
        all_pts = np.vstack([other_points, pt.reshape(1, 3)])
        # Use smooth surrogate for differentiability
        return _smooth_surrogate(all_pts.flatten(), beta=100.0)
    
    x0 = points[idx]
    result = minimize(objective_single, x0, method='L-BFGS-B',
                      options={'maxiter': 200, 'ftol': 1e-14, 'gtol': 1e-12, 'maxfun': 500})
    new_pt = result.x / np.linalg.norm(result.x)
    points[idx] = new_pt
    return points


def _high_precision_polish(points, time_limit=25):
    """
    High-precision polish: alternate between smooth gradient descent (increasing beta),
    epigraph optimization, Gram-matrix optimization, facet-targeted refinement,
    and tight Nelder-Mead on the true objective.
    """
    start_time = time.time()
    pts = _normalize(points).copy()
    current_cost = _covering_radius(pts)
    best_pts = pts.copy()
    best_cost = current_cost
    
    # Phase 1: Smooth gradient descent with increasing beta for precision
    for beta in [60, 100, 150, 200]:
        elapsed = time.time() - start_time
        if elapsed > time_limit - 15:
            break
        remaining = time_limit - elapsed
        if remaining < 5:
            break
        pts = _smooth_gradient_optimize(pts, time_limit=min(5, remaining), beta=beta)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_pts = pts.copy()
    
    # Phase 1.5: Epigraph optimization for a different perspective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 12:
        remaining = time_limit - elapsed
        pts = _epigraph_optimize(best_pts, time_limit=min(5, remaining), beta=80.0, mu=5.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_pts = pts.copy()
    
    # Phase 1.6: Gram-matrix based optimization for complementary signal
    elapsed = time.time() - start_time
    if elapsed < time_limit - 10:
        remaining = time_limit - elapsed
        pts = _smooth_gradient_optimize(best_pts, time_limit=min(4, remaining), beta=50.0, use_gram=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_pts = pts.copy()
    
    # Phase 1.7: Facet-targeted refinement
    elapsed = time.time() - start_time
    if elapsed < time_limit - 8:
        remaining = time_limit - elapsed
        pts = _facet_targeted_refinement(best_pts, time_limit=min(4, remaining), n_rounds=5)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_pts = pts.copy()
    
    # Phase 2: Nelder-Mead with tight tolerances on true objective
    elapsed = time.time() - start_time
    if elapsed < time_limit - 6:
        remaining = time_limit - elapsed
        pts = _nelder_mead_optimize(best_pts, time_limit=min(5, remaining), maxiter=5000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_pts = pts.copy()
    
    # Phase 2.5: Another facet-targeted pass after Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < time_limit - 4:
        remaining = time_limit - elapsed
        pts = _facet_targeted_refinement(best_pts, time_limit=min(3, remaining), n_rounds=3)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_pts = pts.copy()
    
    # Phase 3: Final ultra-tight Nelder-Mead
    elapsed = time.time() - start_time
    if elapsed < time_limit - 2:
        remaining = time_limit - elapsed
        pts = _nelder_mead_optimize(best_pts, time_limit=min(3, remaining), maxiter=8000)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_pts = pts.copy()
    
    return best_pts


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    total_time_limit = 145.0
    start_time = time.time()
    
    def _time_left():
        return total_time_limit - (time.time() - start_time)
    
    # Try multiple initializations and keep the best
    initializations = []
    
    # Fibonacci spiral
    initializations.append(_fibonacci_points())
    
    # Jittered Fibonacci
    initializations.append(_jittered_fibonacci_points(seed=123, jitter=0.05))
    
    # Random with different seeds
    for seed in [42, 123, 777, 999, 555]:
        initializations.append(_random_points(seed=seed))
    
    # Icosahedron-based
    initializations.append(_icosahedron_based_points())
    
    best_points = None
    best_cost = np.inf
    
    # Phase 1: Lloyd's relaxation from multiple starts
    for init_pts in initializations:
        if _time_left() < 50:
            break
        
        pts = _spherical_lloyd_iteration(init_pts, max_iter=60, time_limit=min(10, _time_left() - 45), damping=0.5)
        cost = _covering_radius(pts)
        
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    if best_points is None:
        best_points = _fibonacci_points()
        best_cost = _covering_radius(best_points)
    
    # Phase 2: Smooth gradient optimization (hull-facet based surrogate)
    if _time_left() > 35:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(12, _time_left() - 30), beta=80.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 2.5: Facet smooth surrogate optimization
    if _time_left() > 35:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(8, _time_left() - 30), beta=50.0, use_facet_smooth=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 2.6: Gram-matrix based optimization for complementary signal
    if _time_left() > 35:
        pts = _smooth_gradient_optimize(best_points, time_limit=min(8, _time_left() - 30), beta=50.0, use_gram=True)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 3: Simulated Annealing
    if _time_left() > 40:
        pts = _sa_optimize_covering(best_points, T_start=0.15, T_end=1e-5, n_iter=2000, time_limit=min(15, _time_left() - 35))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 4: Per-point refinement (coordinate descent) - faster with L-BFGS-B
    if _time_left() > 25:
        for idx in range(N):
            if _time_left() < 15:
                break
            pts = _refine_point(best_points, idx, time_limit=min(1.5, _time_left() - 12))
            cost = _covering_radius(pts)
            if cost < best_cost:
                best_cost = cost
                best_points = pts.copy()
    
    # Phase 5: Facet-targeted refinement before final polish
    if _time_left() > 20:
        pts = _facet_targeted_refinement(best_points, time_limit=min(8, _time_left() - 15), n_rounds=5)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 5.5: Epigraph optimization before final polish
    if _time_left() > 25:
        pts = _epigraph_optimize(best_points, time_limit=min(8, _time_left() - 20), beta=100.0, mu=5.0)
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Phase 6: High-precision polish
    if _time_left() > 25:
        pts = _high_precision_polish(best_points, time_limit=min(20, _time_left() - 12))
        cost = _covering_radius(pts)
        if cost < best_cost:
            best_cost = cost
            best_points = pts.copy()
    
    # Final normalization
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
