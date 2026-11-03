"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    
    return max_r


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, seed=42):
    """Generate random points on the unit sphere."""
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts


def _icosahedron_based_points(n):
    """Generate points based on icosahedron symmetry with perturbations."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    
    pts = base.copy()
    rng = np.random.RandomState(123)
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid /= mid_norm
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.01:
                pts = np.vstack([pts, mid])
    
    return pts[:n]


def _spherical_lloyd_iteration(points, max_iters=500, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        new_points = np.zeros_like(points)
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(len(points)):
            if not point_to_facets[i]:
                new_points[i] = points[i]
                continue
            
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    continue
                v = v / v_norm
                dot_a = np.dot(v, a)
                if dot_a < 0:
                    v = -v
                circumcenters.append(v)
            
            if not circumcenters:
                new_points[i] = points[i]
                continue
            
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / centroid_norm
        
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _lbfgs_smooth_minimize(points, time_limit=None, start_time=None):
    """
    Minimize a smooth surrogate of the covering radius using L-BFGS-B.
    
    The objective is a soft-max (log-sum-exp) approximation of the maximum
    angular distance from any facet circumcenter to its nearest point.
    
    We parameterize points in R^3 and project to the sphere after each step.
    """
    from scipy.optimize import minimize
    
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    def _facet_data(pts):
        """Compute facet circumcenters and radii for a configuration."""
        try:
            hull = ConvexHull(pts)
        except Exception:
            return None, None, None
        
        centers = []
        radii = []
        facets = []
        for simplex in hull.simplices:
            pts_f = pts[simplex]
            if len(pts_f) != 3:
                continue
            a, b, c = pts_f[0], pts_f[1], pts_f[2]
            v = np.cross(b - a, c - a)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            v = v / v_norm
            dot_a = np.dot(v, a)
            if dot_a < 0:
                v = -v
                dot_a = -dot_a
            cos_r = np.clip(dot_a, -1.0, 1.0)
            r = np.arccos(cos_r)
            centers.append(v)
            radii.append(r)
            facets.append(simplex)
        
        if not centers:
            return None, None, None
        
        return np.array(centers), np.array(radii), facets
    
    def _objective_and_grad(x_flat):
        """Objective and gradient for L-BFGS-B."""
        x = x_flat.reshape(N, 3)
        # Project to sphere
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms < 1e-15] = 1.0
        x = x / norms
        
        centers, radii, facets = _facet_data(x)
        if centers is None:
            return 100.0, np.zeros_like(x_flat)
        
        # Soft-max approximation of max radius
        # Use log-sum-exp with temperature beta
        beta = 50.0  # Higher = closer to true max
        scaled = radii * beta
        max_scaled = np.max(scaled)
        log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
        soft_max = log_sum_exp / beta
        
        # Gradient: d(soft_max)/d(radii) = softmax weights
        weights = np.exp(scaled - max_scaled) / np.sum(np.exp(scaled - max_scaled))
        
        # Gradient of soft_max w.r.t. points
        grad_flat = np.zeros_like(x_flat)
        for i, (center, r, weight) in enumerate(zip(centers, radii, weights)):
            simplex = facets[i]
            for idx in simplex:
                p = x[idx]
                # Gradient of angular distance from point p to center
                # d(arccos(cos_r))/dp where cos_r = dot(center, p)
                # = -dot(center - cos_r*p, p) / (sin_r * ||p||) ... simplified
                # On unit sphere, the gradient of arccos(dot(c, p)) w.r.t. p
                # projected onto tangent plane:
                cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
                sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
                if sin_r < 1e-10:
                    continue
                # Gradient of arccos(dot(c,p)) w.r.t. p is -c/sin_r
                # But we need the tangent component
                grad_p = -center / sin_r
                # Project to tangent plane
                grad_p_tangent = grad_p - np.dot(grad_p, p) * p
                grad_flat[idx * 3:(idx + 1) * 3] += weight * grad_p_tangent
        
        return soft_max, grad_flat.flatten()
    
    # Use L-BFGS-B
    x0 = points.flatten()
    
    # Time-limited optimization
    start_opt = time.time()
    deadline = start_time + time_limit if (start_time is not None and time_limit is not None) else None
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': 200, 'gtol': 1e-8, 'ftol': 1e-10, 'maxfun': 500}
        )
    except Exception:
        result = None
    
    if result is not None and result.success:
        new_points = result.x.reshape(N, 3)
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr:
            best_cr = new_cr
            best_points = new_points.copy()
    
    return best_points, best_cr


def _gradient_descent(points, max_iters=2000, lr_init=0.005, lr_min=1e-8, time_limit=None, start_time=None):
    """Gradient descent to minimize covering radius."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        facet_data = []
        for simplex in hull.simplices:
            pts = points[simplex]
            if len(pts) != 3:
                continue
            a, b, c = pts[0], pts[1], pts[2]
            v = np.cross(b - a, c - a)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            v = v / v_norm
            dot_a = np.dot(v, a)
            if dot_a < 0:
                v = -v
                dot_a = -dot_a
            cos_r = np.clip(dot_a, -1.0, 1.0)
            r = np.arccos(cos_r)
            facet_data.append((r, simplex, v))
        
        if not facet_data:
            break
        
        facet_data.sort(key=lambda x: -x[0])
        top_k = min(5, len(facet_data))
        
        step = np.zeros_like(points)
        for r, simplex, center in facet_data[:top_k]:
            weight = max(0.0, r - best_cr + 0.01)
            for idx in simplex:
                p = points[idx]
                d = center - p
                d_tangent = d - np.dot(d, p) * p
                step[idx] += weight * d_tangent
        
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            break
        step /= step_norm
        
        new_points = points + lr * step
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr - 1e-13:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
            lr = max(lr * 0.999, lr_min)
        else:
            lr = max(lr * 0.92, lr_min)
            if lr < lr_min * 1.01:
                break
    
    return best_points, best_cr


def _simulated_annealing(points, time_limit=None, start_time=None, seed=42):
    """Simulated annealing to minimize covering radius."""
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    rng = np.random.RandomState(seed)
    T = 0.05
    cooling = 0.995
    T_min = 1e-6
    
    current_cr = best_cr
    current_points = points.copy()
    
    while T > T_min:
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        n_pert = rng.randint(1, 4)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = current_points.copy()
        for idx in indices:
            noise = rng.randn(3) * T
            perturbed[idx] = current_points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        new_cr = _covering_radius(perturbed)
        delta = new_cr - current_cr
        
        if delta < 0 or rng.rand() < np.exp(-delta / T):
            current_points = perturbed
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = perturbed.copy()
        
        T *= cooling
    
    return best_points, best_cr


def _refine(points, time_limit=None, start_time=None):
    """Multi-stage refinement: Lloyd + L-BFGS smooth + gradient descent."""
    # Stage 1: Lloyd's algorithm
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, _covering_radius(points)
        lloyd_time = min(remaining * 0.25, 8)
    else:
        lloyd_time = 8
    
    points, cr1 = _spherical_lloyd_iteration(points, max_iters=300, time_limit=lloyd_time, start_time=start_time)
    
    # Stage 2: L-BFGS smooth minimization
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 3:
            return points, cr1
        lbfgs_time = min(remaining * 0.35, 15)
    else:
        lbfgs_time = 15
    
    points, cr2 = _lbfgs_smooth_minimize(points, time_limit=lbfgs_time, start_time=start_time)
    
    # Stage 3: Gradient descent polish
    if time_limit is not None and start_time is not None:
        remaining = time_limit - (time.time() - start_time)
        if remaining < 2:
            return points, cr2
        gd_time = min(remaining * 0.3, 10)
    else:
        gd_time = 10
    
    points, cr3 = _gradient_descent(points, max_iters=500, lr_init=0.003, time_limit=gd_time, start_time=start_time)
    
    return points, cr3


def construct():
    start_time = time.time()
    deadline = start_time + 145.0
    
    # Generate multiple diverse initial configurations
    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _icosahedron_based_points(N),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    # Phase 1: Refine each initial configuration
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 10:
            break
        
        time_per_config = min(remaining / (len(initial_configs) - i), 25)
        refined, cr = _refine(init_pts, time_limit=time_per_config, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = refined.copy()
    
    # Phase 2: Simulated annealing from best configuration
    remaining = deadline - time.time()
    if remaining > 15:
        sa_time = min(remaining * 0.3, 25)
        sa_points, sa_cr = _simulated_annealing(best_points, time_limit=sa_time, start_time=start_time, seed=123)
        if sa_cr < best_cr:
            best_cr = sa_cr
            best_points = sa_points.copy()
    
    # Phase 3: Final refinement with L-BFGS
    remaining = deadline - time.time()
    if remaining > 10:
        final_time = min(remaining * 0.7, 20)
        final_points, final_cr = _refine(best_points, time_limit=final_time, start_time=start_time)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points.copy()
    
    # Normalize final result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
