import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
from scipy.spatial import cKDTree

N = 50

def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    points = _normalize(points)
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 5000
        samples = rng.randn(n_samples, 3)
        samples = _normalize(samples)
        dists, _ = tree.query(samples)
        return float(np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1))))
    
    max_r = 0.0
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
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
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        if r > max_r:
            max_r = r
    
    return float(max_r)

def _fast_facets(points):
    """Return list of (radius, circumcenter, facet_indices) for triangular facets."""
    points = np.asarray(points, dtype=float)
    points = _normalize(points)
    try:
        hull = ConvexHull(points)
    except Exception:
        return []
    
    out = []
    for simplex in hull.simplices:
        if len(simplex) != 3:
            continue
        pts = points[simplex]
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        vn = np.linalg.norm(v)
        if vn < 1e-12:
            continue
        v /= vn
        d = np.dot(v, a)
        if d < 0:
            v = -v
            d = -d
        cos_r = np.clip(d, -1.0, 1.0)
        r = np.arccos(cos_r)
        out.append((r, v, simplex))
    return out

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
    pts = _normalize(pts)
    return pts

def _icosahedron_based_points(n):
    """Generate points based on icosahedron symmetry with perturbations."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base = _normalize(base)
    
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

def _perturbed_icosahedron(n, seed=42, sigma=0.05):
    """Generate perturbed icosahedron-based points."""
    t = (1.0 + np.sqrt(5.0)) / 2.0
    base = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ], dtype=float)
    base = _normalize(base)
    
    rng = np.random.RandomState(seed)
    pts = base.copy()
    
    # Add points by subdividing and perturbing
    while len(pts) < n:
        i, j = rng.randint(0, len(pts), 2)
        if i == j:
            continue
        mid = (pts[i] + pts[j]) / 2.0
        mid_norm = np.linalg.norm(mid)
        if mid_norm > 1e-10:
            mid /= mid_norm
            # Add small random perturbation
            perturbation = rng.randn(3) * sigma
            mid = mid + perturbation
            mid = _normalize(mid)
            dists = 1.0 - np.sum(mid * pts, axis=1)
            if np.min(dists) > 0.02:
                pts = np.vstack([pts, mid])
    
    return pts[:n]

def _spherical_lloyd_iteration(points, max_iters=200, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm - optimized version."""
    points = _normalize(np.array(points, dtype=float))
    
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

def _smooth_covering_radius_grad(points, beta=50.0):
    """
    Compute the smooth covering radius (log-sum-exp over facet radii) and its
    gradient with respect to the input points.
    """
    points = np.asarray(points, dtype=float)
    points = _normalize(points)
    
    facets = _fast_facets(points)
    if not facets:
        return 0.0, np.zeros_like(points)
    
    radii = np.array([f[0] for f in facets])
    centers = np.array([f[1] for f in facets])
    simplices = [f[2] for f in facets]
    
    m = np.max(radii)
    if m < 1e-12:
        return 0.0, np.zeros_like(points)
    
    val = m + np.log(np.sum(np.exp(beta * (radii - m)))) / beta
    
    w = np.exp(beta * (radii - m))
    w /= w.sum()
    
    grad = np.zeros_like(points)
    for i in range(len(radii)):
        center = centers[i]
        simplex = simplices[i]
        for idx in simplex:
            p = points[idx]
            cos_r = np.clip(np.dot(center, p), -1.0, 1.0)
            sin_r = np.sqrt(np.maximum(0.0, 1.0 - cos_r * cos_r))
            if sin_r < 1e-10:
                continue
            g = -center / sin_r
            g_tan = g - np.dot(g, p) * p
            grad[idx] += w[i] * g_tan
    
    return val, grad

def _lbfgs_refine(points, beta, time_limit=None, start_time=None, maxiter=200, gtol=1e-8):
    """
    Refine points using L-BFGS-B on a log-sum-exp surrogate of the covering radius.
    """
    points = _normalize(np.array(points, dtype=float))
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    def _objective_and_grad(x_flat):
        x = x_flat.reshape(N, 3)
        x = _normalize(x)
        
        val, grad = _smooth_covering_radius_grad(x, beta=beta)
        # Project gradient to tangent space
        grad_tan = grad - np.sum(grad * x, axis=1, keepdims=True) * x
        
        return val, grad_tan.flatten()
    
    x0 = points.flatten()
    
    try:
        result = minimize(
            _objective_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            options={'maxiter': maxiter, 'gtol': gtol, 'ftol': 1e-12, 'maxfun': maxiter * 3}
        )
    except Exception:
        result = None
    
    if result is not None:
        new_points = result.x.reshape(N, 3)
        new_points = _normalize(new_points)
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr:
            best_cr = new_cr
            best_points = new_points.copy()
    
    return best_points, best_cr

def _spherical_gd(points, max_iters=1000, lr_init=0.005, lr_min=1e-10,
                  time_limit=None, start_time=None, beta=30.0):
    """Spherical gradient descent with adaptive step size."""
    points = _normalize(np.array(points, dtype=float))
    
    best = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    no_improve = 0
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        val, grad = _smooth_covering_radius_grad(points, beta=beta)
        gn = np.linalg.norm(grad)
        if gn < 1e-14:
            break
        
        # Normalize gradient for stable step size
        grad_normed = grad / gn
        
        accepted = False
        for trial in range(8):
            cand = points - lr * grad_normed  # Descent: subtract
            cand = _normalize(cand)
            cand_cr = _covering_radius(cand)
            
            if cand_cr < best_cr - 1e-14:
                points = cand
                best = cand.copy()
                best_cr = cand_cr
                lr = min(lr * 1.05, 0.05)
                no_improve = 0
                accepted = True
                break
            
            lr = lr * 0.7
        
        if not accepted:
            no_improve += 1
            lr = max(lr * 0.85, lr_min)
            if no_improve > 50 or lr < lr_min * 1.01:
                break
    
    return best, best_cr

def _pairwise_repulsion_refine(points, time_limit=None, start_time=None, max_iters=500):
    """
    Refine points by maximizing minimum pairwise distance using gradient ascent on Gram matrix.
    This helps ensure points are well-separated, a necessary condition for good covering.
    """
    points = _normalize(np.array(points, dtype=float))
    best = points.copy()
    best_min_dist = np.min(2.0 * np.arcsin(np.clip(np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=-1) / 2.0, 0, 1)))
    
    # Only consider off-diagonal
    mask = ~np.eye(len(points), dtype=bool)
    dists = 2.0 * np.arcsin(np.clip(np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=-1) / 2.0, 0, 1))
    dists[~mask] = np.pi  # Ignore diagonal
    best_min_dist = np.min(dists)
    
    lr = 0.01
    no_improve = 0
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Compute pairwise distances
        dists_mat = 2.0 * np.arcsin(np.clip(np.linalg.norm(points[:, np.newaxis] - points[np.newaxis, :], axis=-1) / 2.0, 0, 1))
        dists_mat[~mask] = np.pi
        min_dist = np.min(dists_mat)
        
        if min_dist > best_min_dist:
            best_min_dist = min_dist
            best = points.copy()
            no_improve = 0
            lr = min(lr * 1.02, 0.05)
        else:
            no_improve += 1
            if no_improve > 30:
                break
            lr = max(lr * 0.9, 1e-6)
        
        # Gradient ascent on minimum distance (smooth approximation)
        # Use log-sum-exp on negative distances to approximate -min(dist)
        beta = 20.0
        neg_dists = -dists_mat[mask]
        m = np.min(neg_dists)
        # log-sum-exp of neg_dists approximates log(sum(exp(neg_dists)))
        # We want to maximize min dist, so minimize -min dist
        # Gradient of -min(dist) wrt points pushes points apart
        
        # Simpler: push apart the pair with minimum distance
        min_idx = np.argmin(dists_mat)
        i, j = np.unravel_index(min_idx, dists_mat.shape)
        
        # Gradient direction: move i away from j and j away from i
        # On sphere, tangent direction
        p_i = points[i]
        p_j = points[j]
        
        # Direction from j to i
        d = p_i - np.dot(p_i, p_j) * p_j
        dn = np.linalg.norm(d)
        if dn < 1e-10:
            continue
        d /= dn
        
        # Move i in direction d, j in direction -d
        new_points = points.copy()
        new_points[i] = p_i + lr * d
        new_points[j] = p_j - lr * d
        new_points = _normalize(new_points)
        
        # Check if this improves the covering radius (not just pairwise)
        # Actually, we want to check if it improves the minimum pairwise distance
        new_dists_mat = 2.0 * np.arcsin(np.clip(np.linalg.norm(new_points[:, np.newaxis] - new_points[np.newaxis, :], axis=-1) / 2.0, 0, 1))
        new_dists_mat[~mask] = np.pi
        new_min_dist = np.min(new_dists_mat)
        
        if new_min_dist > min_dist:
            points = new_points
        else:
            # Try smaller step
            new_points = points.copy()
            new_points[i] = p_i + lr * 0.5 * d
            new_points[j] = p_j - lr * 0.5 * d
            new_points = _normalize(new_points)
            new_dists_mat = 2.0 * np.arcsin(np.clip(np.linalg.norm(new_points[:, np.newaxis] - new_points[np.newaxis, :], axis=-1) / 2.0, 0, 1))
            new_dists_mat[~mask] = np.pi
            new_min_dist = np.min(new_dists_mat)
            if new_min_dist > min_dist:
                points = new_points
    
    return best, best_min_dist

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin below 150s
    
    # More diverse initial configurations
    initial_configs = [
        _fibonacci_points(N),
        _random_points(N, seed=1),
        _random_points(N, seed=2),
        _random_points(N, seed=3),
        _icosahedron_based_points(N),
        _perturbed_icosahedron(N, seed=42, sigma=0.05),
        _perturbed_icosahedron(N, seed=123, sigma=0.1),
    ]
    
    best_points = initial_configs[0].copy()
    best_cr = _covering_radius(best_points)
    
    for i, init_pts in enumerate(initial_configs):
        remaining = deadline - time.time()
        if remaining < 20:
            break
        
        # Stage 1: Lloyd - give it more time for convergence
        lloyd_time = min(remaining * 0.20, 10)
        points, cr = _spherical_lloyd_iteration(init_pts, max_iters=150, time_limit=lloyd_time, start_time=start_time)
        
        # Stage 2: L-BFGS with low beta (smooth)
        remaining = deadline - time.time()
        if remaining < 10:
            continue
        lbfgs_time1 = min(remaining * 0.20, 10)
        points, cr = _lbfgs_refine(points, beta=15.0, time_limit=lbfgs_time1, start_time=start_time, maxiter=150, gtol=1e-7)
        
        # Stage 3: Spherical gradient descent for fine-tuning
        remaining = deadline - time.time()
        if remaining < 8:
            continue
        gd_time = min(remaining * 0.15, 8)
        points, cr = _spherical_gd(points, max_iters=500, time_limit=gd_time, start_time=start_time, beta=30.0)
        
        # Stage 4: L-BFGS with medium beta
        remaining = deadline - time.time()
        if remaining < 8:
            continue
        lbfgs_time2 = min(remaining * 0.20, 10)
        points, cr = _lbfgs_refine(points, beta=50.0, time_limit=lbfgs_time2, start_time=start_time, maxiter=150, gtol=1e-8)
        
        # Stage 5: L-BFGS with high beta (sharp)
        remaining = deadline - time.time()
        if remaining < 8:
            continue
        lbfgs_time3 = min(remaining * 0.20, 10)
        points, cr = _lbfgs_refine(points, beta=150.0, time_limit=lbfgs_time3, start_time=start_time, maxiter=150, gtol=1e-8)
        
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Final high-precision polish cascade
    remaining = deadline - time.time()
    if remaining > 15:
        # Very high beta polish
        final_time = min(remaining * 0.5, 20)
        final_points, final_cr = _lbfgs_refine(best_points, beta=300.0, time_limit=final_time, start_time=start_time, maxiter=200, gtol=1e-9)
        if final_cr < best_cr:
            best_cr = final_cr
            best_points = final_points
    
    remaining = deadline - time.time()
    if remaining > 10:
        # Ultra-high beta final polish
        final_time2 = min(remaining * 0.7, 15)
        final_points2, final_cr2 = _lbfgs_refine(best_points, beta=500.0, time_limit=final_time2, start_time=start_time, maxiter=200, gtol=1e-10)
        if final_cr2 < best_cr:
            best_cr = final_cr2
            best_points = final_points2
    
    # Final gradient descent polish
    remaining = deadline - time.time()
    if remaining > 5:
        gd_final_time = min(remaining * 0.5, 8)
        final_points3, final_cr3 = _spherical_gd(best_points, max_iters=300, time_limit=gd_final_time, start_time=start_time, beta=50.0)
        if final_cr3 < best_cr:
            best_cr = final_cr3
            best_points = final_points3
    
    best_points = _normalize(best_points)
    return best_points
