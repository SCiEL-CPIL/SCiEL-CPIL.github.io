"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius_exact(pts):
    """Compute the true covering radius in degrees using convex hull facets."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
        max_cr_rad = 0.0
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            # Compute circumcenter of spherical triangle a,b,c
            # The circumcenter is perpendicular to the plane of the triangle.
            # v = (a-b) x (b-c) is parallel to the circumcenter direction.
            v = np.cross(a - b, b - c)
            norm_v = np.linalg.norm(v)
            if norm_v < 1e-10:
                # Degenerate case: points are collinear or coincident on the sphere
                # This implies the "triangle" is actually a line segment or point.
                # The covering radius for a segment is half the arc length.
                d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
                d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
                d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
                # If they form a line, the max distance to the "midpoint" is max/2? 
                # Actually, if degenerate, it's likely a vertex of the hull or numerical error.
                # Safe fallback: max distance to any vertex is the radius if it's a point.
                # But for a hull facet, it should be a triangle. If degenerate, treat as max angle / 2.
                cr = max(d_ab, d_bc, d_ca) / 2.0
            else:
                u = v / norm_v
                
                # Ensure u points in the direction of the triangle (away from origin)
                # The normal to the plane containing a,b,c is n = (b-a) x (c-a).
                # The circumcenter lies on the line passing through origin in direction u.
                # We need to pick the sign such that the center is "inside" the spherical triangle.
                # A robust check: the dot product of the circumcenter with the vertices
                # should be positive (for acute triangles) or we just check if it's closer
                # to the centroid direction.
                
                # Better check: The circumcenter C satisfies |C-a|=|C-b|=|C-c|.
                # C = ±u.
                # Let's compute distance to a for both signs.
                d_pos = np.arccos(np.clip(np.dot(u, a), -1, 1))
                d_neg = np.arccos(np.clip(np.dot(-u, a), -1, 1))
                
                # The circumcenter must be such that the angular distance to all vertices is the same.
                # Due to symmetry, if u is the correct direction, d_pos should be the circumradius.
                # If -u is correct, d_neg will be the circumradius (and d_pos will be pi - circumradius).
                # Usually, the circumcenter of a convex hull facet on the unit sphere is the one
                # that is "further out" from the origin relative to the chord, which corresponds
                # to the smaller angle if the triangle is acute, or larger if obtuse?
                
                # Actually, the circumcenter of the spherical triangle is the point on the sphere
                # equidistant from a,b,c.
                # Let's just check which sign gives a consistent distance to b and c.
                # dist(u, a) == dist(u, b) == dist(u, c) ?
                
                dist_a_pos = np.dot(u, a)
                dist_b_pos = np.dot(u, b)
                dist_c_pos = np.dot(u, c)
                
                dist_a_neg = -dist_a_pos
                dist_b_neg = -dist_b_pos
                dist_c_neg = -dist_c_pos
                
                # Variance of dot products for +u
                var_pos = np.var([dist_a_pos, dist_b_pos, dist_c_pos])
                # Variance of dot products for -u
                var_neg = np.var([dist_a_neg, dist_b_neg, dist_c_neg])
                
                # The correct circumcenter will have the 3 dot products very close to each other.
                if var_pos < var_neg:
                    circum_center = u
                    cr = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
                else:
                    circum_center = -u
                    cr = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
            
            max_cr_rad = max(max_cr_rad, cr)
        
        return np.degrees(max_cr_rad)
    except Exception:
        # Fallback to grid sampling if hull fails
        return _covering_radius_grid(pts)

def _covering_radius_grid(pts, n_grid=200):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)
    min_angles = np.arccos(min_dots)
    return np.degrees(np.max(min_angles))

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return _normalize(np.column_stack([r * np.cos(phi), r * np.sin(phi), z]))

def _random_points(n, seed):
    rng = np.random.RandomState(seed)
    pts = rng.randn(n, 3)
    return _normalize(pts)

def _icosahedron_based_points(n, seed):
    """Start from icosahedron vertices and subdivide."""
    rng = np.random.RandomState(seed)
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
        [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
        [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1]
    ])
    ico = _normalize(ico)
    all_pts = np.vstack([ico, _random_points(n - len(ico), seed)])
    selected = [0]
    for _ in range(n - 1):
        min_dists = np.min([np.arccos(np.clip(np.dot(all_pts, all_pts[i]), -1, 1)) for i in selected], axis=0)
        selected.append(np.argmax(min_dists))
    return _normalize(all_pts[selected])

def _smooth_objective(x, samples, beta):
    """
    Smooth surrogate objective: Log-Sum-Exp of squared distances from samples to nearest point.
    x: flattened array of N*3 points.
    """
    pts = x.reshape(-1, 3)
    pts = _normalize(pts)
    
    # Compute distances from each sample to each point
    # samples: (M, 3), pts: (N, 3)
    dots = samples @ pts.T
    dots = np.clip(dots, -1, 1)
    
    # Distance is arccos(dot). We want to minimize the max distance.
    # Using LSE on squared distances makes it smooth and emphasizes the largest ones.
    dists_sq = (1.0 - dots) * 2.0 # Approximation of squared chord distance? 
    # Actually, angular distance theta = arccos(dot). 
    # Minimizing theta is equivalent to maximizing dot.
    # So we can just minimize -LSE(dot) or maximize LSE(dot).
    # Let's minimize the LSE of the "error" where error = 1 - dot (proxy for distance).
    # Better: Use the actual angle.
    angles = np.arccos(dots)
    min_angles = np.min(angles, axis=1) # Distance from sample to nearest point
    
    # LSE approximation of max
    # beta controls sharpness. Higher beta -> closer to max.
    scaled = beta * min_angles
    # Shift to avoid overflow
    max_scaled = np.max(scaled)
    lse = max_scaled / beta + np.log(np.sum(np.exp(scaled - max_scaled))) / beta
    
    return lse

def _gradient_smooth(x, samples, beta):
    """
    Gradient of the smooth objective.
    """
    pts = x.reshape(-1, 3)
    pts = _normalize(pts)
    n = len(pts)
    
    dots = samples @ pts.T
    dots = np.clip(dots, -1, 1)
    angles = np.arccos(dots)
    
    # min_angles for each sample
    min_idx = np.argmin(angles, axis=1)
    min_angles = np.min(angles, axis=1)
    
    # LSE gradient
    scaled = beta * min_angles
    max_scaled = np.max(scaled)
    weights = np.exp(scaled - max_scaled)
    weights /= np.sum(weights)
    
    # d(LSE)/d(min_angles) = weights
    # d(min_angles)/d(pts) ?
    # min_angles[i] = angles[i, min_idx[i]]
    # d(angles)/d(dot) = -1 / sqrt(1 - dot^2)
    # d(dot)/d(pts) = samples
    
    grad_pts = np.zeros_like(pts)
    
    # For each sample i, it contributes to the point j = min_idx[i]
    # d(obj)/d(pts[j]) += weights[i] * d(angles[i,j])/d(pts[j])
    
    # Vectorized approach:
    # Create a matrix G of shape (M, N) where G[i, j] is non-zero only if j == min_idx[i]
    # G[i, min_idx[i]] = weights[i] * (-1 / sqrt(1 - dots[i, min_idx[i]]^2))
    
    inv_sqrt = np.zeros_like(dots)
    valid = dots < 1.0 - 1e-12
    inv_sqrt[valid] = -1.0 / np.sqrt(1.0 - dots[valid]**2)
    
    # Only keep the contribution from the nearest point
    mask = np.zeros_like(dots, dtype=bool)
    mask[np.arange(len(samples)), min_idx] = True
    contrib = weights[:, np.newaxis] * inv_sqrt
    contrib[~mask] = 0.0
    
    # grad_pts = contrib.T @ samples
    # contrib shape (M, N), samples shape (M, 3)
    grad_pts = contrib.T @ samples
    
    # Project gradient onto tangent space
    for i in range(n):
        grad_pts[i] -= np.dot(grad_pts[i], pts[i]) * pts[i]
        
    return grad_pts.flatten()

def _local_refine_smooth(pts, time_limit):
    """Refine points using L-BFGS-B on the smooth objective."""
    pts = _normalize(pts).copy()
    best = pts.copy()
    
    # Generate samples for the smooth objective
    # Use a fixed set of samples for consistency during optimization
    rng = np.random.RandomState(42)
    n_samples = 2000
    samples = rng.randn(n_samples, 3)
    samples = _normalize(samples)
    
    beta = 20.0 # Sharpness of LSE
    
    start_time = time.time()
    
    def obj_wrapper(x):
        return _smooth_objective(x, samples, beta)
        
    def grad_wrapper(x):
        return _gradient_smooth(x, samples, beta)

    x0 = best.flatten()
    
    try:
        res = minimize(obj_wrapper, x0, jac=grad_wrapper, method='L-BFGS-B',
                       options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
        if res.success or res.fun < obj_wrapper(x0):
            best = _normalize(res.x.reshape(-1, 3))
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time < time_limit:
        # Try a second pass with higher beta for sharper max approximation
        beta = 50.0
        x0 = best.flatten()
        try:
            res = minimize(obj_wrapper, x0, jac=grad_wrapper, method='L-BFGS-B',
                           options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8})
            if res.success or res.fun < obj_wrapper(x0):
                best = _normalize(res.x.reshape(-1, 3))
        except Exception:
            pass

    return best

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin for 150s limit
    
    # Multiple diverse initializations
    initializations = []
    initializations.append(('fib', _fibonacci_points(N)))
    for seed in range(5):
        initializations.append((f'rand_{seed}', _random_points(N, seed)))
    for seed in range(3):
        initializations.append((f'ico_{seed}', _icosahedron_based_points(N, seed)))
    
    best_pts = None
    best_cr = np.inf
    
    # Evaluate each initialization with exact metric to pick best start
    for name, pts in initializations:
        if time.time() > deadline - 30:
            break
        cr = _covering_radius_exact(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Refine the best initialization using smooth optimizer
    if best_pts is not None and time.time() < deadline - 10:
        refined_pts = _local_refine_smooth(best_pts, deadline - time.time() - 5)
        refined_cr = _covering_radius_exact(refined_pts)
        if refined_cr < best_cr:
            best_cr = refined_cr
            best_pts = refined_pts
    
    # If we still have time, try refining other good initializations
    if time.time() < deadline - 20:
        # Sort initializations by their initial CR
        scored_inits = []
        for name, pts in initializations:
            if time.time() > deadline - 15:
                break
            cr = _covering_radius_exact(pts)
            scored_inits.append((cr, pts))
            
        scored_inits.sort(key=lambda x: x[0])
        
        for cr, pts in scored_inits:
            if time.time() > deadline - 15:
                break
            if cr > best_cr + 2.0: # Skip if significantly worse
                continue
            if time.time() < deadline - 10:
                refined_pts = _local_refine_smooth(pts, deadline - time.time() - 5)
                refined_cr = _covering_radius_exact(refined_pts)
                if refined_cr < best_cr:
                    best_cr = refined_cr
                    best_pts = refined_pts

    if best_pts is None:
        best_pts = _fibonacci_points(N)
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
