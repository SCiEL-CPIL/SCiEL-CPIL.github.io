"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def _covering_radius(pts):
    """Compute the true covering radius (in radians) via convex hull facets."""
    from scipy.spatial import ConvexHull, Delaunay
    # Project to 2D via stereographic or use 3D hull
    # Better: use the fact that covering radius = max over facets of circumradius
    # For points on sphere, the Voronoi vertices are where 3+ sites are equidistant
    # Use Delaunay triangulation on the sphere via 3D convex hull
    try:
        hull = ConvexHull(pts)
        # For each facet (triangle), compute the circumcenter projected to sphere
        # The covering radius is the max angular distance from facet center to its 3 vertices
        max_r = 0.0
        for simplex in hull.simplices:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Circumcenter of triangle abc in 3D (on the plane through a,b,c)
            # Then project to unit sphere
            ab = b - a
            ac = c - a
            cross = np.cross(ab, ac)
            # If degenerate, skip
            cross_norm = np.linalg.norm(cross)
            if cross_norm < 1e-14:
                continue
            # Circumcenter formula
            # d = 2 * (a . cross)
            d = 2.0 * np.dot(a, cross)
            if abs(d) < 1e-14:
                continue
            circum = (np.dot(b, b) * np.cross(ac, c) + 
                      np.dot(c, c) * np.cross(a, ab) + 
                      np.dot(a, a) * np.cross(ab, ac)) / d
            # Project to unit sphere
            circum_norm = np.linalg.norm(circum)
            if circum_norm < 1e-12:
                continue
            circum_unit = circum / circum_norm
            # Angular distance from circumcenter to each vertex
            for v in [a, b, c]:
                cos_angle = np.clip(np.dot(circum_unit, v), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                if angle > max_r:
                    max_r = angle
        return max_r
    except Exception:
        return np.pi  # fallback

def _smooth_covering_radius(pts, beta=50.0):
    """Smooth approximation of covering radius using log-sum-exp over a grid."""
    # Create a fine grid on the sphere
    n_theta = 60
    n_phi = 120
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi, indexing='ij')
    # Convert to Cartesian
    x = np.sin(TH) * np.cos(PH)
    y = np.sin(TH) * np.sin(PH)
    z = np.cos(TH)
    grid = np.stack([x.ravel(), y.ravel(), z.ravel()], axis=1)
    
    # Compute distances from each grid point to all N sites
    # dist[i,j] = angular distance from grid point i to site j
    # Use dot product: cos(d) = grid[i] . pts[j]
    dots = grid @ pts.T  # (n_grid, N)
    dots = np.clip(dots, -1.0, 1.0)
    dists = np.arccos(dots)  # (n_grid, N)
    
    # For each grid point, find the minimum distance to any site
    min_dists = np.min(dists, axis=1)
    
    # Log-sum-exp for smooth max
    max_d = np.max(min_dists)
    if max_d < 1e-10:
        return 0.0, np.zeros(N)
    
    # Smooth max of min_dists
    s = np.exp(beta * (min_dists - max_d))
    log_sum_exp = max_d + np.log(np.sum(s)) / beta
    
    # Gradient: for each site j, gradient contribution
    # d(log_sum_exp)/d(pts[j]) via chain rule
    # This is complex; instead use subgradient approach
    # For the smooth objective, the gradient w.r.t. pts[j] is:
    # sum_i w_i * (-d_i / sin(d_i)) * (grid_i - pts[j] + (grid_i . pts[j]) * pts[j]) / |...|
    # where w_i = exp(beta*(min_dists_i - max_d)) / sum(exp(...))
    
    # Actually, let's use a simpler approach: the covering radius is determined by
    # the Voronoi vertices. Let's just use gradient descent with a penalty.
    return log_sum_exp, None

def _project_to_sphere(pts, grad):
    """Project gradient to tangent space of sphere."""
    # Tangent space: remove component along radial direction
    radial = np.sum(pts * grad, axis=1, keepdims=True)
    return grad - radial * pts

def construct():
    t0 = time.time()
    deadline = t0 + 130.0  # safety margin
    
    # Initial configuration: Fibonacci sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    # Compute initial covering radius
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Gradient descent with momentum on the sphere
    # Use a smooth surrogate: for each grid point, the distance to nearest site
    # The objective is max over grid points of min over sites of angular distance
    
    # Precompute grid
    n_theta = 50
    n_phi = 100
    theta = np.linspace(0, np.pi, n_theta)
    phi_grid = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    TH, PH = np.meshgrid(theta, phi_grid, indexing='ij')
    gx = np.sin(TH) * np.cos(PH)
    gy = np.sin(TH) * np.sin(PH)
    gz = np.cos(TH)
    grid = np.stack([gx.ravel(), gy.ravel(), gz.ravel()], axis=1)
    n_grid = grid.shape[0]
    
    # Learning rate schedule
    lr = 0.001
    momentum = 0.9
    vel = np.zeros_like(pts)
    
    # Iterative optimization
    max_iter = 500
    for it in range(max_iter):
        if time.time() > deadline - 10.0:
            break
        
        # Compute distances from grid to sites
        dots = grid @ pts.T  # (n_grid, N)
        dots = np.clip(dots, -1.0, 1.0)
        dists = np.arccos(dots)  # (n_grid, N)
        
        # Min distance for each grid point
        min_idx = np.argmin(dists, axis=1)
        min_dists = dists[np.arange(n_grid), min_idx]
        
        # Gradient of max(min_dists) w.r.t. pts
        # Use soft-max weighting
        beta = 30.0
        max_d = np.max(min_dists)
        if max_d < 1e-10:
            break
        
        weights = np.exp(beta * (min_dists - max_d))
        weights /= np.sum(weights)
        
        # Gradient: for each site j, sum over grid points where j is nearest
        grad = np.zeros_like(pts)
        for i in range(n_grid):
            w = weights[i]
            if w < 1e-10:
                continue
            j = min_idx[i]
            g = grid[i]
            p = pts[j]
            d = min_dists[i]
            # d(arccos(g.p))/dp = -g / sin(arccos(g.p)) + (g.p) * p / sin(arccos(g.p))
            # = -(g - (g.p)*p) / sin(d)
            sin_d = np.sin(d)
            if sin_d < 1e-10:
                continue
            # The gradient of distance w.r.t. p is: -(g - (g.p)*p)/sin(d)
            # But we want to MINIMIZE the max distance, so we move p toward g
            # grad_d/dp = -(g - (g.p)*p)/sin(d)
            # To minimize, we move in direction -grad = (g - (g.p)*p)/sin(d)
            tangent = g - np.dot(g, p) * p
            grad[j] += w * tangent / sin_d
        
        # Project to tangent space
        grad = _project_to_sphere(pts, grad)
        
        # Momentum
        vel = momentum * vel + lr * grad
        new_pts = pts + vel
        
        # Normalize to sphere
        new_pts = _normalize(new_pts)
        
        # Check if improved
        # Use smooth objective as proxy
        dots_new = grid @ new_pts.T
        dots_new = np.clip(dots_new, -1.0, 1.0)
        dists_new = np.arccos(dots_new)
        min_dists_new = np.min(dists_new, axis=1)
        new_smooth = np.max(min_dists_new)
        
        dots_old = grid @ pts.T
        dots_old = np.clip(dots_old, -1.0, 1.0)
        dists_old = np.arccos(dots_old)
        min_dists_old = np.min(dists_old, axis=1)
        old_smooth = np.max(min_dists_old)
        
        if new_smooth < old_smooth:
            pts = new_pts
        else:
            # Reduce learning rate
            lr *= 0.9
            vel *= 0.5
            # Try with smaller step
            vel = momentum * vel + lr * grad
            new_pts = _normalize(pts + vel)
            dots_new2 = grid @ new_pts.T
            dots_new2 = np.clip(dots_new2, -1.0, 1.0)
            dists_new2 = np.arccos(dots_new2)
            min_dists_new2 = np.min(dists_new2, axis=1)
            new_smooth2 = np.max(min_dists_new2)
            if new_smooth2 < old_smooth:
                pts = new_pts
            else:
                lr *= 0.9
        
        # Periodically check true covering radius
        if it % 20 == 0:
            cr = _covering_radius(pts)
            if cr < best_cr:
                best_cr = cr
                best_pts = pts.copy()
    
    # Final check
    final_cr = _covering_radius(pts)
    if final_cr < best_cr:
        best_cr = final_cr
        best_pts = pts.copy()
    
    return _normalize(best_pts)
