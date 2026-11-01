"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # seconds, safety margin before 150s kill


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute the covering radius in radians.
    
    The covering radius is the maximum over all points y on the sphere of the
    angular distance to the nearest configuration point. This equals the maximum
    circumradius of the facets of the convex hull of the points (where each facet
    has an empty cap).
    
    We compute this via the Voronoi diagram on the sphere: for each point, find
    its Voronoi cell, and the covering radius is the max over all points of the
    maximum distance from that point to any vertex of its Voronoi cell.
    
    Actually, the covering radius is the maximum distance from any point on the
    sphere to its nearest configuration point. This is equivalent to: for each
    Voronoi vertex (which is the circumcenter of a Delaunay triangle), compute
    its distance to the nearest configuration point, and take the max.
    
    A simpler approach: the covering radius is the maximum over all points on
    the sphere of min_i angular_dist(y, p_i). The worst points are the Voronoi
    vertices (circumcenters of Delaunay triangles). So we can compute the
    Delaunay triangulation of the points on the sphere and find the circumcenter
    of each triangle, then compute the max distance from each circumcenter to
    the nearest point.
    
    For a practical approach, we use the fact that the covering radius is
    approximately the maximum over all pairs/triples of the circumradius.
    """
    # Use scipy's ConvexHull to find facets
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a grid-based approximation
        return _covering_radius_grid(pts)
    
    # For each facet (triangle), compute the circumcenter and its distance to nearest point
    max_r = 0.0
    for simplex in hull.simplices:
        if len(simplex) == 3:
            a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
            # Circumcenter of triangle abc on the sphere
            # The circumcenter is the point equidistant from a, b, c
            # It lies along the normal to the plane of abc
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-12:
                continue
            n = n / n_norm
            
            # The circumcenter is at distance d from origin where:
            # |d*n - a|^2 = |d*n - b|^2 = |d*n - c|^2
            # d^2 - 2d(a.n) + |a|^2 = d^2 - 2d(b.n) + |b|^2
            # Since |a|=|b|=|c|=1: a.n = b.n = c.n (they're coplanar with normal n)
            # So the circumcenter is at d*n where d is determined by:
            # The circumradius R satisfies R^2 = d^2 - 1 (since points are on unit sphere)
            # Actually, let's just compute the circumcenter properly.
            
            # For three points on the unit sphere, the circumcenter is:
            # circumcenter = (a + b + c) / |a + b + c| if they form an acute triangle
            # More generally, we need to find the point on the line through origin
            # perpendicular to the plane that is equidistant from all three.
            
            # The plane through a, b, c has equation n.x = h where h = a.n
            # The circumcenter lies on the line t*n. We need |t*n - a| = |t*n - b|
            # t^2 - 2t(a.n) + 1 = t^2 - 2t(b.n) + 1
            # This is always true since a.n = b.n = c.n = h.
            # So any point on t*n is equidistant from a, b, c.
            # The circumcenter is the one closest to the triangle, which is at
            # t = h (the foot of the perpendicular from origin to the plane).
            # But this is the incenter of the tetrahedron, not the circumcenter.
            
            # Let me think again. The circumcenter of triangle abc is the point
            # equidistant from a, b, c. On the sphere, this is the point where
            # the great circles through pairs of antipodal points intersect.
            
            # Actually, for points on the unit sphere, the circumcenter of triangle
            # abc (as a spherical triangle) is the point x on the unit sphere such
            # that angular_dist(x,a) = angular_dist(x,b) = angular_dist(x,c).
            # This means x.a = x.b = x.c (since cos is monotonic for angles in [0,pi]).
            # So x is in the null space of the matrix with rows [a-b, a-c], i.e.,
            # x is parallel to cross(b-a, c-a) = n.
            # So x = ±n/|n|. We pick the sign such that x is on the same side
            # as the triangle (i.e., x.a > 0 for acute, etc.)
            
            # The circumradius is arccos(x.a) where x is the circumcenter.
            x = n  # or -n
            # Pick the sign: we want the circumcenter to be the one that gives
            # the smaller cap (the one containing the triangle)
            if x.dot(a) < 0:
                x = -x
            
            r = np.arccos(np.clip(x.dot(a), -1.0, 1.0))
            max_r = max(max_r, r)
        elif len(simplex) == 4:
            # Tetrahedral facet - shouldn't happen for points in general position
            # on the sphere, but handle it
            pass
    
    if max_r == 0.0:
        # Fallback
        return _covering_radius_grid(pts)
    
    return max_r


def _covering_radius_grid(pts):
    """Approximate covering radius using a grid."""
    # Use a reasonably fine grid
    n_theta = 40
    n_phi = 80
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    
    # Convert to Cartesian
    X = np.sin(T) * np.cos(P)
    Y = np.sin(T) * np.sin(P)
    Z = np.cos(T)
    
    grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # For each grid point, find min distance to any configuration point
    # Use broadcasting (might be memory intensive but N=50, grid ~3200)
    # dist^2 = |g|^2 + |p|^2 - 2*g.p = 2 - 2*g.p (since both on unit sphere)
    # angular_dist = arccos(g.p)
    
    dots = grid_pts @ pts.T  # shape (n_grid, N)
    min_dots = dots.max(axis=1)  # max dot product = min angular distance
    min_dots = np.clip(min_dots, -1.0, 1.0)
    max_angle = np.arccos(min_dots).max()
    
    return max_angle


def _soft_max_covering(pts, temperature=0.05):
    """Smooth surrogate for the covering radius using log-sum-exp.
    
    For each point on a set of test points, compute the min distance to nearest
    configuration point, then use a soft-max over the test points to get a
    smooth approximation of the max.
    """
    # Generate test points (Voronoi vertices would be ideal, but use a grid)
    n_theta = 30
    n_phi = 60
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    
    X = np.sin(T) * np.cos(P)
    Y = np.sin(T) * np.sin(P)
    Z = np.cos(T)
    
    grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # dots[i,j] = grid_pts[i] . pts[j]
    dots = grid_pts @ pts.T  # (n_grid, N)
    # Min angular distance for each grid point = arccos(max_j dots[i,j])
    max_dots = dots.max(axis=1)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)  # (n_grid,)
    
    # Soft max: temperature * log(sum(exp(angles / temperature)))
    scaled = angles / temperature
    max_scaled = scaled.max()
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return temperature * log_sum_exp


def _soft_max_covering_grad(pts, temperature=0.05):
    """Gradient of the soft-max covering radius surrogate."""
    n_theta = 30
    n_phi = 60
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    
    X = np.sin(T) * np.cos(P)
    Y = np.sin(T) * np.sin(P)
    Z = np.cos(T)
    
    grid_pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    dots = grid_pts @ pts.T  # (n_grid, N)
    max_dots = dots.max(axis=1)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    
    scaled = angles / temperature
    max_scaled = scaled.max()
    exp_scaled = np.exp(scaled - max_scaled)
    sum_exp = np.sum(exp_scaled)
    weights = exp_scaled / sum_exp  # softmax weights over grid points
    
    # d(soft_max)/d(angles) = weights
    # d(angles)/d(max_dots) = -1/sqrt(1 - max_dots^2)
    # d(max_dots)/d(dots) = one_hot(argmax)
    # d(dots)/d(pts) = grid_pts.T (for the argmax component)
    
    # This is getting complex. Let's use numerical gradient instead for simplicity.
    return None


def _repulsion_relax(pts, n_iter=200, dt=0.01):
    """Repulsion-based relaxation to spread points on the sphere."""
    pts = _normalize(pts.copy())
    for it in range(n_iter):
        forces = np.zeros_like(pts)
        for i in range(N):
            for j in range(i+1, N):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 1e-10:
                    d = np.random.randn(3) * 0.01
                    dist = np.linalg.norm(d)
                # Repulsive force: stronger for closer points
                f = d / (dist ** 3)
                forces[i] += f
                forces[j] -= f
        pts += dt * forces
        pts = _normalize(pts)
    return pts


def construct():
    start_time = time.time()
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    best_pts = pts.copy()
    best_radius = _covering_radius(pts)
    
    # Step 1: Repulsion relaxation for better initial spread
    if time.time() - start_time < DEADLINE - 30:
        pts = _repulsion_relax(pts, n_iter=100, dt=0.02)
        r = _covering_radius(pts)
        if r < best_radius:
            best_radius = r
            best_pts = pts.copy()
    
    # Step 2: L-BFGS optimization with soft-max surrogate
    if time.time() - start_time < DEADLINE - 10:
        temperature = 0.05
        
        def objective(x):
            p = _normalize(x.reshape(N, 3))
            return _soft_max_covering(p, temperature)
        
        # Use multiple restarts with different temperatures
        for temp in [0.1, 0.05, 0.02]:
            if time.time() - start_time >= DEADLINE - 10:
                break
            
            temperature = temp
            result = minimize(objective, pts.flatten(), method='L-BFGS-B',
                             options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.success:
                new_pts = _normalize(result.x.reshape(N, 3))
                new_r = _covering_radius(new_pts)
                if new_r < best_radius:
                    best_radius = new_r
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
    
    # Step 3: Lloyd's algorithm / iterative refinement
    if time.time() - start_time < DEADLINE - 5:
        pts = best_pts.copy()
        for it in range(50):
            if time.time() - start_time >= DEADLINE - 5:
                break
            
            # For each point, move it toward the centroid of its Voronoi cell
            # Approximate: for each point, compute the average of nearby points
            # weighted by inverse distance, then move toward that direction
            
            new_pts = pts.copy()
            for i in range(N):
                # Find the "Voronoi cell" approximation: points for which i is
                # the nearest or among nearest
                dots = pts @ pts[i]
                # Points where i is the unique nearest
                mask = (dots > 1e-6) & (dots < 1.0)
                # Weight by how much closer i is than others
                # Simple approach: move toward the average of points in a cap
                # around i, weighted by the angular distance
                
                # Better: compute the gradient of the soft-max and take a step
                pass
            
            # Alternative: use the soft-max gradient
            def objective(x):
                p = _normalize(x.reshape(N, 3))
                return _soft_max_covering(p, 0.02)
            
            result = minimize(objective, pts.flatten(), method='L-BFGS-B',
                             options={'maxiter': 20, 'ftol': 1e-14, 'gtol': 1e-12})
            if result.success:
                new_pts = _normalize(result.x.reshape(N, 3))
                new_r = _covering_radius(new_pts)
                if new_r < best_radius:
                    best_radius = new_r
                    best_pts = new_pts.copy()
                    pts = new_pts.copy()
    
    return best_pts
