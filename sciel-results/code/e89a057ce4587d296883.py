"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import math

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using the convex hull approach.
    The covering radius is the maximum circumradius of all facets of the convex hull
    that have an empty cap (i.e., the facet's circumcenter is on the same side as the origin).
    For points on the unit sphere, the covering radius equals the maximum over all
    Voronoi cells of the distance from the cell's circumcenter to the nearest point.
    
    A more robust approach: the covering radius is the largest angle such that
    there exists a point on the sphere at that angle from all chosen points.
    This is equivalent to the maximum circumradius of the spherical Voronoi diagram.
    
    Practically, we compute the convex hull of the points, and for each triangular face,
    compute the circumradius of that triangle (as a spherical cap). The covering radius
    is the maximum over all such circumradii, but we need to be careful: only faces
    whose circumcenter (normalized) is not inside the convex hull contribute.
    
    Actually, for points on the unit sphere, the covering radius can be computed as:
    For each face of the convex hull (triangle with vertices a, b, c), the circumcenter
    of the spherical triangle is the normalized vector (a + b + c) / |a + b + c| if the
    triangle is acute, or the vertex with the largest angle if obtuse. The distance from
    this circumcenter to the nearest of a, b, c is the circumradius of that cell.
    
    However, a simpler and more robust method: the covering radius is the maximum, over
    all points y on the sphere, of the minimum angular distance from y to any of the N points.
    The maximum is achieved at a vertex of the spherical Voronoi diagram, which is the
    circumcenter of some spherical triangle formed by 3 of the N points.
    
    So we compute the convex hull, and for each face, compute the spherical circumradius.
    """
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        # Fallback: use a grid-based approximation
        return _covering_radius_grid(pts)
    
    max_cr = 0.0
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is proportional to a + b + c if the triangle is acute
        # For obtuse triangles, the circumcenter is the vertex with the largest angle
        
        # First, check if the triangle is acute by checking dot products
        # The circumcenter vector (unnormalized) is the cross product approach:
        # Actually, for a spherical triangle, the circumcenter is the normalized
        # vector in the direction of the solution to the system:
        # c · u = cos(r), b · u = cos(r), a · u = cos(r) where u is the circumcenter
        # This means u is equidistant from a, b, c on the sphere.
        # u is proportional to a × b × ... no, let's use the linear algebra approach.
        
        # The circumcenter u satisfies:
        # |u - a|^2 = |u - b|^2 = |u - c|^2
        # Since |u| = |a| = |b| = |c| = 1, this simplifies to:
        # u · a = u · b = u · c = cos(r)
        # So u is in the null space of the matrix [a-b; b-c; c-a] (approximately)
        # More precisely, u is proportional to (b-c) × (c-a) ... no.
        
        # Let's use: u is the normalized vector that minimizes max(u·a, u·b, u·c) ... no.
        # Actually u·a = u·b = u·c means u is orthogonal to (a-b) and (b-c).
        # So u is parallel to (a-b) × (b-c).
        
        v = np.cross(a - b, b - c)
        if np.linalg.norm(v) < 1e-12:
            # Degenerate triangle
            # The circumradius is the maximum pairwise angular distance / 2
            d_ab = np.arccos(np.clip(np.dot(a, b), -1, 1))
            d_bc = np.arccos(np.clip(np.dot(b, c), -1, 1))
            d_ca = np.arccos(np.clip(np.dot(c, a), -1, 1))
            max_d = max(d_ab, d_bc, d_ca)
            cr = max_d / 2.0
            max_cr = max(max_cr, cr)
            continue
        
        u = v / np.linalg.norm(v)
        
        # Check which side: u or -u
        # We want the circumcenter that is "outside" the triangle, i.e., the one
        # where the cap is empty. For the convex hull face, the outward normal
        # points away from the origin. The circumcenter should be on the outward side.
        
        # Compute the circumradius for u
        cos_r_u = np.dot(u, a)
        cos_r_u = np.clip(cos_r_u, -1, 1)
        r_u = np.arccos(cos_r_u)
        
        # Compute for -u
        cos_r_neg = np.dot(-u, a)
        cos_r_neg = np.clip(cos_r_neg, -1, 1)
        r_neg = np.arccos(cos_r_neg)
        
        # The correct circumcenter is the one that gives the larger circumradius
        # (the one on the "empty cap" side)
        # Actually, for a convex hull face, the outward normal n satisfies n · a > 0 (roughly)
        # The circumcenter on the outward side is the one we want.
        
        # Let's determine the outward normal of the face
        n = np.cross(b - a, c - a)
        n = n / np.linalg.norm(n)
        
        # The outward normal should point away from the origin (since all points are on the unit sphere
        # and the origin is inside the convex hull)
        if np.dot(n, a) < 0:
            n = -n
        
        # Choose u or -u based on which aligns with the outward normal
        if np.dot(u, n) > 0:
            circum_center = u
            cr = r_u
        else:
            circum_center = -u
            cr = r_neg
        
        # But we also need to verify that the circumcenter is actually a valid Voronoi vertex,
        # i.e., it's closer to a, b, c than to any other point. For now, we trust the hull.
        # Also, the circumradius should be the distance to the nearest of a, b, c.
        # Due to numerical issues, let's compute it properly.
        
        # Recompute cr as the maximum angular distance from circum_center to a, b, c
        d_a = np.arccos(np.clip(np.dot(circum_center, a), -1, 1))
        d_b = np.arccos(np.clip(np.dot(circum_center, b), -1, 1))
        d_c = np.arccos(np.clip(np.dot(circum_center, c), -1, 1))
        cr = max(d_a, d_b, d_c)
        
        max_cr = max(max_cr, cr)
    
    return max_cr

def _covering_radius_grid(pts, n_grid=100):
    """Fallback covering radius computation using a spherical grid."""
    pts = _normalize(pts)
    # Create a grid of points on the sphere
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    grid = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid = _normalize(grid)
    
    # Compute distances from each grid point to each chosen point
    # Use matrix multiplication for efficiency
    dots = grid @ pts.T
    dots = np.clip(dots, -1, 1)
    min_dots = np.max(dots, axis=1)  # Maximum dot product = minimum angle
    min_angles = np.arccos(min_dots)
    return np.max(min_angles)

def _fibonacci_points(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _gradient_descent(pts, max_iter=2000, lr_init=0.01, tol=1e-10):
    """Optimize the covering radius using gradient descent on the smooth surrogate.
    We use a soft-max of the squared angular distances as a smooth surrogate for the max."""
    pts = _normalize(pts.copy())
    n = len(pts)
    
    # Use a log-sum-exp based soft-max for smoothness
    beta = 10.0  # Sharpness of the soft-max
    
    def objective_and_grad(pts_flat):
        pts = _normalize(pts_flat.reshape(n, 3))
        # Compute pairwise dot products
        dots = pts @ pts.T
        np.fill_diagonal(dots, 2.0)  # Avoid self-distances
        dots = np.clip(dots, -1 + 1e-15, 1 - 1e-15)
        angles = np.arccos(dots)
        
        # For each point, the distance to its nearest neighbor
        # We want to minimize the maximum of these distances
        # Use soft-max: sum_i exp(beta * d_i^2) / n
        # Actually, let's use the sum of squared distances to nearest neighbor
        # as a smooth surrogate that correlates with the covering radius.
        
        # Better: use the soft-max of the distances themselves
        d_sq = angles * angles
        # Soft-max of d_sq
        max_d_sq = np.max(d_sq, axis=1)
        exp_d = np.exp(beta * (d_sq - max_d_sq[:, None]))
        soft_max = np.log(np.sum(exp_d, axis=1) / n) / beta + max_d_sq
        
        # The covering radius is related to the maximum distance from any sphere point
        # to its nearest chosen point. This is NOT the same as the maximum nearest-neighbor
        # distance among the chosen points.
        
        # Actually, for the covering radius, we need to consider all points on the sphere.
        # The gradient descent on the nearest-neighbor distance among the points is a
        # heuristic that works well in practice for Tammes-type problems.
        
        # Let's use a different approach: minimize the sum of squared nearest-neighbor distances
        # as a smooth surrogate.
        min_d_sq = np.min(d_sq, axis=1)
        obj = np.sum(min_d_sq)
        
        # Gradient computation (approximate)
        # This is complex, so let's use finite differences or a simpler approach.
        # For now, let's use a repulsive force model.
        return obj
    
    # Use a simpler optimization: repulsive forces between points
    # Each point is repelled by others with a force proportional to 1/d^2
    # This tends to maximize the minimum distance, which helps reduce the covering radius.
    
    pts = pts.copy()
    for it in range(max_iter):
        # Compute forces
        dots = pts @ pts.T
        np.fill_diagonal(dots, 2.0)
        dots = np.clip(dots, -1 + 1e-15, 1 - 1e-15)
        angles = np.arccos(dots)
        
        # Force on point i from point j is proportional to (p_j - p_i) / d_ij^2
        # where d_ij is the angular distance
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                d = angles[i, j]
                if d < 1e-10:
                    continue
                # Repulsive force
                f = (pts[j] - pts[i]) / (d * d)
                forces[i] += f
        
        # Update
        lr = lr_init / (1.0 + it * 0.01)
        pts = pts + lr * forces
        pts = _normalize(pts)
        
        # Check convergence
        if it > 0:
            if np.max(np.abs(forces)) < tol:
                break
    
    return _normalize(pts)

def _local_search(pts, max_iter=500, step_size=0.01):
    """Local search using simulated annealing."""
    pts = _normalize(pts.copy())
    n = len(pts)
    current_cr = _covering_radius(pts)
    best_pts = pts.copy()
    best_cr = current_cr
    
    T = 0.05  # Initial temperature
    T_min = 1e-6
    
    for it in range(max_iter):
        # Pick a random point and perturb it
        i = np.random.randint(n)
        # Random direction
        d = np.random.randn(3)
        d = d / np.linalg.norm(d)
        new_pts = pts.copy()
        new_pts[i] = pts[i] + step_size * d
        new_pts = _normalize(new_pts)
        
        new_cr = _covering_radius(new_pts)
        
        # Accept if better or with probability exp(-(new_cr - current_cr) / T)
        if new_cr < current_cr or np.random.rand() < np.exp(-(new_cr - current_cr) / T):
            pts = new_pts
            current_cr = new_cr
            if current_cr < best_cr:
                best_cr = current_cr
                best_pts = pts.copy()
        
        T = max(T_min, T * 0.995)
    
    return _normalize(best_pts), best_cr

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5 second safety margin
    
    # Start with Fibonacci spiral
    pts = _fibonacci_points(N)
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    # Multiple restarts with different initial conditions
    np.random.seed(42)
    
    for restart in range(10):
        if time.time() > deadline:
            break
        
        # Different initializations
        if restart == 0:
            pts = _fibonacci_points(N)
        else:
            # Perturb the best solution found so far
            pts = best_pts + 0.1 * np.random.randn(N, 3)
            pts = _normalize(pts)
        
        # Gradient descent phase
        pts = _gradient_descent(pts, max_iter=1000, lr_init=0.01)
        
        if time.time() > deadline:
            break
        
        # Local search phase
        pts, cr = _local_search(pts, max_iter=300, step_size=0.01)
        
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
        
        if time.time() > deadline:
            break
    
    # Final polish with smaller steps
    if time.time() < deadline - 10:
        pts, cr = _local_search(best_pts, max_iter=500, step_size=0.005)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return _normalize(best_pts)
# EVOLVE-BLOCK-END
