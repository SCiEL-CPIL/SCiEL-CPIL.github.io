"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time


def _angular_dist(p, q):
    """Angular distance in radians between unit vectors p and q."""
    d = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(d)


def _covering_radius(points):
    """
    Estimate the covering radius in degrees.
    Uses a dense sampling of the sphere plus the Voronoi vertices (circumcenters of facets).
    For efficiency, we use a moderately fine Fibonacci grid and also check the
    circumcenters of the convex hull facets via scipy if available.
    """
    # Method 1: dense Fibonacci sampling
    M = 2000
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # For each grid point, find min angular distance to any point
    # Use matrix multiplication for speed
    # dist^2 = 2 - 2 * (grid @ points.T)
    dots = grid @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)  # max dot = min angular distance
    min_angles = np.arccos(min_dots)
    cr_grid = np.max(min_angles) * 180.0 / np.pi

    # Method 2: check circumcenters of convex hull facets using scipy
    try:
        from scipy.spatial import ConvexHull
        hull = ConvexHull(points)
        # For each facet, compute the circumcenter and check if it's in the cap
        max_cr = cr_grid
        for simplex in hull.simplices:
            facet_pts = points[simplex]
            # Circumcenter of a spherical triangle (3 points on unit sphere)
            # The circumcenter is the normalized cross product of (p2-p1) x (p3-p1)
            # Actually for a triangle on the unit sphere, the circumcenter direction is
            # the normalized sum of the three points weighted appropriately, or we can
            # use the formula: circumcenter = (p1 + p2 + p3) normalized only if equilateral.
            # Better: find the point on the sphere equidistant from all 3 vertices.
            # This is the intersection of the unit sphere with the line through the
            # centroid of the 3 points in 3D space (the circumcenter of the planar triangle
            # projected). Actually, the spherical circumcenter is in the direction of
            # the cross products.
            p1, p2, p3 = facet_pts[0], facet_pts[1], facet_pts[2]
            # The circumcenter direction is parallel to (p1 x p2) x (p1 x p3) or similar.
            # Actually, it's the normalized vector: (p1 x p2) x (p2 x p3) + ... 
            # Simpler: the circumcenter is along the normal to the plane containing p1,p2,p3
            # only if the triangle is right-angled. Let's use the formula:
            # circumcenter = normalize( (|p2|^2(p3-p1) + |p3|^2(p1-p2)) x ... )
            # Since all points are on unit sphere, the circumcenter of the spherical triangle
            # is in the direction of the vector: (p1 x p2) x (p1 x p3)
            v = np.cross(np.cross(p1, p2), np.cross(p1, p3))
            norm = np.linalg.norm(v)
            if norm < 1e-15:
                continue
            cc = v / norm
            # Check if this circumcenter is actually the farthest point in the Voronoi cell
            # It should be the point equidistant from p1, p2, p3 and farther from all others
            # Compute its angular distance to the nearest point
            dots_cc = np.clip(points @ cc, -1.0, 1.0)
            min_dot_cc = np.max(dots_cc)
            angle_cc = np.arccos(min_dot_cc)
            # Verify that the circumcenter is actually in the Voronoi cell of this facet
            # i.e., it's equidistant to p1, p2, p3 and no closer to any other point
            d1 = np.arccos(np.clip(np.dot(cc, p1), -1, 1))
            d2 = np.arccos(np.clip(np.dot(cc, p2), -1, 1))
            d3 = np.arccos(np.clip(np.dot(cc, p3), -1, 1))
            # The circumcenter should be equidistant to the 3 vertices
            # If the Voronoi vertex is this circumcenter, then the covering radius contribution is d1
            if abs(d1 - d2) < 1e-6 and abs(d1 - d3) < 1e-6:
                # Check no other point is closer
                other_dots = dots_cc.copy()
                other_dots[simplex] = -1.0  # exclude the 3 vertices
                max_other_dot = np.max(other_dots)
                if max_other_dot <= np.dot(cc, p1) + 1e-9:
                    cr_facet = d1 * 180.0 / np.pi
                    max_cr = max(max_cr, cr_facet)
    except Exception:
        pass

    return max_cr


def _fibonacci_spiral(n, offset=0.5):
    i = np.arange(n) + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.where(norms < 1e-15, 1.0, norms)
    return pts / norms


def _repulsion_energy(points):
    """Sum of inverse-square distances (soft repulsion)."""
    # Compute all pairwise dots
    dots = points @ points.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    # Energy = sum over i<j of 1/(1 - dot_ij)^2 to avoid singularity
    # Use 1 - dot as proxy for distance squared (approx)
    denom = np.maximum(1.0 - dots, 1e-12)
    energy = np.sum(1.0 / (denom ** 2))
    # Subtract self terms
    energy -= N * 1.0 / (1e-12 ** 2)  # this is huge, let's use a different formulation
    # Better: use only off-diagonal
    mask = ~np.eye(N, dtype=bool)
    energy = np.sum(1.0 / (denom[mask] ** 2))
    return energy


def _gradient_repulsion(points):
    """
    Gradient of the repulsion energy with respect to each point.
    E = sum_{i<j} (1 - p_i . p_j)^(-2)
    dE/dp_i = -2 * sum_{j != i} (1 - p_i . p_j)^(-3) * (p_j - p_i)
    Then project to tangent space: grad_i -= (grad_i . p_i) * p_i
    """
    dots = points @ points.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    denom = np.maximum(1.0 - dots, 1e-12)
    # (1 - dot)^(-3)
    weights = 1.0 / (denom ** 3)
    np.fill_diagonal(weights, 0.0)
    # grad_i = -2 * sum_j weights[i,j] * (p_j - p_i)
    # = -2 * (weights @ points - weights @ (points * weights_sum_per_row)[:,None])
    # Actually: sum_j w_ij * p_j - sum_j w_ij * p_i = (W @ P)_i - (rowsum(W)_i) * P_i
    W = weights  # (N, N)
    Wsum = W.sum(axis=1)  # (N,)
    grad = -2.0 * (W @ points - Wsum[:, None] * points)
    # Project to tangent space
    grad_dot_p = np.sum(grad * points, axis=1, keepdims=True)
    grad = grad - grad_dot_p * points
    return grad


def _grad_covering_radius(points):
    """
    Approximate gradient of the covering radius.
    The covering radius is the max over Voronoi vertices of the distance to the nearest point.
    For a given Voronoi vertex v (circumcenter of a facet), the gradient w.r.t. the points
    in that facet is: for each point p_i in the facet, the gradient is in the direction
    that moves p_i toward v (to reduce the distance).
    
    We approximate by: for each point, find the farthest Voronoi vertex (or grid point)
    and move toward it.
    """
    # Use a moderate grid for gradient estimation
    M = 500
    i = np.arange(M) + 0.5
    z = 1.0 - 2.0 * i / M
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    grid = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    dots = grid @ points.T  # (M, N)
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)  # (M,)
    max_angles = np.arccos(max_dots)  # (M,)
    
    # Find the worst grid point
    worst_idx = np.argmax(max_angles)
    worst_grid_pt = grid[worst_idx]
    worst_dot = max_dots[worst_idx]
    nearest_idx = np.argmax(dots[worst_idx])
    
    # Gradient: move the nearest point toward the worst grid point
    grad = np.zeros_like(points)
    # The direction to move point nearest_idx is toward worst_grid_pt
    direction = worst_grid_pt - points[nearest_idx]
    # Project to tangent space
    dir_dot_p = np.dot(direction, points[nearest_idx])
    direction = direction - dir_dot_p * points[nearest_idx]
    grad[nearest_idx] = direction
    
    # Also add a small repulsion gradient to maintain spread
    rep_grad = _gradient_repulsion(points)
    grad += 0.01 * rep_grad
    
    return grad


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # 10s safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_spiral(N)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Phase 1: Gradient-based optimization with adaptive step size
    alpha = 0.01
    for iteration in range(5000):
        if time.time() > deadline - 20.0:
            break
        
        grad = _grad_covering_radius(points)
        
        # Adaptive step size: reduce if gradient is large
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-10:
            break
        
        # Normalize gradient to have unit norm per point
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms = np.where(grad_norms < 1e-15, 1.0, grad_norms)
        grad_normalized = grad / grad_norms
        
        # Take a step
        new_points = points + alpha * grad_normalized
        new_points = _normalize(new_points)
        
        new_cr = _covering_radius(new_points)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
            alpha = min(alpha * 1.05, 0.05)  # slowly increase step
        else:
            alpha = alpha * 0.9  # reduce step
            # Try smaller step
            small_step = points + alpha * 0.1 * grad_normalized
            small_step = _normalize(small_step)
            small_cr = _covering_radius(small_step)
            if small_cr < best_cr:
                best_cr = small_cr
                best_points = small_step.copy()
                points = small_step
    
    # Phase 2: Local refinement with very small steps
    alpha = 0.001
    for iteration in range(3000):
        if time.time() > deadline - 5.0:
            break
        
        grad = _grad_covering_radius(points)
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-12:
            break
        
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms = np.where(grad_norms < 1e-15, 1.0, grad_norms)
        grad_normalized = grad / grad_norms
        
        new_points = points + alpha * grad_normalized
        new_points = _normalize(new_points)
        
        new_cr = _covering_radius(new_points)
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
    
    # Phase 3: High-precision polish with adaptive step-size reduction
    base_alpha = 0.0001
    for scale in [1.0, 0.5, 0.25, 0.1]:
        if time.time() > deadline - 2.0:
            break
        alpha = base_alpha * scale
        for iteration in range(1000):
            if time.time() > deadline - 2.0:
                break
            
            grad = _grad_covering_radius(points)
            grad_norm = np.linalg.norm(grad)
            if grad_norm < 1e-14:
                break
            
            grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
            grad_norms = np.where(grad_norms < 1e-15, 1.0, grad_norms)
            grad_normalized = grad / grad_norms
            
            new_points = points + alpha * grad_normalized
            new_points = _normalize(new_points)
            
            new_cr = _covering_radius(new_points)
            
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = new_points.copy()
                points = new_points
    
    # Final check: ensure no coincident points and valid configuration
    # Verify all points are on the unit sphere
    best_points = _normalize(best_points)
    
    # Check for coincident points
    dots = best_points @ best_points.T
    np.fill_diagonal(dots, 0.0)
    if np.max(dots) > 0.999999:
        # Perturb coincident points
        for i in range(N):
            for j in range(i + 1, N):
                if dots[i, j] > 0.999999:
                    # Move point j slightly
                    noise = np.random.normal(0, 0.01, 3)
                    best_points[j] += noise
                    best_points[j] = best_points[j] / np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
