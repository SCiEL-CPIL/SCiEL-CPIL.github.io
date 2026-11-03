"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import Delaunay
from scipy.spatial.distance import cdist

def _spherical_centroid(pts, cell_points):
    """
    Compute the spherical centroid of a set of points on the unit sphere.
    Uses the normalized sum of vectors weighted by solid angle approximation.
    """
    if len(cell_points) == 0:
        return None
    if len(cell_points) == 1:
        return cell_points[0].copy()
    
    # Weight by the angle subtended at the center
    # For a Voronoi cell, the centroid is approximately the normalized sum
    # of the vertex vectors, but better to use solid angle weights
    
    # Simple approach: normalize the sum of all vertex vectors
    # This gives the "mean direction" which is a good approximation for small cells
    vec_sum = np.sum(cell_points, axis=0)
    norm = np.linalg.norm(vec_sum)
    if norm < 1e-10:
        # Degenerate case: use uniform average direction
        vec_sum = np.mean(cell_points, axis=0)
        norm = np.linalg.norm(vec_sum)
        if norm < 1e-10:
            return cell_points[0].copy()
    return vec_sum / norm

def _compute_voronoi_centroids(pts, n_iter=1):
    """
    Compute Voronoi cell centroids for points on the sphere using
    Delaunay triangulation in 3D (projected).
    
    Returns array of centroids, one per input point.
    """
    N = len(pts)
    centroids = np.zeros_like(pts)
    
    for i in range(N):
        # Find all points that are "neighbors" of pts[i] in the Delaunay sense
        # For now, use a simpler approach: find the k-nearest neighbors
        # and compute their spherical centroid
        pass
    
    return centroids

def _lloyd_iteration(pts, grid, alpha=1.0):
    """
    Perform one Lloyd iteration: move each point to the centroid of its
    spherical Voronoi cell.
    """
    N = len(pts)
    M = grid.shape[0]
    
    # Find nearest point for each grid point
    dots = grid @ pts.T  # (M, N)
    nearest = np.argmax(dots, axis=1)  # (M,)
    
    # For each point, collect the grid points in its Voronoi cell
    centroids = np.zeros_like(pts)
    
    for i in range(N):
        mask = (nearest == i)
        if np.sum(mask) > 0:
            cell_pts = grid[mask]
            vec_sum = np.sum(cell_pts, axis=0)
            norm = np.linalg.norm(vec_sum)
            if norm > 1e-10:
                centroids[i] = vec_sum / norm
            else:
                centroids[i] = pts[i]
        else:
            centroids[i] = pts[i]
    
    # Move points toward their centroids
    new_pts = pts + alpha * (centroids - pts)
    new_pts = new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)
    
    return new_pts

def _lloyd_optimize(pts, grid, time_limit=10.0):
    """
    Run Lloyd optimization for a given time limit.
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts, grid)
    
    # Adaptive alpha: start large, decay
    alpha = 0.5
    min_alpha = 0.01
    
    while time.time() - start_time < time_limit:
        new_pts = _lloyd_iteration(current_pts, grid, alpha=alpha)
        
        # Check for improvement
        new_radius = _compute_covering_radius(new_pts, grid)
        
        if new_radius < best_radius - 1e-8:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts
            # Decay alpha slowly
            alpha = max(alpha * 0.99, min_alpha)
        else:
            # No improvement, reduce alpha more aggressively
            alpha = max(alpha * 0.9, min_alpha)
            if alpha <= min_alpha * 1.01:
                break
    
    return best_pts, best_radius

def _compute_covering_radius(pts, grid):
    """Compute the covering radius (max angular distance) using grid."""
    dots = grid @ pts.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=1)
    max_dots = np.clip(max_dots, -1.0, 1.0)
    angles = np.arccos(max_dots)
    return np.max(angles)

def _compute_facet_circumradii(pts):
    """
    Compute the circumradius of each facet of the convex hull.
    The covering radius is the maximum circumradius among all facets
    whose circumcenter lies outside the sphere (empty cap condition).
    """
    from scipy.spatial import ConvexHull
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.zeros(len(pts)), np.zeros(len(pts))
    
    N = len(pts)
    facet_radii = np.zeros(N)
    facet_indices = np.zeros(N, dtype=int)
    
    # For each triangular facet, compute the circumradius
    # The circumradius of a triangle with vertices a, b, c on the unit sphere
    # is related to the angles between them
    for face_idx, face in enumerate(hull.simplices):
        a, b, c = pts[face[0]], pts[face[1]], pts[face[2]]
        
        # Compute angles between pairs
        ab = np.clip(np.dot(a, b), -1, 1)
        ac = np.clip(np.dot(a, c), -1, 1)
        bc = np.clip(np.dot(b, c), -1, 1)
        
        alpha = np.arccos(ab)
        beta = np.arccos(ac)
        gamma = np.arccos(bc)
        
        # Circumradius of spherical triangle
        # Using the formula: R = sqrt(1 - (sin(s/2) * sin((s-alpha)/2) * sin((s-beta)/2) * sin((s-gamma)/2)) / (sin(alpha/2)*sin(beta/2)*sin(gamma/2)))^2
        # where s = (alpha + beta + gamma) / 2
        s = (alpha + beta + gamma) / 2
        
        # Alternative: compute circumcenter directly
        # The circumcenter is the point equidistant from all three vertices
        # Solve: |x - a|^2 = |x - b|^2 = |x - c|^2, |x| = 1
        # This gives: x·(b-a) = (|b|^2 - |a|^2)/2 = 0 (since on unit sphere)
        # x·(c-a) = 0
        # So x is perpendicular to both (b-a) and (c-a), i.e., x = ±(b-a)×(c-a)/|(b-a)×(c-a)|
        
        cross = np.cross(b - a, c - a)
        cross_norm = np.linalg.norm(cross)
        if cross_norm < 1e-10:
            continue
        
        circumcenter = cross / cross_norm
        
        # Check if the circumcenter is on the same side as the facet normal
        # The facet normal should point outward
        normal = np.cross(b - a, c - a)
        normal = normal / np.linalg.norm(normal)
        
        # Check empty cap: the circumcenter should be such that the cap
        # centered at circumcenter with radius R contains no other points
        # For now, just compute the circumradius
        R = np.arccos(np.clip(np.dot(a, circumcenter), -1, 1))
        
        # Assign to the first vertex of this facet (for tracking)
        facet_radii[face[0]] = max(facet_radii[face[0]], R)
        facet_indices[face[0]] = face_idx
    
    return facet_radii, facet_indices

def _polish_largest_facets(pts, grid, time_limit, n_polish=50):
    """
    Polish the configuration by moving points that are centers of large
    Voronoi cells toward the circumcenters of their largest facets.
    """
    start_time = time.time()
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _compute_covering_radius(pts, grid)
    
    for polish_iter in range(n_polish):
        if time.time() - start_time > time_limit:
            break
        
        # Find the point with the largest Voronoi cell (using grid)
        dots = grid @ current_pts.T
        nearest = np.argmax(dots, axis=1)
        
        # Count grid points per cell
        cell_sizes = np.bincount(nearest, minlength=len(current_pts))
        
        # Find the largest cell
        max_cell_idx = np.argmax(cell_sizes)
        
        if cell_sizes[max_cell_idx] == 0:
            break
        
        # Get the cell points
        mask = (nearest == max_cell_idx)
        cell_pts = grid[mask]
        
        # Compute the centroid of the largest cell
        vec_sum = np.sum(cell_pts, axis=0)
        norm = np.linalg.norm(vec_sum)
        if norm < 1e-10:
            break
        centroid = vec_sum / norm
        
        # Move the point toward the centroid with a small step
        step_size = 0.05
        new_pt = current_pts[max_cell_idx] + step_size * (centroid - current_pts[max_cell_idx])
        new_pt = new_pt / np.linalg.norm(new_pt)
        
        # Try this move
        trial_pts = current_pts.copy()
        trial_pts[max_cell_idx] = new_pt
        
        trial_radius = _compute_covering_radius(trial_pts, grid)
        
        if trial_radius < best_radius - 1e-10:
            best_radius = trial_radius
            best_pts = trial_pts.copy()
            current_pts = trial_pts
    
    return best_pts, best_radius

def _generate_grid(n_points=20000):
    """Generate a dense set of points on the sphere for discretization."""
    i = np.arange(n_points) + 0.5
    z = 1.0 - 2.0 * i / n_points
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _fibonacci_sphere(n):
    """Generate n points on the sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, rng):
    """Generate n random points on the sphere."""
    # Uniform distribution on sphere
    z = rng.uniform(-1, 1, n)
    phi = rng.uniform(0, 2 * np.pi, n)
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    total_time_limit = 130.0  # Safety margin before 150s kill
    
    # Use a dense grid for accurate covering radius estimation
    grid = _generate_grid(20000)
    fine_grid = _generate_grid(50000)
    
    best_pts = None
    best_radius = np.inf
    
    rng = np.random.RandomState(42)
    
    # Try multiple initializations
    initializations = []
    
    # 1. Fibonacci spiral
    initializations.append(_fibonacci_sphere(N))
    
    # 2-5. Random initializations
    for _ in range(4):
        initializations.append(_random_sphere(N, rng))
    
    # 6. Perturbed Fibonacci
    fib = _fibonacci_sphere(N)
    perturbed = fib + rng.normal(0, 0.1, fib.shape)
    perturbed = perturbed / np.linalg.norm(perturbed, axis=1, keepdims=True)
    initializations.append(perturbed)
    
    time_per_init = (total_time_limit - 10.0) / len(initializations)
    
    for init_idx, init_pts in enumerate(initializations):
        if time.time() - start_time > total_time_limit - 10:
            break
        
        time_left = min(time_per_init, total_time_limit - (time.time() - start_time) - 5)
        if time_left < 2:
            continue
        
        # Run Lloyd optimization
        result_pts, result_rad = _lloyd_optimize(init_pts, grid, time_limit=time_left * 0.7)
        
        # Polish the largest facets
        polish_time = min(time_left * 0.3, total_time_limit - (time.time() - start_time) - 5)
        if polish_time > 1:
            polished_pts, polished_rad = _polish_largest_facets(result_pts, grid, time_limit=polish_time)
            if polished_rad < result_rad:
                result_pts = polished_pts
                result_rad = polished_rad
        
        # Verify with a finer grid for final score
        fine_rad = _compute_covering_radius(result_pts, fine_grid)
        
        if fine_rad < best_radius:
            best_radius = fine_rad
            best_pts = result_pts.copy()
    
    if best_pts is None:
        # Fallback to Fibonacci
        best_pts = _fibonacci_sphere(N)
    
    return best_pts
# EVOLVE-BLOCK-END
