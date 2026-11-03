import numpy as np
from scipy.spatial import ConvexHull
import time

N = 130

def _compute_circumradii(points, hull):
    """
    Compute circumradii and circumcenters for all hull facets.
    Returns:
        radii: (M,) array of circumradii (angular)
        centers: (M, 3) array of circumcenters on sphere
        facets: (M, 3) array of facet indices
    """
    facets = hull.simplices
    M = len(facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    cross_sq = np.einsum('ij,ij->i', cross, cross)
    
    valid = cross_sq > 1e-28
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        dot_cp = np.einsum('ij,ij->i', cross[valid], p1[valid])
        t = dot_cp / cross_sq[valid]
        c_3d = p1[valid] + t[:, np.newaxis] * cross[valid]
        
        norms = np.linalg.norm(c_3d, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        c = c_3d / norms
        
        centroids = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dots = np.einsum('ij,ij->i', c, centroids)
        flip = dots < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        
        dot_vals = np.einsum('ij,ij->i', c, p1[valid])
        dot_vals = np.clip(dot_vals, -1.0, 1.0)
        radii[valid] = np.arccos(dot_vals)
    
    return radii, centers, facets
    """
    Compute circumradii and circumcenters for all hull facets.
    Returns:
        radii: (M,) array of circumradii (angular)
        centers: (M, 3) array of circumcenters on sphere
        facets: (M, 3) array of facet indices
    """
    facets = hull.simplices
    M = len(facets)
    
    p1 = points[facets[:, 0]]
    p2 = points[facets[:, 1]]
    p3 = points[facets[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    cross = np.cross(v1, v2)
    cross_sq = np.einsum('ij,ij->i', cross, cross)
    
    valid = cross_sq > 1e-28
    centers = np.zeros((M, 3))
    radii = np.zeros(M)
    
    if np.any(valid):
        # Circumcenter via: p1 + t * cross, where t = (cross . p1) / (cross . cross)
        # This gives the 3D circumcenter; normalize to get spherical circumcenter
        dot_cp = np.einsum('ij,ij->i', cross[valid], p1[valid])
        t = dot_cp / cross_sq[valid]
        c_3d = p1[valid] + t[:, np.newaxis] * cross[valid]
        
        # Normalize to unit sphere
        norms = np.linalg.norm(c_3d, axis=1, keepdims=True)
        norms[norms < 1e-14] = 1.0
        c = c_3d / norms
        
        # Ensure centers point to the empty cap side (same side as centroid)
        centroids = (p1[valid] + p2[valid] + p3[valid]) / 3.0
        dots = np.einsum('ij,ij->i', c, centroids)
        flip = dots < 0
        c[flip] = -c[flip]
        
        centers[valid] = c
        
        # Compute angular distances
        dot_vals = np.einsum('ij,ij->i', c, p1[valid])
        dot_vals = np.clip(dot_vals, -1.0, 1.0)
        radii[valid] = np.arccos(dot_vals)
    
    return radii, centers, facets

def _compute_covering_radius(points, hull=None):
    """
    Compute the covering radius in degrees.
    """
    try:
        if hull is None:
            hull = ConvexHull(points)
        radii, centers, facets = _compute_circumradii(points, hull)
        return np.degrees(np.max(radii))
    except Exception:
        return 180.0

def _fibonacci_init(seed=42):
    """Generate a Fibonacci spiral initialization."""
    rng = np.random.default_rng(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    # Small random perturbation to break symmetry
    points += rng.normal(0, 0.01, points.shape)
    points /= np.linalg.norm(points, axis=1, keepdims=True)
    return points

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-14] = 1.0
    return points / norms

def _simulated_annealing(points, start_time, deadline, rng):
    """
    Simulated annealing search: perturb random subsets of points,
    accept/reject based on Metropolis criterion.
    """
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    current_radius = best_radius
    current_points = points.copy()
    
    # Annealing schedule
    T0 = 0.5  # initial temperature (in degrees)
    Tend = 0.001
    total_iterations = 5000
    cool_rate = (Tend / T0) ** (1.0 / max(total_iterations, 1))
    
    # Base perturbation magnitude
    base_perturb = 0.02
    
    for it in range(total_iterations):
        if time.time() >= deadline:
            break
        
        T = T0 * (cool_rate ** it)
        
        # Decide how many points to perturb
        if T > 0.1:
            n_perturb = rng.integers(1, 5)  # more aggressive early
        elif T > 0.01:
            n_perturb = rng.integers(1, 3)
        else:
            n_perturb = 1
        
        # Pick which points to perturb
        idx = rng.choice(N, size=n_perturb, replace=False)
        
        # Generate perturbations: random directions on tangent plane
        new_points = current_points.copy()
        for i in idx:
            # Random direction in 3D
            rand_dir = rng.normal(0, 1, 3)
            # Project to tangent plane
            tangent = rand_dir - np.dot(rand_dir, current_points[i]) * current_points[i]
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-14:
                tangent /= norm_t
            # Perturb magnitude scaled by temperature
            magnitude = base_perturb * (0.5 + T / T0)
            new_points[i] = current_points[i] + magnitude * tangent
        
        new_points = _normalize(new_points)
        
        # Evaluate
        new_radius = _compute_covering_radius(new_points)
        
        # Metropolis acceptance
        delta = new_radius - current_radius
        if delta < 0 or rng.random() < np.exp(-delta / max(T, 1e-10)):
            current_points = new_points
            current_radius = new_radius
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
    
    return best_points, best_radius

def _local_refinement(points, start_time, deadline, rng):
    """
    Local refinement: for each point, try small moves in random directions
    and keep improvements. This is a form of coordinate descent.
    """
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    current_radius = best_radius
    current_points = points.copy()
    
    n_rounds = 200
    perturb_magnitudes = [0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    for round_idx in range(n_rounds):
        if time.time() >= deadline:
            break
        
        pmag = perturb_magnitudes[min(round_idx // 40, len(perturb_magnitudes) - 1)]
        
        for i in range(N):
            if time.time() >= deadline:
                    break
            
            # Try several random directions
            for trial in range(3):
                rand_dir = rng.normal(0, 1, 3)
                tangent = rand_dir - np.dot(rand_dir, current_points[i]) * current_points[i]
                norm_t = np.linalg.norm(tangent)
                if norm_t < 1e-14:
                    continue
                tangent /= norm_t
                
                new_points = current_points.copy()
                new_points[i] = current_points[i] + pmag * tangent
                new_points = _normalize(new_points)
                
                new_radius = _compute_covering_radius(new_points)
                
                if new_radius < current_radius:
                    current_points = new_points
                    current_radius = new_radius
                    if new_radius < best_radius:
                        best_radius = new_radius
                        best_points = new_points.copy()
                    break
    
    return best_points, best_radius

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    rng = np.random.default_rng(42)
    
    # Phase 1: Simulated annealing for global exploration
    points = _fibonacci_init(seed=42)
    sa_deadline = start_time + 60.0
    points, best_radius = _simulated_annealing(points, start_time, sa_deadline, rng)
    best_points = points.copy()
    
    # Phase 2: Local refinement
    lr_deadline = start_time + 120.0
    points, best_radius = _local_refinement(points, start_time, lr_deadline, rng)
    if best_radius < _compute_covering_radius(best_points):
        best_points = points.copy()
    
    # Phase 3: Final micro-refinement
    micro_deadline = deadline
    points, best_radius = _local_refinement(points, start_time, micro_deadline, rng)
    if best_radius < _compute_covering_radius(best_points):
        best_points = points.copy()
    
    return best_points
