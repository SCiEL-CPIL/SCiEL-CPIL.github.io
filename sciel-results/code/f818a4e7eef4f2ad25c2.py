"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools


def _angular_distance(p, q):
    """Angular distance in radians between two unit vectors."""
    dot = np.clip(np.dot(p, q), -1.0, 1.0)
    return np.arccos(dot)


def _covering_radius(points):
    """
    Compute the covering radius (in radians) of a set of points on the unit sphere.
    This is the maximum angular distance from any point on the sphere to its nearest
    configuration point. We approximate this by checking the vertices of the convex hull
    and the circumcenters of the facets.
    """
    points = np.asarray(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Build convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use grid-based approximation
        return _covering_radius_grid(points)
    
    # The covering radius is determined by the largest empty spherical cap.
    # For a set of points on the sphere, the farthest point from the set lies at
    # the circumcenter of some facet of the convex hull (or at a vertex if all
    # points are in a hemisphere).
    
    # Get hull facets
    simplices = hull.simplices
    
    max_angle = 0.0
    
    for simplex in simplices:
        # For each facet (triangle or higher), find the point on the sphere
        # farthest from all configuration points. This is the circumcenter of
        # the facet's intersection with the sphere, projected back.
        
        # Get the points of this facet
        facet_points = points[simplex]
        
        # Compute the circumcenter of these points on the sphere.
        # For a triangle, the circumcenter is the intersection of the sphere
        # with the plane through the three points, on the side away from the origin
        # (or the side that gives the farthest point).
        
        if len(simplex) == 3:
            # Triangle facet
            a, b, c = facet_points[0], facet_points[1], facet_points[2]
            
            # Normal to the plane through a, b, c
            normal = np.cross(b - a, c - a)
            norm = np.linalg.norm(normal)
            if norm < 1e-10:
                continue
            normal = normal / norm
            
            # The circumcenter of the spherical triangle is along the normal
            # from the origin. We need to find which direction (normal or -normal)
            # gives the point farthest from all points.
            
            # Compute distances from both candidate centers to all points
            for center in [normal, -normal]:
                # Make sure center is on the unit sphere
                center = center / np.linalg.norm(center)
                
                # Compute angular distance from center to all points
                dots = np.clip(points @ center, -1.0, 1.0)
                angles = np.arccos(dots)
                min_angle = np.min(angles)
                
                if min_angle > max_angle:
                    max_angle = min_angle
    
    return max_angle


def _covering_radius_grid(points):
    """Fallback grid-based approximation of covering radius."""
    points = np.asarray(points)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Create a grid on the sphere
    n_theta = 60
    n_phi = 120
    theta = np.linspace(0, np.pi, n_theta)
    phi = np.linspace(0, 2 * np.pi, n_phi)
    
    max_angle = 0.0
    for t in theta:
        for p in phi:
            x = np.sin(t) * np.cos(p)
            y = np.sin(t) * np.sin(p)
            z = np.cos(t)
            y_pt = np.array([x, y, z])
            
            dots = np.clip(points @ y_pt, -1.0, 1.0)
            angles = np.arccos(dots)
            min_angle = np.min(angles)
            if min_angle > max_angle:
                max_angle = min_angle
    
    return max_angle


def _objective(x, N):
    """Objective function: covering radius in radians."""
    points = x.reshape(-1, 3)
    # Normalize
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    points = points / norms
    return _covering_radius(points)


def _gradient_descent_step(points, learning_rate=0.01, n_iter=100):
    """
    Simple gradient descent step to minimize covering radius.
    Uses subgradient estimation via finite differences.
    """
    N = len(points)
    
    for _ in range(n_iter):
        # Estimate gradient for each point
        grads = np.zeros_like(points)
        eps = 1e-5
        base_obj = _covering_radius(points)
        
        for i in range(N):
            for d in range(3):
                orig = points[i, d]
                points[i, d] = orig + eps
                obj_plus = _covering_radius(points)
                points[i, d] = orig - eps
                obj_minus = _covering_radius(points)
                points[i, d] = orig
                
                grads[i, d] = (obj_plus - obj_minus) / (2 * eps)
        
        # Update
        points -= learning_rate * grads
        # Renormalize
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        points = points / norms
    
    return points


def _refine_with_optimization(initial_points, n_restarts=3, max_iter=200):
    """
    Refine the initial configuration using scipy's minimize with multiple restarts.
    """
    best_points = initial_points.copy()
    best_obj = _covering_radius(initial_points)
    
    for restart in range(n_restarts):
        # Perturb the initial points slightly
        if restart == 0:
            x0 = initial_points.flatten()
        else:
            x0 = (initial_points + np.random.randn(*initial_points.shape) * 0.1).flatten()
        
        # Normalize
        pts = x0.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        pts = pts / norms
        x0 = pts.flatten()
        
        result = minimize(
            _objective,
            x0,
            args=(N,),
            method='L-BFGS-B',
            options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-8}
        )
        
        if result.success or result.fun < best_obj:
            pts = result.x.reshape(-1, 3)
            norms = np.linalg.norm(pts, axis=1, keepdims=True)
            norms[norms < 1e-10] = 1.0
            pts = pts / norms
            best_obj = result.fun
            best_points = pts
    
    return best_points


def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    Uses a multi-stage approach:
    1. Start with a good initial configuration (Fibonacci spiral + perturbation)
    2. Refine using gradient-based optimization
    """
    np.random.seed(42)
    
    # Stage 1: Generate multiple initial candidates
    candidates = []
    
    # Candidate 1: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    candidates.append(fib_points)
    
    # Candidate 2: Fibonacci spiral with different offset
    i2 = np.arange(N) + 0.25
    z2 = 1.0 - 2.0 * i2 / N
    phi2 = np.pi * (1.0 + np.sqrt(5.0)) * i2
    r2 = np.sqrt(1.0 - z2 * z2)
    fib_points2 = np.column_stack([r2 * np.cos(phi2), r2 * np.sin(phi2), z2])
    candidates.append(fib_points2)
    
    # Candidate 3: Random points (well-distributed via rejection)
    random_points = []
    while len(random_points) < N:
        p = np.random.randn(3)
        p = p / np.linalg.norm(p)
        random_points.append(p)
    candidates.append(np.array(random_points))
    
    # Candidate 4: Points from a quasicrystal-like structure
    # Use golden ratio for 3D distribution
    gr = (1 + np.sqrt(5)) / 2
    angles = np.arange(N) * 2 * np.pi / gr
    z_vals = np.linspace(-1, 1, N)
    r_vals = np.sqrt(1 - z_vals**2)
    qc_points = np.column_stack([
        r_vals * np.cos(angles),
        r_vals * np.sin(angles),
        z_vals
    ])
    candidates.append(qc_points)
    
    # Evaluate all candidates and pick the best
    best_candidate = None
    best_obj = np.inf
    for cand in candidates:
        obj = _covering_radius(cand)
        if obj < best_obj:
            best_obj = obj
            best_candidate = cand
    
    # Stage 2: Refine the best candidate
    refined = _refine_with_optimization(best_candidate, n_restarts=2, max_iter=150)
    
    # Final normalization
    norms = np.linalg.norm(refined, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    refined = refined / norms
    
    return refined
# EVOLVE-BLOCK-END
