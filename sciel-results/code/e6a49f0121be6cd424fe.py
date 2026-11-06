"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _normalize(v):
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n

def _covering_radius(points):
    """Estimate covering radius by dense sampling of the sphere."""
    # Use Fibonacci sphere for dense sampling
    n_samples = 20000
    i = np.arange(n_samples) + 0.5
    z = 1.0 - 2.0 * i / n_samples
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute angular distance from each sample to nearest point
    # cos(theta) = dot product
    dots = np.clip(samples @ points.T, -1.0, 1.0)
    min_cos = np.max(dots, axis=1)
    max_theta = np.arccos(min_cos)
    return np.max(max_theta) * 180.0 / np.pi

def _hull_covering_radius(points):
    """Compute exact covering radius via convex hull facets."""
    try:
        hull = ConvexHull(points)
        max_radius = 0.0
        for simplex in hull.simplices:
            facet_points = points[simplex]
            # Find circumcenter of facet
            # For a triangle, circumcenter is intersection of perpendicular bisectors
            if len(facet_points) == 3:
                a, b, c = facet_points
                # Solve for circumcenter in 3D
                # |x-a|^2 = |x-b|^2 = |x-c|^2
                # 2*x.(b-a) = |b|^2 - |a|^2
                # 2*x.(c-a) = |c|^2 - |a|^2
                A = np.array([2*(b-a), 2*(c-a)])
                rhs = np.array([np.dot(b,b)-np.dot(a,a), np.dot(c,c)-np.dot(a,a)])
                try:
                    x = np.linalg.solve(A, rhs)
                    # Project to sphere
                    x_norm = np.linalg.norm(x)
                    if x_norm > 0:
                        x = x / x_norm
                    # Angular distance from facet center to nearest point
                    dots = np.clip(x @ points.T, -1.0, 1.0)
                    min_dist = np.arccos(np.max(dots))
                    max_radius = max(max_radius, min_dist)
                except:
                    pass
            elif len(facet_points) >= 3:
                # Use centroid approximation
                centroid = np.mean(facet_points, axis=0)
                centroid = _normalize(centroid)
                dots = np.clip(centroid @ points.T, -1.0, 1.0)
                min_dist = np.arccos(np.max(dots))
                max_radius = max(max_radius, min_dist)
        return max_radius * 180.0 / np.pi
    except:
        return 180.0

def _random_point_on_sphere():
    v = np.random.randn(3)
    return v / np.linalg.norm(v)

def _perturb_point(point, angle):
    """Perturb a point by a random angle in a random direction."""
    # Generate random rotation axis
    axis = np.random.randn(3)
    axis = axis / np.linalg.norm(axis)
    # Rodrigues' rotation formula
    c = np.cos(angle)
    s = np.sin(angle)
    k = axis
    kx = np.array([0, k[2], -k[1]])
    ky = np.array([-k[2], 0, k[0]])
    kz = np.array([k[1], -k[0], 0])
    # Rotation matrix
    R = c * np.eye(3) + s * np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ]) + (1-c) * np.outer(k, k)
    return _normalize(R @ point)

def _local_optimize(points, max_iter=500):
    """Local optimization using coordinate descent with random steps."""
    best = points.copy()
    best_cr = _covering_radius(best)
    for it in range(max_iter):
        improved = False
        # Try to improve each point
        for i in range(N):
            best_i = best[i].copy()
            best_cr_i = best_cr
            for _ in range(20):
                # Random perturbation
                new_point = _perturb_point(best_i, np.random.uniform(0.01, 0.1))
                candidate = best.copy()
                candidate[i] = new_point
                cr = _covering_radius(candidate)
                if cr < best_cr_i - 1e-8:
                    best_i = new_point
                    best_cr_i = cr
                    improved = True
            best[i] = best_i
            best_cr = _covering_radius(best)
        if not improved:
            break
    return best, best_cr

def construct():
    np.random.seed(42)
    
    # Start with a better initial configuration: use icosahedron-based subdivision
    # Create initial points using a combination of Fibonacci and random
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Simulated annealing
    T = 0.5
    T_min = 0.001
    cooling = 0.995
    current_cr = _covering_radius(points)
    best_points = points.copy()
    best_cr = current_cr
    
    max_iters = 3000
    for it in range(max_iters):
        # Random perturbation of 1-3 points
        n_perturb = np.random.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        candidate = points.copy()
        indices = np.random.choice(N, n_perturb, replace=False)
        for idx in indices:
            angle = np.random.uniform(0.02, 0.15) * T
            candidate[idx] = _perturb_point(candidate[idx], angle)
        
        new_cr = _covering_radius(candidate)
        
        # Accept if better or with probability exp(-delta/T)
        if new_cr < current_cr:
            points = candidate
            current_cr = new_cr
            if new_cr < best_cr:
                best_cr = new_cr
                best_points = points.copy()
        else:
            delta = new_cr - current_cr
            if np.random.random() < np.exp(-delta / max(T, 1e-10)):
                points = candidate
                current_cr = new_cr
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final local optimization
    optimized, opt_cr = _local_optimize(best_points, max_iter=200)
    if opt_cr < best_cr:
        best_points = optimized
        best_cr = opt_cr
    
    return best_points
# EVOLVE-BLOCK-END
