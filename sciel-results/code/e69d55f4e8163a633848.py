"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _covering_radius(points):
    """Compute the covering radius (in radians) for a set of points on the unit sphere.
    
    The covering radius is the maximum angular distance from any point on the sphere
    to the nearest chosen point. This is equivalent to half the largest edge length
    of the Delaunay triangulation on the sphere, or more practically, we can compute
    it as the maximum over the vertices of the Voronoi cells of the distance to the
    nearest point. A practical approximation is to sample many points and find the
    max min-distance, but a better approach is to use the fact that the covering radius
    equals the maximum circumradius of the spherical Delaunay triangles.
    
    For efficiency, we'll compute it by checking all pairs and using the fact that
    the covering radius is at least half the maximum nearest-neighbor distance, but
    for a tighter bound we can use the convex hull approach or sample-based.
    
    Actually, the most accurate way without heavy computation: the covering radius is
    the maximum over all points x on the sphere of min_i ||x - p_i|| (angular).
    This maximum occurs at a vertex of the Voronoi diagram on the sphere, i.e., at
    a point equidistant from at least 3 (or 2 on the boundary) of the chosen points.
    
    For a practical optimization, we'll use a sampling-based estimate that is fast:
    sample many random points and compute the max of the min angular distance.
    """
    # Use a denser sampling for accuracy
    n_samples = 20000
    # Random points on sphere
    theta = np.random.uniform(0, 2 * np.pi, n_samples)
    phi = np.arccos(np.random.uniform(-1, 1, n_samples))
    samples = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    
    # Compute min angular distance from each sample to the nearest point
    # Angular distance = arccos(dot)
    dots = np.dot(samples, points.T)  # (n_samples, N)
    dots = np.clip(dots, -1.0, 1.0)
    min_dots = np.max(dots, axis=1)  # max dot = min angular distance
    angles = np.arccos(min_dots)
    return np.max(angles)


def _objective(params):
    """Objective function: covering radius in radians."""
    points = _params_to_points(params)
    # Ensure points are on unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    return _covering_radius(points)


def _params_to_points(params):
    """Convert 3N parameters to N points on the unit sphere."""
    params = params.reshape(N, 3)
    norms = np.linalg.norm(params, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return params / norms


def _points_to_params(points):
    """Convert N points on the unit sphere to 3N parameters."""
    return points.flatten()


def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    np.random.seed(42)
    
    # Start with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize
    points0 = points0 / np.linalg.norm(points0, axis=1, keepdims=True)
    
    params0 = _points_to_params(points0)
    
    # Use L-BFGS-B for optimization
    # We'll do a few restarts with different initializations
    best_params = params0
    best_val = _objective(params0)
    
    # Try a few random restarts
    for restart in range(3):
        # Random initial points
        if restart == 0:
            # Use Fibonacci spiral (already done)
            continue
        elif restart == 1:
            # Slightly perturbed Fibonacci
            points_init = points0 + np.random.normal(0, 0.1, (N, 3))
            points_init = points_init / np.linalg.norm(points_init, axis=1, keepdims=True)
        else:
            # Random points
            points_init = np.random.normal(0, 1, (N, 3))
            points_init = points_init / np.linalg.norm(points_init, axis=1, keepdims=True)
        
        params_init = _points_to_params(points_init)
        
        res = minimize(
            _objective,
            params_init,
            method='L-BFGS-B',
            options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        
        if res.fun < best_val:
            best_val = res.fun
            best_params = res.x
    
    # Final normalization
    points = _params_to_points(best_params)
    return points
# EVOLVE-BLOCK-END
