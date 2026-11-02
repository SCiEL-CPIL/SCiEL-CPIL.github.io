"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _pairwise_angular_distances(points):
    """Return NxN matrix of angular distances (radians) between points."""
    # points: (N, 3) unit vectors
    cos_d = np.clip(points @ points.T, -1.0, 1.0)
    return np.arccos(cos_d)


def _covering_radius(points):
    """Compute the covering radius (max angular distance to nearest point) for points on unit sphere.
    
    The covering radius is the maximum over all directions y on the sphere of min_i angle(y, p_i).
    This equals half the maximum gap in the 1D angular sense along the sphere, but more precisely
    it's the largest circumradius of the spherical Voronoi cells.
    
    For a set of points on the sphere, the covering radius is the maximum over all facets of the
    convex hull of the circumradius of that facet's spherical polygon. However, a simpler and 
    correct approach: the covering radius is the maximum, over all pairs of points, of half the
    angular distance IF the sphere is covered such that the worst case is between two points.
    
    Actually, the covering radius for a set of points on S^2 is the maximum angular distance from
    any point on the sphere to its nearest neighbor in the set. This can be computed as:
    - Find the Delaunay triangulation on the sphere (or use the convex hull facets)
    - For each spherical triangle (facet), compute its circumradius
    - The covering radius is the maximum of these circumradii
    
    A practical approximation: the covering radius is at least half the maximum pairwise angular
    distance, but this is only a lower bound. For a better estimate, we can use the fact that
    the worst point is at the center of a spherical Voronoi cell, which is the circumcenter of
    a face of the convex hull.
    
    For optimization purposes, we'll use a simpler but effective metric: the maximum over all
    points on a fine grid of the distance to the nearest configuration point. But this is expensive.
    
    Alternative: use the fact that the covering radius equals the maximum, over all edges of the
    convex hull's spherical Voronoi diagram... 
    
    Let's use a practical approach: sample many points on the sphere and compute the max distance
    to nearest configuration point. We'll use a reasonably fine grid.
    """
    # Generate a fine grid of test points on the sphere
    # Use Fibonacci spiral for uniform distribution
    n_test = 5000
    i = np.arange(n_test) + 0.5
    z = 1.0 - 2.0 * i / n_test
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    test_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Compute distances from each test point to each configuration point
    cos_d = np.clip(test_points @ points.T, -1.0, 1.0)
    min_cos = np.max(cos_d, axis=1)  # max cosine = min angle
    min_angles = np.arccos(np.clip(min_cos, -1.0, 1.0))
    
    return np.max(min_angles)


def _objective(params):
    """Objective function: covering radius of the configuration defined by params."""
    # params: 2N parameters (theta, phi for each point)
    thetas = params[0::2]
    phis = params[1::2]
    
    points = np.column_stack([
        np.sin(thetas) * np.cos(phis),
        np.sin(thetas) * np.sin(phis),
        np.cos(thetas)
    ])
    
    # Normalize to ensure unit vectors
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return _covering_radius(points)


def construct():
    # Start with a Fibonacci spiral configuration
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    points0 = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Convert to spherical coordinates for optimization
    thetas0 = np.arccos(np.clip(points0[:, 2], -1.0, 1.0))
    phis0 = np.arctan2(points0[:, 1], points0[:, 0])
    
    params0 = np.zeros(2 * N)
    params0[0::2] = thetas0
    params0[1::2] = phis0
    
    # Use Nelder-Mead optimization
    result = minimize(
        _objective,
        params0,
        method='Nelder-Mead',
        options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'disp': False}
    )
    
    # Extract optimized points
    thetas = result.x[0::2]
    phis = result.x[1::2]
    
    points = np.column_stack([
        np.sin(thetas) * np.cos(phis),
        np.sin(thetas) * np.sin(phis),
        np.cos(thetas)
    ])
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    return points
# EVOLVE-BLOCK-END
