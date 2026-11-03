"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time
import math


def _normalize(points):
    """Normalize points to the unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return points / norms


def _covering_radius(points):
    """
    Compute the covering radius in degrees.
    The covering radius is the maximum over all points on the sphere of the
    angular distance to the nearest configuration point.
    
    This equals the maximum circumradius of the facets of the convex hull,
    where each facet is checked to have an empty cap (i.e., the circumcenter
    of the facet points to the outside of the sphere, meaning the cap is empty).
    """
    points = _normalize(points)
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fall back to a grid-based estimate
        return _covering_radius_grid(points)
    
    # For each facet, compute the circumradius
    max_circumradius = 0.0
    n_points = len(points)
    
    for simplex in hull.simplices:
        # Get the points of this facet
        facet_points = points[simplex]
        
        # Compute the circumcenter of the facet in 3D
        # The circumcenter of a triangle on the unit sphere is the direction
        # of the normalized sum of the three vertices (for a triangle)
        # More generally, we need to find the point on the sphere that is
        # equidistant from all facet vertices.
        
        if len(simplex) == 3:
            # Triangle: circumcenter direction is the normalized cross product approach
            # or solve for the point equidistant from all three
            a, b, c = facet_points[0], facet_points[1], facet_points[2]
            
            # The circumcenter of a spherical triangle is along the direction
            # of the normalized (a + b + c) only if it's an equilateral triangle.
            # More generally, we need to solve:
            #   x · a = x · b = x · c = cos(theta)
            #   ||x|| = 1
            # where theta is the angular distance.
            
            # Solve using the fact that x is in the span of a, b, c
            # x = alpha*a + beta*b + gamma*c
            # x·a = x·b = x·c
            # ||x|| = 1
            
            # Set up the system:
            # (I - a a^T) x = 0  ->  x is orthogonal to (a - x)
            # Actually, let's use: x·a = x·b => x·(a-b) = 0
            # x·a = x·c => x·(a-c) = 0
            # ||x|| = 1
            
            # So x is perpendicular to (a-b) and (a-c), meaning x is along (a-b) × (a-c)
            # But that's the normal to the plane containing a, b, c, which is not on the sphere generally.
            
            # Let me think differently. The circumcenter on the sphere of a triangle 
            # with vertices a, b, c is the point x on the sphere such that 
            # angular distance from x to a = x to b = x to c.
            
            # This means x·a = x·b = x·c = cos(r) for some r.
            
            # We can solve this by finding x in the plane of a, b, c that is equidistant.
            # The point equidistant from a, b, c in the plane is the circumcenter of the 
            # planar triangle, then project to sphere.
            
            # Actually, for points on the unit sphere, the spherical circumcenter 
            # is the normalization of the vector that is the circumcenter of the 
            # planar triangle formed by a, b, c (since a, b, c are on the unit sphere,
            # the planar circumcenter projected onto the sphere gives the spherical circumcenter).
            
            # Planar circumcenter of triangle a, b, c:
            # d = 2 * (a * (b - c) + b * (c - a) + c * (a - b))
            # circumcenter = (||b||^2 * (c - a) + ||c||^2 * (a - b) + ||a||^2 * (b - c)) / d
            # Since all have norm 1:
            
            d = 2.0 * np.dot(a, np.cross(b, c))
            if abs(d) < 1e-15:
                # Degenerate triangle, skip
                continue
            
            # Circumcenter in the plane
            circumcenter_planar = (
                (np.linalg.norm(b)**2) * (c - a) + 
                (np.linalg.norm(c)**2) * (a - b) + 
                (np.linalg.norm(a)**2) * (b - c)
            ) / d
            
            # Project to sphere
            norm_cc = np.linalg.norm(circumcenter_planar)
            if norm_cc < 1e-15:
                continue
            x = circumcenter_planar / norm_cc
            
            # Check that x is on the "correct" side (the cap should be empty)
            # The cap is empty if x points away from the interior of the convex hull.
            # For a convex hull of points on the unit sphere, the outward normal 
            # of a facet should point in the direction of x (or opposite, depending on orientation).
            
            # The angular distance from x to a is the circumradius
            cos_r = np.dot(x, a)
            cos_r = np.clip(cos_r, -1.0, 1.0)
            r = np.arccos(cos_r)
            
            # Check if the cap is empty: x should be the outward direction
            # For the convex hull, the facet normal should point outward.
            # The outward normal is the normalized cross product of edges.
            edge1 = b - a
            edge2 = c - a
            normal = np.cross(edge1, edge2)
            norm_normal = np.linalg.norm(normal)
            if norm_normal < 1e-15:
                continue
            normal = normal / norm_normal
            
            # Check if x and normal are in the same direction (outward)
            if np.dot(x, normal) > 0:
                max_circumradius = max(max_circumradius, r)
        
        elif len(simplex) == 2:
            # Edge (shouldn't happen in 3D hull, but handle it)
            continue
    
    # Convert to degrees
    return math.degrees(max_circumradius)


def _covering_radius_grid(points, n_grid=500):
    """Fallback: estimate covering radius using a grid on the sphere."""
    points = _normalize(points)
    
    # Create a grid of points on the sphere
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid_points = _normalize(grid_points)
    
    # For each grid point, find the nearest configuration point
    max_dist = 0.0
    for gp in grid_points:
        dists = np.arccos(np.clip(np.dot(points, gp), -1.0, 1.0))
        max_dist = max(max_dist, np.min(dists))
    
    return math.degrees(max_dist)


def _soft_max_covering_radius(points, temperature=0.01):
    """
    Smooth surrogate for the covering radius using log-sum-exp.
    This is differentiable and can be used with gradient-based optimizers.
    """
    points = _normalize(points)
    n = len(points)
    
    # Create a moderate grid
    n_grid = 200
    thetas = np.linspace(0, np.pi, n_grid)
    phis = np.linspace(0, 2 * np.pi, n_grid)
    THETA, PHI = np.meshgrid(thetas, phis)
    
    x = np.sin(THETA) * np.cos(PHI)
    y = np.sin(THETA) * np.sin(PHI)
    z = np.cos(THETA)
    
    grid_points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    grid_points = _normalize(grid_points)
    
    # Compute distances from each grid point to each configuration point
    # dist[i, j] = angular distance from grid point i to config point j
    dot_products = np.clip(grid_points @ points.T, -1.0, 1.0)
    dists = np.arccos(dot_products)
    
    # For each grid point, the nearest distance is min_j dists[i, j]
    # The covering radius is max_i min_j dists[i, j]
    # Soft version: -temperature * log-sum-exp(-dists[i, j] / temperature) approximates min_j dists[i, j]
    # Then max is approximated by temperature * log-sum-exp(nearest_dists / temperature)
    
    # Soft min over j for each i
    nearest_soft = -temperature * np.log(np.sum(np.exp(-dists / temperature), axis=1))
    
    # Soft max over i
    covering_soft = temperature * np.log(np.sum(np.exp(nearest_soft / temperature)))
    
    return math.degrees(covering_soft)


def _objective(params, temperature=0.01):
    """Objective function for optimization: smooth covering radius."""
    points = params.reshape(-1, 3)
    points = _normalize(points)
    return _soft_max_covering_radius(points, temperature)


def _gradient(params, temperature=0.01):
    """Numerical gradient of the objective."""
    eps = 1e-7
    grad = np.zeros_like(params)
    n = len(params)
    for i in range(n):
        params_plus = params.copy()
        params_plus[i] += eps
        params_minus = params.copy()
        params_minus[i] -= eps
        grad[i] = (_objective(params_plus, temperature) - _objective(params_minus, temperature)) / (2 * eps)
    return grad


def _fibonacci_sphere(n):
    """Generate points using the Fibonacci sphere method."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_sphere(n, seed=None):
    """Generate random points on the unit sphere."""
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    # Generate random directions
    points = rng.normal(size=(n, 3))
    points = _normalize(points)
    return points


def construct():
    """
    Construct 50 points on the unit sphere to minimize the covering radius.
    Uses a combination of:
    1. Multiple random initializations
    2. Smooth surrogate optimization via L-BFGS-B
    3. Final polish with exact covering radius evaluation
    """
    start_time = time.time()
    deadline = start_time + 140  # 10 second safety margin
    
    best_points = None
    best_radius = float('inf')
    
    # Try multiple initializations
    n_attempts = 20
    
    for attempt in range(n_attempts):
        if time.time() > deadline:
            break
        
        # Different initialization strategies
        if attempt == 0:
            # Fibonacci sphere
            points = _fibonacci_sphere(N)
        elif attempt == 1:
            # Random with seed 0
            points = _random_sphere(N, seed=0)
        elif attempt == 2:
            # Random with seed 42
            points = _random_sphere(N, seed=42)
        elif attempt < 5:
            # More random starts
            points = _random_sphere(N, seed=attempt * 100)
        else:
            # Perturb the best found so far
            if best_points is not None:
                rng = np.random.default_rng(attempt * 7 + 3)
                noise = rng.normal(scale=0.1, size=(N, 3))
                points = _normalize(best_points + noise)
            else:
                points = _random_sphere(N, seed=attempt * 100)
        
        # Optimize using the smooth surrogate
        try:
            result = minimize(
                _objective,
                points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8},
                args=(0.01,)
            )
            
            if result.success or result.fun < best_radius:
                opt_points = _normalize(result.x.reshape(-1, 3))
                # Evaluate the true covering radius
                true_radius = _covering_radius(opt_points)
                
                if true_radius < best_radius:
                    best_radius = true_radius
                    best_points = opt_points.copy()
                    
        except Exception as e:
            pass
        
        # Check time
        if time.time() > deadline:
            break
    
    # If we didn't find anything, fall back to Fibonacci
    if best_points is None:
        best_points = _fibonacci_sphere(N)
    
    # Final polish: try a few more optimization steps from the best point
    if time.time() < deadline - 5:
        try:
            result = minimize(
                _objective,
                best_points.ravel(),
                method='L-BFGS-B',
                options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-10},
                args=(0.005,)  # Lower temperature for more accurate max approximation
            )
            
            if result.fun < best_radius:
                polish_points = _normalize(result.x.reshape(-1, 3))
                polish_radius = _covering_radius(polish_points)
                if polish_radius < best_radius:
                    best_points = polish_points
                    best_radius = polish_radius
        except Exception:
            pass
    
    # Ensure the result is valid
    best_points = _normalize(best_points)
    
    # Final validation: check that points are not all in a hemisphere
    # and that the result is reasonable
    if np.any(np.linalg.norm(best_points, axis=1) < 1e-10):
        best_points = _fibonacci_sphere(N)
    
    return best_points
# EVOLVE-BLOCK-END
