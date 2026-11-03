"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

def _circumradius_of_facet(v1, v2, v3):
    """Compute the circumradius (in radians) of the spherical triangle with vertices v1, v2, v3."""
    # Gram matrix
    g11 = np.dot(v1, v1)
    g12 = np.dot(v1, v2)
    g13 = np.dot(v1, v3)
    g22 = np.dot(v2, v2)
    g23 = np.dot(v2, v3)
    g33 = np.dot(v3, v3)
    G = np.array([[g11, g12, g13],
                  [g12, g22, g23],
                  [g13, g23, g33]])
    ones = np.ones(3)
    try:
        c = np.linalg.solve(G, ones)
    except np.linalg.LinAlgError:
        return np.pi  # degenerate, return large value
    norm_c = np.linalg.norm(c)
    if norm_c < 1e-12:
        return np.pi
    # The unnormalized circumcenter c has c·vi = 1 for all i
    # Normalized circumcenter is c/norm_c, and circumradius = arccos(1/norm_c)
    cos_r = 1.0 / norm_c
    cos_r = np.clip(cos_r, -1.0, 1.0)
    r = np.arccos(cos_r)
    return r

def _compute_covering_radius(points):
    """Compute the covering radius in radians for a set of points on the unit sphere."""
    points = np.asarray(points)
    # Normalize to ensure they're on the sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi
    
    max_r = 0.0
    for simplex in hull.simplices:
        v1 = points[simplex[0]]
        v2 = points[simplex[1]]
        v3 = points[simplex[2]]
        r = _circumradius_of_facet(v1, v2, v3)
        if r > max_r:
            max_r = r
    return max_r

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.clip(1.0 - z * z, 0, 1.0))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _points_to_angles(points):
    """Convert 3D points on sphere to (theta, phi) angles."""
    points = np.asarray(points)
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    theta = np.arctan2(y, x)
    phi = np.arccos(np.clip(z, -1.0, 1.0))
    return np.column_stack([theta, phi])

def _angles_to_points(angles):
    """Convert (theta, phi) angles to 3D points on sphere."""
    theta = angles[:, 0]
    phi = angles[:, 1]
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    return np.column_stack([x, y, z])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    # Initial configuration: Fibonacci spiral
    points = _fibonacci_sphere(N)
    angles = _points_to_angles(points)
    
    best_points = points.copy()
    best_radius = _compute_covering_radius(points)
    
    # Define objective function
    def objective(angle_params):
        pts = _angles_to_points(angle_params.reshape(-1, 2))
        return _compute_covering_radius(pts)
    
    # Try optimization with L-BFGS-B
    try:
        result = minimize(
            objective,
            angles.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 200, 'maxfun': 20000}
        )
        if time.time() < deadline:
            new_points = _angles_to_points(result.x.reshape(-1, 2))
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_points = new_points
                best_radius = new_radius
    except Exception:
        pass
    
    # If we have time, try a few more restarts with perturbed initial conditions
    for restart in range(10):
        if time.time() > deadline - 5.0:
            break
        try:
            # Perturb the best configuration
            noise = np.random.randn(N, 2) * 0.1
            perturbed_angles = best_angles = _points_to_angles(best_points) + noise
            perturbed_angles[:, 0] = np.mod(perturbed_angles[:, 0], 2 * np.pi)
            perturbed_angles[:, 1] = np.clip(perturbed_angles[:, 1], 0.01, np.pi - 0.01)
            
            result = minimize(
                objective,
                perturbed_angles.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxfun': 10000}
            )
            new_points = _angles_to_points(result.x.reshape(-1, 2))
            new_radius = _compute_covering_radius(new_points)
            if new_radius < best_radius:
                best_points = new_points
                best_radius = new_radius
        except Exception:
            continue
    
    # Final normalization
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    best_points = best_points / norms
    
    return best_points
# EVOLVE-BLOCK-END
