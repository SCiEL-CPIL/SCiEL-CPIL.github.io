"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import math

def _covering_radius(points):
    """Compute the covering radius in radians for points on the unit sphere."""
    points = np.asarray(points)
    if points.shape[0] < 3:
        return np.pi
    
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1
    points = points / norms
    
    # Compute convex hull
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.pi  # Fallback: worst case
    
    # For each facet, compute the circumradius
    max_radius = 0.0
    for simplex in hull.simplices:
        face_points = points[simplex]
        # Compute the face normal (center of the cap)
        face_center = face_points.sum(axis=0)
        face_center_norm = np.linalg.norm(face_center)
        if face_center_norm < 1e-10:
            continue
        face_center = face_center / face_center_norm
        # Compute angular distances from face center to each vertex
        for v in face_points:
            dot = np.clip(np.dot(face_center, v), -1.0, 1.0)
            angle = math.acos(dot)
            if angle > max_radius:
                max_radius = angle
    
    return max_radius

def _objective(params):
    """Objective function: covering radius in radians."""
    thetas = params[0::2]
    phis = params[1::2]
    
    # Clamp thetas to avoid numerical issues at poles
    thetas = np.clip(thetas, 1e-8, np.pi - 1e-8)
    
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    points = np.column_stack([x, y, z])
    
    return _covering_radius(points)

def _params_to_points(params):
    """Convert parameter vector to points on unit sphere."""
    thetas = params[0::2]
    phis = params[1::2]
    
    thetas = np.clip(thetas, 1e-8, np.pi - 1e-8)
    
    x = np.sin(thetas) * np.cos(phis)
    y = np.sin(thetas) * np.sin(phis)
    z = np.cos(thetas)
    return np.column_stack([x, y, z])

def _points_to_params(points):
    """Convert points on unit sphere to parameter vector (theta, phi)."""
    points = np.asarray(points)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1
    points = points / norms
    
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    thetas = np.arccos(np.clip(z, -1.0 + 1e-10, 1.0 - 1e-10))
    phis = np.arctan2(y, x)
    
    params = np.empty(2 * N)
    params[0::2] = thetas
    params[1::2] = phis
    return params

def _initial_config():
    """Generate initial Fibonacci spiral configuration."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return points

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Leave 10s margin
    
    # Start with Fibonacci spiral
    points = _initial_config()
    best_points = points.copy()
    best_radius = _covering_radius(points)
    best_params = _points_to_params(points)
    
    # Optimization loop with restarts
    max_attempts = 8
    for attempt in range(max_attempts):
        if time.time() > deadline - 25.0:
            break
        
        # Use L-BFGS-B for optimization from current best
        try:
            result = minimize(
                _objective,
                best_params,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-10}
            )
            
            new_points = _params_to_points(result.x)
            new_radius = _covering_radius(new_points)
            
            if new_radius < best_radius:
                best_radius = new_radius
                best_points = new_points.copy()
                best_params = result.x.copy()
        except Exception:
            pass
        
        # If not the last attempt, perturb to try new basin
        if attempt < max_attempts - 1 and time.time() < deadline - 25.0:
            # Scale of perturbation decreases with attempts
            scale = 0.3 / (attempt + 1)
            noise = np.random.normal(0, scale, size=best_params.shape)
            perturbed_params = best_params + noise
            # Clamp theta to valid range [0, pi]
            perturbed_params[0::2] = np.clip(perturbed_params[0::2], 1e-6, np.pi - 1e-6)
            best_params = perturbed_params
    
    # Ensure points are on unit sphere and nonzero
    norms = np.linalg.norm(best_points, axis=1, keepdims=True)
    norms[norms == 0] = 1
    best_points = best_points / norms
    
    return best_points

# EVOLVE-BLOCK-END
