"""Rewritten program for Heilbronn triangles task."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11
# Equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])


def _area(a, b, c):
    """Area of triangle formed by points a, b, c."""
    return 0.5 * abs(np.cross(b - a, c - a))


def _all_triangle_areas(points):
    """Compute areas of all C(n,3) triangles formed by the points."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas.append(_area(points[i], points[j], points[k]))
    return np.array(areas)


def _objective(points_flat):
    """
    Objective function: minimize the negative log-sum-exp of areas.
    This is a smooth approximation to maximizing the minimum area.
    """
    points = points_flat.reshape(-1, 2)
    areas = _all_triangle_areas(points)
    
    # Clip areas to avoid log(0)
    areas_clipped = np.clip(areas, 1e-12, None)
    
    # Use a large temperature for sharper max-min approximation
    temp = 0.01
    # Log-sum-exp trick for numerical stability
    max_a = np.max(areas_clipped)
    log_sum_exp = np.log(np.sum(np.exp((areas_clipped - max_a) / temp))) + max_a / temp
    
    # We want to maximize the min area, so minimize -log_sum_exp
    return -log_sum_exp


def _gradient(points_flat):
    """
    Numerical gradient via finite differences.
    """
    eps = 1e-6
    n = len(points_flat)
    grad = np.zeros(n)
    for i in range(n):
        orig = points_flat[i]
        points_flat[i] = orig + eps
        f_plus = _objective(points_flat)
        points_flat[i] = orig - eps
        f_minus = _objective(points_flat)
        points_flat[i] = orig
        grad[i] = (f_plus - f_minus) / (2 * eps)
    return grad


def _project_to_triangle(points):
    """
    Project points that are outside the triangle back inside.
    Uses barycentric coordinates for the equilateral triangle.
    """
    h = np.sqrt(3) / 2
    for i in range(len(points)):
        x, y = points[i]
        # Check if point is inside the triangle
        # Barycentric coordinates for triangle (0,0), (1,0), (0.5, h)
        # lambda0 = 1 - x - y/h
        # lambda1 = x
        # lambda2 = y/h
        lam0 = 1 - x - y / h
        lam1 = x
        lam2 = y / h
        
        if lam0 < -1e-9 or lam1 < -1e-9 or lam2 < -1e-9:
            # Point is outside, project to nearest edge
            # Simple approach: clamp barycentric coords to [0,1] and normalize
            lams = np.clip([lam0, lam1, lam2], 0, None)
            s = np.sum(lams)
            if s > 0:
                lams /= s
            else:
                lams = np.array([1/3, 1/3, 1/3])
            # Convert back to Cartesian
            new_x = lams[1] * 1.0 + lams[2] * 0.5
            new_y = lams[2] * h
            points[i] = [new_x, new_y]
    return points


def _initial_points_grid():
    """
    Generate initial points on a structured grid inside the triangle.
    Uses a hexagonal-like packing for better coverage.
    """
    points = []
    # Center point
    h = np.sqrt(3) / 2
    centroid = np.array([0.5, h / 3])
    points.append(centroid.copy())
    
    # Add points in rings around the centroid
    # Ring 1: 6 points
    for angle in np.linspace(0, 2 * np.pi, 7)[:-1]:
        r = 0.25
        x = 0.5 + r * np.cos(angle)
        y = h / 3 + r * np.sin(angle) * (h / (h / 3))  # Scale y to account for triangle shape
        # Actually, let's just use Cartesian and project
        p = np.array([x, h / 3 + r * np.sin(angle)])
        points.append(p)
    
    # Add more points to reach 11
    # Use a simple spiral or grid fill
    while len(points) < N:
        # Try adding points at various positions
        for r in np.linspace(0.1, 0.4, 10):
            for angle in np.linspace(0, 2 * np.pi, 12)[:-1]:
                x = 0.5 + r * np.cos(angle)
                y = h / 3 + r * np.sin(angle)
                p = np.array([x, y])
                # Check if inside triangle
                lam0 = 1 - x - y / h
                lam1 = x
                lam2 = y / h
                if lam0 > 0.01 and lam1 > 0.01 and lam2 > 0.01:
                    # Check distance to existing points
                    if all(np.linalg.norm(p - q) > 0.05 for q in points):
                        points.append(p)
                        break
            if len(points) >= N:
                break
    
    # Ensure we have exactly N points
    while len(points) < N:
        # Add random point inside triangle
        u, v = np.random.random(2)
        if u + v < 1:
            x = v * 0.5
            y = v * h
            p = np.array([x, y])
            points.append(p)
        else:
            u, v = 1 - u, 1 - v
            x = v * 0.5
            y = v * h
            p = np.array([x, y])
            points.append(p)
    
    return np.array(points[:N])


def construct():
    """
    Construct 11 points in the unit equilateral triangle to maximize
    the minimum area of any triangle formed by three points.
    """
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    np.random.seed(42)
    
    best_points = None
    best_score = -np.inf
    
    # Multiple restarts with different initializations
    for restart in range(20):
        if time.time() > deadline - 10:
            break
        
        if restart == 0:
            points = _initial_points_grid()
        else:
            # Random initialization with some structure
            points = np.zeros((N, 2))
            h = np.sqrt(3) / 2
            for i in range(N):
                u, v = np.random.random(2)
                if u + v < 1:
                    points[i] = [v * 0.5, v * h]
                else:
                    u, v = 1 - u, 1 - v
                    points[i] = [v * 0.5, v * h]
        
        points_flat = points.flatten()
        
        # Run L-BFGS-B optimization
        try:
            result = minimize(
                _objective,
                points_flat,
                method='L-BFGS-B',
                jac=_gradient,
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if result.success or result.fun < best_score:
                opt_points = result.x.reshape(-1, 2)
                opt_points = _project_to_triangle(opt_points)
                
                # Evaluate the actual max-min area
                areas = _all_triangle_areas(opt_points)
                min_area = np.min(areas)
                
                if min_area > best_score:
                    best_score = min_area
                    best_points = opt_points.copy()
                    
        except Exception as e:
            pass
    
    # If no valid points found, return a default
    if best_points is None:
        return _initial_points_grid()
    
    return best_points
