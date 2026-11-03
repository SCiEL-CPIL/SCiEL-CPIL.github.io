# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

N = 50

def _normalize(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms

def _fibonacci_sphere(n):
    """Generate n points on the unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _create_test_points(n=5000):
    """Create a fixed set of test points to approximate covering radius."""
    return _fibonacci_sphere(n)

def _covering_radius(points, test_points):
    """
    Calculate the approximate covering radius using a set of test points.
    """
    points = _normalize(points)
    dots = points @ test_points.T
    dots = np.clip(dots, -1.0, 1.0)
    max_dots = np.max(dots, axis=0)
    min_dot = np.min(max_dots)
    return np.arccos(np.clip(min_dot, -1.0, 1.0))

def _sa_optimize_covering(points, test_points, T_start=0.1, T_end=1e-5, n_iter=5000):
    """
    Simulated Annealing to minimize the covering radius.
    """
    points = _normalize(points).copy()
    current_rad = _covering_radius(points, test_points)
    best_points = points.copy()
    best_rad = current_rad
    
    cooling = (T_end / T_start) ** (1.0 / n_iter)
    
    for i in range(n_iter):
        T = T_start * (cooling ** i)
        
        # Select a random point to move
        idx = np.random.randint(N)
        original_point = points[idx].copy()
        
        # Generate a new position
        # Step size scaled by temperature
        step_size = 0.2 * T / T_start
        noise = np.random.normal(0, step_size, 3)
        new_point = original_point + noise
        new_point = new_point / np.linalg.norm(new_point)
        
        points[idx] = new_point
        new_rad = _covering_radius(points, test_points)
        delta = new_rad - current_rad
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            current_rad = new_rad
            if current_rad < best_rad:
                best_rad = current_rad
                best_points = points.copy()
        else:
            points[idx] = original_point
            
    return best_points

def construct():
    """Construct 50 points on the unit sphere minimizing the covering radius."""
    # 1. Initialize with Fibonacci Spiral
    x0 = _fibonacci_sphere(N)
    
    # 2. Create test points
    test_points = _create_test_points(5000)
    
    # 3. Simulated Annealing on Covering Radius
    points_sa = _sa_optimize_covering(x0, test_points, T_start=0.5, T_end=1e-5, n_iter=5000)
    
    # 4. Local Polish with Powell's Method
    def objective(params, tp):
        pts = params.reshape(N, 3)
        return _covering_radius(pts, tp)
        
    x0_flat = points_sa.flatten()
    result = minimize(
        objective, 
        x0_flat, 
        args=(test_points,),
        method='Powell',
        options={'maxiter': 100, 'xtol': 1e-8, 'ftol': 1e-8}
    )
    
    best_points = result.x.reshape(N, 3)
    best_points = _normalize(best_points)
    
    return best_points
# EVOLVE-BLOCK-END
