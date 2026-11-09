# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize, differential_evolution

def calculate_score(points):
    """Calculate the metric R = (d_min / d_max)^2"""
    if len(points) < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio

def objective(x):
    """
    Objective function for minimization.
    We want to maximize (d_min / d_max)^2, so we minimize its negative.
    x is a flat array of shape (32,).
    """
    points = x.reshape(-1, 2)
    score = calculate_score(points)
    return -score

def hexagonal_init(n, scale=1.0, rotation=0.0):
    """Generate a hexagonal grid initialization."""
    points = []
    # 4x4 grid is 16 points
    for i in range(4):
        for j in range(4):
            x = j * 1.0 + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            points.append([x, y])
    pts = np.array(points)
    # Center
    center = pts.mean(axis=0)
    pts = pts - center
    # Rotate
    theta = rotation
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])
    pts = pts @ R.T
    # Normalize to roughly fit in [-1, 1]
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * 0.9
    return pts

def random_init(n, seed=42):
    """Generate a random initialization within unit circle."""
    rng = np.random.RandomState(seed)
    # Random points in unit circle
    r = rng.rand(n) ** 0.5
    theta = rng.rand(n) * 2 * np.pi
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack((x, y))

def square_init(n, scale=1.0):
    """Generate a square grid initialization."""
    points = []
    for i in range(4):
        for j in range(4):
            points.append([j * scale, i * scale])
    pts = np.array(points)
    center = pts.mean(axis=0)
    pts = pts - center
    max_norm = np.max(np.linalg.norm(pts, axis=1))
    if max_norm > 0:
        pts = pts / max_norm * 0.9
    return pts

def circle_init(n, scale=1.0):
    """Generate points on a circle."""
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x = scale * np.cos(angles)
    y = scale * np.sin(angles)
    return np.column_stack((x, y))

def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 16
    deadline = time.time() + 580 # Safety margin
    
    best_result = None
    best_score = -1.0
    
    # Generate a set of diverse initial points
    # We will try to optimize each one locally
    initializations = []
    
    # Hexagonal grids with various rotations
    for rot in [0, np.pi/6, np.pi/3, np.pi/2]:
        initializations.append(hexagonal_init(n, rotation=rot))
        
    # Square grid
    initializations.append(square_init(n))
    
    # Points on a circle
    initializations.append(circle_init(n))
    
    # Random points
    for seed in [1, 2, 3, 4, 5]:
        initializations.append(random_init(n, seed=seed))
        
    # Jittered hexagonal grids
    rng = np.random.RandomState(123)
    for _ in range(5):
        base = hexagonal_init(n, rotation=rng.rand() * np.pi)
        jitter = rng.randn(n, 2) * 0.1
        initializations.append(base + jitter)
        
    # Loop through initializations and try to improve them
    # We use Powell's method which is robust and derivative-free
    # We limit the time per optimization to ensure we try many starts
    
    time_per_start = 10.0 # seconds
    start_time = time.time()
    
    for i, init_pts in enumerate(initializations):
        if time.time() > deadline - 30: # Leave time for final polish
            break
            
        # Flatten for optimizer
        x0 = init_pts.flatten()
        
        # We want to maximize score, so minimize -score
        # Powell method is good for this
        try:
            # maxiter is a hard stop, but we also check time in a callback if needed
            # However, minimize doesn't have a built-in time limit easily without callbacks.
            # We can set a reasonable maxiter.
            res = minimize(
                objective, 
                x0, 
                method='Powell', 
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
            )
            
            if res.success or res.fun < 0:
                points_opt = res.x.reshape(-1, 2)
                score_opt = -res.fun
                
                if score_opt > best_score:
                    best_score = score_opt
                    best_result = points_opt
                    
        except Exception as e:
            pass
            
    # If we didn't find anything, fallback
    if best_result is None:
        best_result = hexagonal_init(n)
        best_score = calculate_score(best_result)
        
    # Final Polish with Simulated Annealing if time permits
    if time.time() < deadline - 10:
        # Simple SA polish
        current_points = best_result.copy()
        current_score = best_score
        best_points = current_points.copy()
        
        start_time_sa = time.time()
        temp = 0.05
        cooling_rate = 0.999
        rng = np.random.RandomState(999)
        
        while time.time() - start_time_sa < 30: # Limit SA to 30s
            if temp < 1e-7:
                break
                
            idx = rng.randint(n)
            delta = rng.randn(2) * temp
            new_points = current_points.copy()
            new_points[idx] += delta
            
            new_score = calculate_score(new_points)
            
            if new_score > current_score:
                current_points = new_points
                current_score = new_score
                if current_score > best_score:
                    best_points = current_points.copy()
                    best_score = current_score
            else:
                delta_score = new_score - current_score
                if delta_score < 0:
                    if rng.rand() < np.exp(delta_score / temp):
                        current_points = new_points
                        current_score = new_score
                        
            temp *= cooling_rate
            
        if best_score > calculate_score(best_result):
            best_result = best_points

    return best_result

# EVOLVE-BLOCK-END
