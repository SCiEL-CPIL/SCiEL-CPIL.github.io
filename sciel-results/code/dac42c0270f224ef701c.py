"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 100
DEADLINE = 145.0  # safety margin before 150s kill

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _log_energy(points):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|) using math.fsum for precision."""
    n = len(points)
    # Compute pairwise squared distances
    # points is (N, 3)
    sq_dist = np.sum(points**2, axis=1)
    sq_dist = sq_dist[:, np.newaxis] + sq_dist[np.newaxis, :] - 2.0 * (points @ points.T)
    sq_dist = np.maximum(sq_dist, 0.0)
    # Upper triangle indices
    iu = np.triu_indices(n, k=1)
    d_sq = sq_dist[iu]
    d = np.sqrt(d_sq)
    # Use math.fsum for precision
    terms = [-math.log(max(di, 1e-15)) for di in d]
    return math.fsum(terms)

def _gradient(points):
    """Vectorized gradient of log energy w.r.t. each point on sphere.
    E = sum_{i<j} log(1/|x_i - x_j|)
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(points)
    # Compute pairwise differences and distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]  # (n, n, 3)
    sq_dist = np.sum(diff * diff, axis=2)  # (n, n)
    # Avoid division by zero on diagonal
    sq_dist = np.maximum(sq_dist, 1e-30)
    # grad[i] = sum_j diff[i,j] / sq_dist[i,j]
    inv_sq_dist = 1.0 / sq_dist  # (n, n)
    # Set diagonal to 0
    np.fill_diagonal(inv_sq_dist, 0.0)
    # grad = diff * inv_sq_dist[..., np.newaxis], summed over j
    grad = np.sum(diff * inv_sq_dist[:, :, np.newaxis], axis=1)
    return grad

def _project_to_sphere(points):
    """Project points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return points / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _momentum_descent(points, max_iters=5000, tol=1e-12, deadline=None, momentum=0.9):
    """Momentum-based projected gradient descent on sphere with line search."""
    points = points.copy()
    n = len(points)
    E = _log_energy(points)
    velocity = np.zeros_like(points)
    
    # Adaptive step size with line search
    lr = 0.05
    min_lr = 1e-10
    max_lr = 0.5
    
    for it in range(max_iters):
        if deadline is not None and (time.time() - deadline) > -15:
            break
        
        g = _gradient(points)
        g_norm = np.linalg.norm(g)
        
        if g_norm < tol:
            break
        
        # Momentum update
        velocity = momentum * velocity - lr * g
        new_points = points + velocity
        new_points = _project_to_sphere(new_points)
        new_E = _log_energy(new_points)
        
        # Line search: accept if energy decreased, else reduce lr
        if new_E < E:
            E = new_E
            points = new_points
            # Increase step size slightly if improvement is significant
            if new_E < E + 1e-10:
                lr = min(lr * 1.05, max_lr)
        else:
            # Backtrack: reduce lr and retry
            lr *= 0.5
            if lr < min_lr:
                # Reset velocity and try with smaller step
                velocity = np.zeros_like(points)
                lr = 0.01
                continue
    
    return points, E

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE - 5  # Safety margin
    
    # Start with Fibonacci spiral
    points = _fibonacci_sphere(N)
    E = _log_energy(points)
    best_points = points.copy()
    best_E = E
    
    # Run momentum descent from Fibonacci start
    points, E = _momentum_descent(points, max_iters=8000, tol=1e-12, deadline=start_time)
    if E < best_E:
        best_points = points.copy()
        best_E = E
    
    # Try several random restarts if time permits
    for restart in range(10):
        elapsed = time.time() - start_time
        if elapsed > deadline:
            break
        
        # Random start
        rng_points = _random_sphere(N, seed=restart * 42 + 7)
        # Quick momentum descent
        points, E = _momentum_descent(rng_points, max_iters=4000, tol=1e-11, deadline=start_time)
        if E < best_E:
            best_points = points.copy()
            best_E = E
    
    # Final high-precision polish
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        points, E = _momentum_descent(best_points, max_iters=3000, tol=1e-14, momentum=0.5, deadline=start_time)
        if E < best_E:
            best_points = points.copy()
            best_E = E
    
    # Ensure valid output
    best_points = _project_to_sphere(best_points)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_points[i] - best_points[j])
            if d < 1e-9:
                # Nudge apart
                best_points[j] = best_points[j] + 1e-8 * best_points[i]
                best_points[j] /= np.linalg.norm(best_points[j])
    
    return best_points
# EVOLVE-BLOCK-END
