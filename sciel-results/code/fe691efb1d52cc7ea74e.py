import numpy as np
import time
import math

N = 100

def energy_and_grad(points):
    """
    Calculate the logarithmic energy and its gradient.
    points: (N, 3) array of unit vectors.
    Returns:
        energy: float
        grad: (N, 3) array
    """
    # Calculate pairwise distances
    # dist2[i, j] = ||points[i] - points[j]||^2
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist2 = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    # We only sum over i < j, so we can mask the diagonal
    np.fill_diagonal(dist2, 1.0)
    
    # Energy: sum_{i<j} log(1/d) = - sum_{i<j} log(d)
    # We compute the full sum and divide by 2
    # log(d) = 0.5 * log(dist2)
    # So energy = -0.5 * sum_{i != j} log(dist2[i,j]) / 2 ? No.
    # sum_{i<j} log(d_ij) = 0.5 * sum_{i != j} log(d_ij)
    # E = - sum_{i<j} log(d_ij) = -0.5 * sum_{i != j} log(d_ij)
    
    # Clip dist2 to avoid log(0)
    dist2_clipped = np.maximum(dist2, 1e-12)
    
    log_dist2 = np.log(dist2_clipped)
    # Sum over all i != j
    total_log_dist2 = np.sum(log_dist2) - N * np.log(1.0) # diagonal is log(1)=0
    energy = -0.5 * total_log_dist2
    
    # Gradient:
    # grad_i = sum_{j != i} (x_j - x_i) / ||x_i - x_j||^2
    # Let's compute this efficiently.
    # diff[i, j] = x_i - x_j
    # We need sum_j (x_j - x_i) / dist2[i, j]
    # Note: diff is x_i - x_j, so x_j - x_i = -diff
    # grad_i = - sum_j diff[i, j] / dist2[i, j]
    
    # dist2 is (N, N)
    # We need to divide each row by dist2 row-wise, but only for j != i
    # Since dist2[i, i] is 1.0 (masked), and diff[i, i] is 0, the term is 0.
    # So we can just divide the whole matrix.
    
    inv_dist2 = 1.0 / dist2_clipped
    # Multiply diff by inv_dist2 element-wise
    # diff is (N, N, 3)
    # inv_dist2 is (N, N) -> (N, N, 1)
    weighted_diff = diff * inv_dist2[:, :, np.newaxis]
    
    # grad_i = - sum_j weighted_diff[i, j]
    grad = -np.sum(weighted_diff, axis=1)
    
    return energy, grad

def _project(X):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return X / norms

def _fibonacci_spiral():
    """Generate initial points using Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _sobol_like(seed=42):
    """Generate quasi-random points on the sphere using a Sobol-like sequence."""
    rng = np.random.RandomState(seed)
    points = np.zeros((N, 3))
    for i in range(N):
        # Use stratified sampling in spherical coordinates
        u = (i + 0.5) / N
        v = rng.uniform(0, 1)
        theta = 2 * np.pi * v
        phi = np.arccos(2 * u - 1)
        points[i] = [np.sin(phi) * np.cos(theta), np.sin(phi) * np.sin(theta), np.cos(phi)]
    return _project(points)

def _gradient_descent(points, deadline, lr=0.01, momentum=0.9, max_iter=2000):
    """Gradient descent with momentum and adaptive learning rate."""
    v = np.zeros_like(points)
    best_energy, best_points = np.inf, points.copy()
    e0, g0 = energy_and_grad(points)
    best_energy = e0
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
            
        v = momentum * v - lr * g0
        new_points = _project(points + v)
        e1, g1 = energy_and_grad(new_points)
        
        if e1 > e0 * 1.001:  # Allow small increases
            lr *= 0.5
            continue
        
        points = new_points
        g0 = g1
        if e1 < best_energy:
            best_energy = e1
            best_points = points.copy()
        
        lr = max(0.001, lr * 0.999)
    
    return best_points, best_energy

def _langevin_verlet(points, deadline, temperature=0.5, dt=0.01, max_iter=5000):
    """
    High-temperature Langevin dynamics with velocity Verlet integrator.
    This explores configuration space more broadly than local optimization.
    """
    rng = np.random.RandomState(123)
    velocities = np.zeros_like(points)
    best_energy, best_points = np.inf, points.copy()
    
    e0, g0 = energy_and_grad(points)
    best_energy = e0
    
    # Initialize velocities with thermal noise
    velocities = rng.randn(N, 3) * np.sqrt(temperature)
    
    for k in range(max_iter):
        if time.time() > deadline:
            break
        
        # Velocity Verlet step 1: half-step velocity
        v_half = velocities - 0.5 * dt * g0
        
        # Full-step position update
        new_points = points + dt * v_half
        new_points = _project(new_points)
        
        # Compute new gradient
        e1, g1 = energy_and_grad(new_points)
        
        # Velocity Verlet step 2: half-step velocity with new gradient
        velocities = v_half - 0.5 * dt * g1
        
        # Add stochastic kick (Langevin term)
        kick = np.sqrt(2 * temperature * dt) * rng.randn(N, 3)
        velocities = velocities + kick
        
        # Damping to maintain temperature
        velocities *= (1 - 0.01 * dt)
        
        points = new_points
        g0 = g1
        
        if e1 < best_energy:
            best_energy = e1
            best_points = points.copy()
        
        # Gradually cool down
        temperature *= 0.9995
    
    return best_points, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # 20s safety margin
    
    # Try multiple initialization strategies
    initializations = [_fibonacci_spiral(), _sobol_like(42), _sobol_like(123)]
    
    best_points = None
    best_energy = np.inf
    
    for init_idx, init_points in enumerate(initializations):
        if time.time() > deadline:
            break
        
        # Phase 1: Langevin dynamics for broad exploration
        lang_points, lang_energy = _langevin_verlet(
            init_points, deadline, 
            temperature=0.3, dt=0.02, 
            max_iter=2000
        )
        
        # Phase 2: Gradient descent for local refinement
        grad_points, grad_energy = _gradient_descent(
            lang_points, deadline,
            lr=0.005, momentum=0.8,
            max_iter=3000
        )
        
        if grad_energy < best_energy:
            best_energy = grad_energy
            best_points = grad_points.copy()
    
    # Final polish with very small steps
    if best_points is not None and time.time() < deadline - 10:
        final_points, final_energy = _gradient_descent(
            best_points, deadline,
            lr=0.001, momentum=0.5,
            max_iter=5000
        )
        if final_energy < best_energy:
            best_points = final_points
            best_energy = final_energy
    
    if best_points is None:
        best_points = _fibonacci_spiral()
    
    return best_points
