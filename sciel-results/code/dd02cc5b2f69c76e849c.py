"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import time
import math


def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    # Compute all pairwise distances
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff * diff, axis=2))
    # Zero out diagonal to avoid log(0)
    np.fill_diagonal(dist, 1.0)
    # Energy is sum of log(1/d) = -sum of log(d) over all pairs i<j
    # We need to sum only upper triangle
    iu = np.triu_indices(N, k=1)
    return -math.fsum(np.log(dist[iu]).tolist())


def _gradient(pts):
    """Compute gradient of log energy w.r.t. points.
    
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    
    dE/dx_i = sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    
    Wait, let me re-derive:
    E = sum_{i<j} -log(|x_i - x_j|)
    dE/dx_i = sum_{j != i} -1/|x_i - x_j| * (x_i - x_j)/|x_i - x_j|
            = -sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
            = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # diff[i,j] = x_i - x_j
    dist_sq = np.sum(diff * diff, axis=2)
    np.fill_diagonal(dist_sq, 1.0)  # avoid division by zero
    dist = np.sqrt(dist_sq)
    
    # grad[i] = sum_j (x_j - x_i) / |x_i - x_j|^2 = -sum_j diff[i,j] / dist_sq[i,j]
    # diff[i,j] = x_i - x_j, so (x_j - x_i) = -diff[i,j]
    # grad[i] = -sum_j diff[i,j] / dist_sq[i,j]
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    return grad


def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _random_points(n, rng):
    """Generate random points on unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _gradient_descent(pts, max_iter=5000, lr_init=0.01, deadline=None):
    """Gradient descent with adaptive learning rate and sphere projection."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    current_energy = _log_energy(pts)
    
    lr = lr_init
    patience = 0
    best_pts = pts.copy()
    best_energy = current_energy
    min_improvement = 1e-10
    
    start_time = time.time()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        grad = _gradient(pts)
        
        # Adaptive learning rate: reduce if energy increases
        new_pts = pts - lr * grad
        new_pts = _project_to_sphere(new_pts)
        
        # Check for very close points
        min_dist = np.min(np.linalg.norm(new_pts[:, np.newaxis, :] - new_pts[np.newaxis, :, :], axis=2) * (1 - np.eye(N)))
        
        if min_dist < 1e-8:
            lr *= 0.5
            patience += 1
            if patience > 50:
                break
            continue
        
        new_energy = _log_energy(new_pts)
        
        if new_energy < current_energy - min_improvement:
            pts = new_pts
            current_energy = new_energy
            lr *= 1.05  # slight increase
            patience = 0
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        else:
            lr *= 0.7
            patience += 1
            if patience > 100:
                break
    
    return best_pts, best_energy


def _simulated_annealing(pts, max_iter=3000, t_init=0.01, t_min=1e-6, deadline=None):
    """Simulated annealing for escaping local minima."""
    pts = pts.copy()
    pts = _project_to_sphere(pts)
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    T = t_init
    cooling = (t_min / t_init) ** (1.0 / max_iter)
    
    start_time = time.time()
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Random perturbation
        noise_scale = 0.01 * (1 + it / max_iter)
        new_pts = pts + noise_scale * np.random.standard_normal((N, 3))
        new_pts = _project_to_sphere(new_pts)
        
        new_energy = _log_energy(new_pts)
        delta = new_energy - current_energy
        
        if delta < 0 or np.random.random() < math.exp(-delta / max(T, 1e-10)):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
        
        T *= cooling
    
    return best_pts, best_energy


def construct():
    deadline = time.time() + 145.0  # Safety margin before 150s limit
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Start with Fibonacci spiral
    fib_pts = _fibonacci_points(N)
    
    # Multi-restart strategy: try several initializations
    initializations = [fib_pts]
    
    # Add some random starts
    for _ in range(3):
        initializations.append(_random_points(N, rng))
    
    # Add perturbed Fibonacci
    for _ in range(2):
        perturbed = fib_pts + 0.1 * rng.standard_normal((N, 3))
        perturbed = _project_to_sphere(perturbed)
        initializations.append(perturbed)
    
    for init_pts in initializations:
        if time.time() > deadline - 30.0:
            break
        
        # Phase 1: Gradient descent
        gd_pts, gd_energy = _gradient_descent(init_pts, max_iter=3000, lr_init=0.01, deadline=deadline - 10.0)
        
        if gd_energy < best_energy:
            best_energy = gd_energy
            best_pts = gd_pts.copy()
        
        # Phase 2: Simulated annealing from GD result
        if time.time() < deadline - 15.0:
            sa_pts, sa_energy = _simulated_annealing(gd_pts, max_iter=2000, t_init=0.005, t_min=1e-7, deadline=deadline - 5.0)
            
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
        
        # Phase 3: Another round of GD from SA result
        if time.time() < deadline - 10.0:
            gd2_pts, gd2_energy = _gradient_descent(sa_pts, max_iter=2000, lr_init=0.005, deadline=deadline - 5.0)
            
            if gd2_energy < best_energy:
                best_energy = gd2_energy
                best_pts = gd2_pts.copy()
    
    if best_pts is None:
        best_pts = fib_pts
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    
    # Check for very close points
    dists = np.linalg.norm(best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :], axis=2)
    np.fill_diagonal(dists, 1.0)
    if np.min(dists) < 1e-9:
        # Fallback to Fibonacci if points are too close
        best_pts = fib_pts
    
    return best_pts
# EVOLVE-BLOCK-END
