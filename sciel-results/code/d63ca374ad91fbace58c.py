"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)"""
    n = len(pts)
    # Vectorized distance calculation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    # Avoid log(0) by setting diagonal to 1 (dist 1) and ignoring it in sum
    np.fill_diagonal(d2, 1.0)
    d = np.sqrt(d2)
    
    # Mask for upper triangle (i < j)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    d_upper = d[mask]
    
    if np.any(d_upper < 1e-15):
        return -1e18
    return math.fsum(np.log(1.0 / d_upper).tolist())

def _project_to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    return pts / norms

def _fibonacci_sphere(n):
    """Generate Fibonacci spiral points on unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=None):
    """Generate random points on unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms

def _simulated_annealing(pts0, start_temp, end_temp, steps, deadline):
    """
    Optimize using Simulated Annealing with Langevin-like updates.
    """
    pts = pts0.copy()
    current_energy = _log_energy(pts)
    best_pts = pts.copy()
    best_energy = current_energy
    
    # Pre-calculate mask for efficiency
    n = len(pts)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    
    # Parameters for noise
    # Scale noise based on temperature
    # We want to explore the manifold. 
    # A simple random walk on the sphere: pts = normalize(pts + noise)
    # Noise magnitude should be small enough to stay local but large enough to explore.
    
    for i in range(steps):
        if time.time() > deadline:
            break
            
        # Cooling schedule: linear or exponential? Exponential is standard.
        # t = start_temp * (end_temp/start_temp) ** (i/steps)
        temp = start_temp * (end_temp / start_temp) ** (i / steps)
        
        # Generate random perturbations
        # Use Gaussian noise in tangent space or just Euclidean then project?
        # Euclidean noise + projection is simpler and robust.
        # Step size needs to be tuned. 
        # Let's try a fixed small step size scaled by sqrt(temp) or similar.
        # Actually, standard SA uses a fixed neighborhood size or adaptive.
        # Let's use a base step size that decreases slightly or stays constant.
        # If we use random restarts of the noise, we might jump too far.
        # Let's stick to small perturbations.
        
        noise = np.random.randn(n, 3) * 0.05 
        # Scale noise by temperature? 
        # High T -> larger jumps. Low T -> smaller jumps.
        noise *= np.sqrt(temp)
        
        new_pts = _project_to_sphere(pts + noise)
        
        # Calculate energy change
        # To be fast, we could calculate full energy, but for N=100, it's cheap enough.
        # 100*100/2 = 5000 pairs. sqrt + log is fast.
        new_energy = _log_energy(new_pts)
        
        # Acceptance criterion
        delta_e = new_energy - current_energy
        if delta_e < 0 or np.random.rand() < np.exp(-delta_e / temp):
            pts = new_pts
            current_energy = new_energy
            if current_energy < best_energy:
                best_energy = current_energy
                best_pts = pts.copy()
                
    return best_pts, best_energy

def _gradient_descent_polish(pts0, deadline, max_iter=500):
    """
    Final polish with gradient descent to ensure we are at a local minimum.
    """
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    lr = 0.01
    v = np.zeros_like(pts)
    momentum = 0.9
    
    for i in range(max_iter):
        if time.time() > deadline:
            break
            
        # Calculate gradient
        n = len(pts)
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1.0)
        d = np.sqrt(d2)
        
        # Gradient of -log(d) wrt pts
        # d/dx_i sum_{j!=i} -log(|x_i - x_j|) = - sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
        # Note: d2 = |x_i - x_j|^2. So 1/d2 is the weight.
        weights = 1.0 / d2
        np.fill_diagonal(weights, 0.0)
        
        grad = -np.sum(weights[:, :, np.newaxis] * diff, axis=1)
        
        # Project gradient to tangent space
        radial = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - radial * pts
        
        # Momentum update
        v = momentum * v - lr * grad_tangent
        new_pts = _project_to_sphere(pts + v)
        
        new_energy = _log_energy(new_pts)
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # If energy increases, revert and reduce step
            v = 0.5 * v
            lr *= 0.9
            if lr < 1e-8:
                break
                
    return best_pts, best_energy

def construct():
    """Construct 100 points on unit sphere minimizing logarithmic energy."""
    start_time = time.time()
    # Safety margin: 150s limit, stop at 140s to be safe
    deadline = start_time + 140.0
    
    best_pts = None
    best_energy = np.inf
    
    # 1. Initial diverse starts
    starts = [
        ("fib", _fibonacci_sphere(N)),
        ("rand_1", _random_sphere(N, 1)),
        ("rand_2", _random_sphere(N, 2)),
        ("rand_3", _random_sphere(N, 3)),
    ]
    
    # 2. Run Simulated Annealing on each start
    # SA is good at escaping local minima.
    # We allocate time based on remaining budget.
    # Let's give each SA run a fixed number of steps, but check time.
    
    sa_configs = [
        {"start_temp": 0.1, "end_temp": 0.001, "steps": 2000},
        {"start_temp": 0.5, "end_temp": 0.01, "steps": 1500},
        {"start_temp": 0.2, "end_temp": 0.005, "steps": 1800},
        {"start_temp": 0.8, "end_temp": 0.02, "steps": 1200},
    ]
    
    for i, (name, pts0) in enumerate(starts):
        if time.time() > deadline - 10:
            break
            
        # Pick a config, maybe cycle or use i-th
        cfg = sa_configs[i % len(sa_configs)]
        
        # Run SA
        pts_sa, e_sa = _simulated_annealing(pts0, cfg["start_temp"], cfg["end_temp"], cfg["steps"], deadline)
        
        if e_sa < best_energy:
            best_energy = e_sa
            best_pts = pts_sa
            
        # 3. Polish with Gradient Descent
        if time.time() < deadline - 5:
            pts_gd, e_gd = _gradient_descent_polish(pts_sa, deadline)
            if e_gd < best_energy:
                best_energy = e_gd
                best_pts = pts_gd

    # 4. Final global polish on the best found so far
    if best_pts is not None and time.time() < deadline - 2:
        pts_final, e_final = _gradient_descent_polish(best_pts, deadline, max_iter=1000)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final
            
    # 5. Last resort: if we haven't improved much, try a random restart + SA + GD
    if time.time() < deadline - 15:
        pts_r = _random_sphere(N, 12345)
        pts_sa, e_sa = _simulated_annealing(pts_r, 0.3, 0.005, 2000, deadline)
        if e_sa < best_energy:
            best_energy = e_sa
            best_pts = pts_sa
        if time.time() < deadline - 2:
             pts_gd, e_gd = _gradient_descent_polish(pts_sa, deadline)
             if e_gd < best_energy:
                 best_energy = e_gd
                 best_pts = pts_gd

    return best_pts
# EVOLVE-BLOCK-END
