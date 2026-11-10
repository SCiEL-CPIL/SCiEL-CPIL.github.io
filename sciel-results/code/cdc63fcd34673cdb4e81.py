"""Rewritten program for minimizing logarithmic energy of 100 points on the sphere."""
import time
import math
import numpy as np

N = 100

# EVOLVE-BLOCK-START

def _log_energy(pts):
    """Compute logarithmic energy: sum_{i<j} log(1/|x_i - x_j|)."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    iu = np.triu_indices(n, k=1)
    d_sq = dist_sq[iu]
    d = np.sqrt(d_sq)
    d = np.maximum(d, 1e-15)
    return math.fsum(math.log(1.0 / d) for d in d)

def _log_energy_grad(pts):
    """Compute logarithmic energy and its gradient."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    inv_dist_sq = np.zeros_like(dist_sq)
    off_diag = dist_sq > 1e-30
    inv_dist_sq[off_diag] = 1.0 / dist_sq[off_diag]
    
    grad = -np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    iu = np.triu_indices(n, k=1)
    d = np.sqrt(dist_sq[iu])
    d = np.maximum(d, 1e-15)
    energy = math.fsum(math.log(1.0 / d) for d in d)
    
    return energy, grad

def _project_to_tangent(pts, grad):
    """Project gradient onto tangent space of unit sphere."""
    return grad - np.sum(grad * pts, axis=1, keepdims=True) * pts

def _make_fibonacci_sphere():
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    pts = pts / norms
    return pts

def _random_sphere(n, seed=None):
    """Generate n random points on the unit sphere."""
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random
    pts = rng.randn(n, 3)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    return pts

def _adam_optimize(pts0, time_limit=100.0, lr_init=0.005):
    """Optimize points on the sphere using Adam gradient descent with adaptive learning rate."""
    pts = pts0.copy()
    best_pts = pts.copy()
    best_energy = _log_energy(pts)
    
    lr = lr_init
    beta1 = 0.9
    beta2 = 0.999
    epsilon = 1e-8
    
    m = np.zeros_like(pts)
    v = np.zeros_like(pts)
    t = 0
    
    start_time = time.time()
    stall_count = 0
    stall_threshold = 50
    
    while time.time() - start_time < time_limit:
        t += 1
        energy, grad = _log_energy_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
            stall_count = 0
        else:
            stall_count += 1
        
        grad_tangent = _project_to_tangent(pts, grad)
        
        m = beta1 * m + (1 - beta1) * grad_tangent
        v = beta2 * v + (1 - beta2) * (grad_tangent * grad_tangent)
        
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        step = lr * m_hat / (np.sqrt(v_hat) + epsilon)
        pts_new = pts - step
        
        norms = np.linalg.norm(pts_new, axis=1, keepdims=True)
        pts_new = pts_new / norms
        
        if not np.all(np.isfinite(pts_new)):
            break
        
        pts = pts_new
        
        # Adaptive learning rate
        if t % 50 == 0:
            if stall_count > stall_threshold:
                lr *= 0.95
                stall_count = 0
                if lr < 1e-7:
                    break
            elif energy < best_energy * 0.999:
                lr = min(lr * 1.02, 0.01)
    
    return best_pts, best_energy

def _lbfgs_optimize(pts0, time_limit=50.0):
    """Optimize using L-BFGS-B with sphere constraint via penalty."""
    from scipy.optimize import minimize
    
    def objective_flat(flat_pts):
        pts = flat_pts.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        energy = _log_energy(pts)
        # Penalty for not being on sphere
        penalty = 100.0 * np.sum((norms - 1.0)**2)
        return energy + penalty
    
    def gradient_flat(flat_pts):
        pts = flat_pts.reshape(-1, 3)
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        pts = pts / norms
        energy, grad = _log_energy_grad(pts)
        # Project to tangent
        grad_tangent = _project_to_tangent(pts, grad)
        # Add penalty gradient
        penalty_grad = 200.0 * (norms - 1.0) * pts / (norms**2)
        total_grad = grad_tangent + penalty_grad
        return total_grad.flatten()
    
    flat_pts = pts0.flatten()
    
    start_time = time.time()
    result = minimize(
        objective_flat,
        flat_pts,
        jac=gradient_flat,
        method='L-BFGS-B',
        options={'maxiter': 500, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    if time.time() - start_time > time_limit:
        return pts0, _log_energy(pts0)
    
    if result.success or result.fun < _log_energy(pts0):
        best_pts = result.x.reshape(-1, 3)
        norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
        best_pts = best_pts / norms
        return best_pts, _log_energy(best_pts)
    
    return pts0, _log_energy(pts0)

def construct():
    """Construct 100 points on the unit sphere minimizing logarithmic energy."""
    deadline = time.time() + 145.0  # 5 second safety margin before 150s kill
    
    # Start with Fibonacci spiral
    pts0 = _make_fibonacci_sphere()
    best_pts = pts0.copy()
    best_energy = _log_energy(pts0)
    
    # Phase 1: Adam optimization from Fibonacci start
    remaining = deadline - time.time()
    if remaining > 10.0:
        pts_opt, energy_opt = _adam_optimize(pts0, time_limit=remaining * 0.6, lr_init=0.005)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 2: L-BFGS refinement
    remaining = deadline - time.time()
    if remaining > 15.0:
        pts_opt, energy_opt = _lbfgs_optimize(best_pts, time_limit=min(remaining * 0.5, 40.0))
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 3: Random restarts with Adam
    rng = np.random.RandomState(42)
    n_restarts = 8
    for r in range(n_restarts):
        if time.time() > deadline - 15.0:
            break
        
        # Different starting points
        if r % 3 == 0:
            # Random
            pts_rand = _random_sphere(N, seed=42 + r)
        elif r % 3 == 1:
            # Perturbed Fibonacci
            pts_rand = pts0 + 0.15 * rng.randn(N, 3)
            pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        else:
            # Perturbed best
            pts_rand = best_pts + 0.05 * rng.randn(N, 3)
            pts_rand /= np.linalg.norm(pts_rand, axis=1, keepdims=True)
        
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
        
        t_limit = max(remaining * 0.4, 2.0)
        pts_opt, energy_opt = _adam_optimize(pts_rand, time_limit=t_limit, lr_init=0.003)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 4: Final fine-tuning with smaller learning rate
    if time.time() < deadline - 5.0:
        remaining = max(deadline - time.time() - 3.0, 1.0)
        pts_opt, energy_opt = _adam_optimize(best_pts, time_limit=remaining, lr_init=0.001)
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Phase 5: Final L-BFGS polish
    if time.time() < deadline - 3.0:
        remaining = max(deadline - time.time() - 2.0, 1.0)
        pts_opt, energy_opt = _lbfgs_optimize(best_pts, time_limit=min(remaining, 20.0))
        if energy_opt < best_energy:
            best_energy = energy_opt
            best_pts = pts_opt
    
    # Ensure points are on the unit sphere and valid
    norms = np.linalg.norm(best_pts, axis=1, keepdims=True)
    best_pts = best_pts / norms
    
    # Check for very close points and perturb if necessary
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist = np.linalg.norm(diff, axis=2)
    iu = np.triu_indices(N, k=1)
    min_dist = np.min(dist[iu])
    if min_dist < 1e-9:
        for _ in range(200):
            min_idx = np.argmin(dist[iu])
            i, j = iu[0][min_idx], iu[1][min_idx]
            rand_dir = np.random.randn(3)
            rand_dir /= np.linalg.norm(rand_dir)
            best_pts[j] = best_pts[j] + 1e-6 * rand_dir
            best_pts[j] /= np.linalg.norm(best_pts[j])
            diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
            dist = np.linalg.norm(diff, axis=2)
            if np.min(dist[iu]) >= 1e-9:
                break
    
    return best_pts

# EVOLVE-BLOCK-END
