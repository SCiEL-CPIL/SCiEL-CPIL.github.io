"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import math
import time
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """Compute logarithmic energy and Euclidean gradient for N points on sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    n = len(pts)
    # Pairwise differences: diff[i,j] = x_i - x_j
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    dist = np.sqrt(dist_sq)
    
    # Energy: sum over i<j of -log(dist_ij)
    iu = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[iu]))
    
    # Gradient: grad_i = sum_{j!=i} (x_i - x_j) / dist_ij^2
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad


def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms


def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci sphere points with random rotation."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Apply random rotation
    theta = rng.uniform(0, 2 * np.pi)
    phi_rot = rng.uniform(-np.pi/2, np.pi/2)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    cos_p, sin_p = np.cos(phi_rot), np.sin(phi_rot)
    
    R = np.array([
        [cos_t, sin_t, 0],
        [-sin_t, cos_t, 0],
        [0, 0, 1]
    ])
    R2 = np.array([
        [1, 0, 0],
        [0, cos_p, sin_p],
        [0, -sin_p, cos_p]
    ])
    R_total = R2 @ R
    pts = pts @ R_total.T
    
    return pts


def _random_sphere(n, rng):
    """Generate n random points on the unit sphere."""
    pts = rng.standard_normal((n, 3))
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    return pts / norms


def _riemannian_descent(pts, max_iters=5000, tol=1e-12, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    
    lr = 0.01
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, euclidean_grad = _energy_and_grad(pts)
        
        # Riemannian gradient: project to tangent space
        dot = np.sum(euclidean_grad * pts, axis=1, keepdims=True)
        rgrad = euclidean_grad - dot * pts
        
        # Check convergence
        grad_norm = np.linalg.norm(rgrad)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        alpha = lr
        for backtrack in range(30):
            pts_trial = pts + alpha * rgrad
            pts_trial = _project_to_sphere(pts_trial)
            
            trial_energy, _ = _energy_and_grad(pts_trial)
            
            if trial_energy < energy:
                break
            alpha *= 0.5
        
        pts = pts_trial
        energy, _ = _energy_and_grad(pts)
        
        if energy < best_energy:
            best_energy = energy
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if trial_energy < energy - 1e-10:
            lr = min(lr * 1.1, 0.1)
        else:
            lr = max(lr * 0.9, 1e-6)
    
    return best_pts, best_energy


def _lbfgs_optimize(pts, deadline=None):
    """Optimize using L-BFGS-B on flattened coordinates with sphere constraint via projection."""
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    
    def objective(x_flat):
        if deadline is not None and time.time() > deadline:
            return float('inf')
        x = x_flat.reshape(n, 3)
        x = _project_to_sphere(x)
        energy, _ = _energy_and_grad(x)
        return energy
    
    def gradient(x_flat):
        if deadline is not None and time.time() > deadline:
            return np.zeros_like(x_flat)
        x = x_flat.reshape(n, 3)
        x = _project_to_sphere(x)
        energy, grad = _energy_and_grad(x)
        # Project gradient to tangent space
        dot = np.sum(grad * x, axis=1, keepdims=True)
        grad_tangent = grad - dot * x
        return grad_tangent.flatten()
    
    x0 = pts.flatten()
    result = minimize(objective, x0, jac=gradient, method='L-BFGS-B',
                      options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12,
                               'maxls': 50, 'maxfun': 10000})
    
    if result.success or result.fun < float('inf'):
        opt_pts = _project_to_sphere(result.x.reshape(n, 3))
        opt_energy, _ = _energy_and_grad(opt_pts)
        return opt_pts, opt_energy
    else:
        return pts, _energy_and_grad(pts)[0]


def _simulated_annealing(pts, t_start=0.05, t_end=0.0005, max_iter=150, cooling_rate=0.985, deadline=None):
    """Refine with simulated annealing to escape local minima."""
    pts = _project_to_sphere(pts.copy())
    n = len(pts)
    energy, _ = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    t = t_start
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate random perturbation in tangent space
        noise = np.random.randn(n, 3)
        radial_comp = (noise * pts).sum(axis=1, keepdims=True)
        noise_tangent = noise - radial_comp * pts
        noise_tangent *= 0.01 * t
        
        new_pts = pts + noise_tangent
        new_pts = _project_to_sphere(new_pts)
        
        new_energy, _ = _energy_and_grad(new_pts)
        delta_e = new_energy - energy
        
        if delta_e < 0 or np.random.rand() < math.exp(-delta_e / t):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        t *= cooling_rate
        if t < t_end:
            break
    
    return best_pts, best_energy


def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 120s limit
    
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Multiple restarts with Riemannian descent
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() > deadline - 10.0:
            break
        
        if restart == 0:
            pts = _fibonacci_sphere(N, seed=0)
        elif restart < 4:
            pts = _random_sphere(N, rng)
        elif restart < 8:
            pts = _fibonacci_sphere(N, seed=restart + 100)
        else:
            # Clustered starts
            centers = rng.standard_normal((5, 3))
            centers = _project_to_sphere(centers)
            pts = []
            for i in range(N):
                center = centers[i % 5]
                noise = rng.standard_normal(3) * 0.1
                pt = center + noise
                pt = pt / np.linalg.norm(pt)
                pts.append(pt)
            pts = np.array(pts)
        
        try:
            opt_pts, opt_energy = _riemannian_descent(pts, max_iters=2000, tol=1e-12, deadline=deadline)
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Phase 2: L-BFGS polish on best result
    if best_pts is not None and time.time() < deadline - 5.0:
        try:
            lbfgs_pts, lbfgs_energy = _lbfgs_optimize(best_pts, deadline=deadline)
            if lbfgs_energy < best_energy:
                best_energy = lbfgs_energy
                best_pts = lbfgs_pts.copy()
        except Exception:
            pass
    
    # Phase 3: Simulated annealing refinement
    if best_pts is not None and time.time() < deadline - 3.0:
        try:
            sa_pts, sa_energy = _simulated_annealing(best_pts, t_start=0.03, t_end=0.0003,
                                                     max_iter=100, deadline=deadline)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 1.0:
        try:
            final_pts, final_energy = _riemannian_descent(best_pts, max_iters=3000, tol=1e-14,
                                                          deadline=deadline)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, seed=0)
    
    # Ensure points are on the unit sphere and valid
    best_pts = _project_to_sphere(best_pts)
    
    # Check for near-duplicate points
    for i in range(N):
        for j in range(i + 1, N):
            d = np.linalg.norm(best_pts[i] - best_pts[j])
            if d < 1e-9:
                best_pts[j] += rng.standard_normal(3) * 1e-6
                best_pts[j] /= np.linalg.norm(best_pts[j])
    
    # Final validation
    if not np.all(np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N, seed=0)
    
    return best_pts
# EVOLVE-BLOCK-END
