"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math
from scipy.optimize import minimize

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for N points on sphere.
    
    Energy = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log(|x_i - x_j|)
    Gradient w.r.t. x_i: sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    """
    pts = np.asarray(pts, dtype=np.float64)
    n = len(pts)
    # Compute pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    dist = np.sqrt(dist_sq)
    
    # Energy: -sum_{i<j} log(d_ij)
    upper_tri = np.triu_indices(n, k=1)
    energy = -np.sum(np.log(dist[upper_tri]))
    
    # Gradient: for each i, grad_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # = sum_{j!=i} diff[i,j] / dist_sq[i,j]
    with np.errstate(divide='ignore', invalid='ignore'):
        weight = np.zeros_like(dist_sq)
        weight[upper_tri] = 1.0 / dist_sq[upper_tri]
        weight = weight + weight.T  # Symmetrize
    grad = np.sum(weight[:, :, np.newaxis] * diff, axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _riemannian_descent(pts, max_iter=8000, lr_init=0.01, lr_min=1e-7, tol=1e-12, deadline=None):
    """Riemannian gradient descent with backtracking line search on the sphere."""
    pts = _project_to_sphere(pts.copy())
    best_energy, _ = _energy_and_grad(pts)
    best_pts = pts.copy()
    lr = lr_init
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        energy, grad = _energy_and_grad(pts)
        
        # Riemannian gradient: project Euclidean gradient tangent to sphere
        grad_dot_x = np.sum(grad * pts, axis=1, keepdims=True)
        grad_tangent = grad - grad_dot_x * pts
        
        gnorm = np.linalg.norm(grad_tangent, ord=np.inf)
        if gnorm < tol:
            break
        
        # Backtracking line search
        step = lr
        for _ in range(30):
            new_pts = _project_to_sphere(pts + step * grad_tangent)
            new_energy, _ = _energy_and_grad(new_pts)
            if new_energy < energy:
                break
            step *= 0.5
        
        pts = new_pts
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = pts.copy()
        
        # Adaptive learning rate
        if new_energy < energy - 1e-12:
            lr = min(lr * 1.05, 0.08)
        else:
            lr = max(lr * 0.85, lr_min)
    
    return best_pts, best_energy

def _lbfgs_polish(pts, max_iter=500, ftol=1e-13, gtol=1e-11, deadline=None):
    """L-BFGS-B polish in ambient space with projection."""
    if deadline is not None and time.time() > deadline - 10:
        return pts, _energy_and_grad(pts)[0]
    
    def objective_and_grad(flat_pts):
        if deadline is not None and time.time() > deadline:
            raise TimeoutError
        pts_reshaped = flat_pts.reshape(-1, 3)
        pts_reshaped = _project_to_sphere(pts_reshaped)
        energy, grad = _energy_and_grad(pts_reshaped)
        return energy, grad.flatten()
    
    try:
        result = minimize(objective_and_grad, pts.flatten(), method='L-BFGS-B',
                          jac=True, options={'maxiter': max_iter, 'ftol': ftol, 
                                            'gtol': gtol, 'maxls': 50, 'maxfun': 1000})
        opt_pts = _project_to_sphere(result.x.reshape(-1, 3))
        opt_energy, _ = _energy_and_grad(opt_pts)
        return opt_pts, opt_energy
    except (TimeoutError, Exception):
        return pts, _energy_and_grad(pts)[0]

def _high_precision_polish(pts, deadline=None):
    """Final high-precision polish with very tight tolerances."""
    if deadline is not None and time.time() > deadline - 5:
        return pts, _energy_and_grad(pts)[0]
    
    # First pass with moderate precision
    pts1, e1 = _lbfgs_polish(pts, max_iter=400, ftol=1e-14, gtol=1e-12, deadline=deadline)
    
    # Second pass with even tighter precision
    if deadline is None or time.time() < deadline - 3:
        pts2, e2 = _lbfgs_polish(pts1, max_iter=300, ftol=1e-15, gtol=1e-13, deadline=deadline)
        if e2 < e1:
            return pts2, e2
    
    return pts1, e1

def _simulated_annealing(pts, t_start=0.08, t_end=0.0005, max_iter=150, cooling_rate=0.97, deadline=None):
    """Refine with simulated annealing to escape local minima."""
    pts = _project_to_sphere(pts.copy())
    energy, _ = _energy_and_grad(pts)
    best_energy = energy
    best_pts = pts.copy()
    
    t = t_start
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Generate random perturbation in tangent space
        noise = np.random.randn(N, 3)
        radial_comp = (noise * pts).sum(axis=1, keepdims=True)
        noise_tangent = noise - radial_comp * pts
        step_scale = 0.04 * (t / t_start) ** 0.5
        noise_tangent *= step_scale
        
        new_pts = _project_to_sphere(pts + noise_tangent)
        new_energy, _ = _energy_and_grad(new_pts)
        delta_e = new_energy - energy
        
        # Accept with probability exp(-delta_e / t)
        if delta_e < 0 or (t > 1e-6 and np.random.rand() < math.exp(min(0, -delta_e / t))):
            pts = new_pts
            energy = new_energy
            if energy < best_energy:
                best_energy = energy
                best_pts = pts.copy()
        
        t *= cooling_rate
        if t < t_end:
            break
    
    return best_pts, best_energy

def _fibonacci_sphere(n, seed=0):
    """Generate Fibonacci sphere points with random rotation."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = math.pi * (1.0 + math.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(1.0 - z * z, 0.0))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Apply random rotation
    theta = rng.uniform(0, 2 * math.pi)
    phi_rot = rng.uniform(-math.pi/2, math.pi/2)
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
    
    return _project_to_sphere(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin before 150s kill
    
    best_pts = None
    best_energy = float('inf')
    
    # Phase 1: Fibonacci sphere start with Riemannian descent + L-BFGS polish
    if time.time() < deadline - 30:
        pts = _fibonacci_sphere(N, seed=0)
        pts, e = _riemannian_descent(pts, max_iter=5000, lr_init=0.01, deadline=deadline - 20)
        pts, e = _lbfgs_polish(pts, max_iter=400, ftol=1e-13, gtol=1e-11, deadline=deadline - 10)
        if e < best_energy:
            best_energy = e
            best_pts = pts.copy()
    
    # Phase 2: Additional restarts with different initializations
    rng = np.random.default_rng(42)
    n_restarts = 12
    for restart in range(n_restarts):
        if time.time() > deadline - 25:
            break
        
        # Different initialization strategies
        if restart == 0:
            pts = _fibonacci_sphere(N, seed=100)
        elif restart < 4:
            # Random points
            pts = rng.normal(size=(N, 3))
            pts = _project_to_sphere(pts)
        elif restart < 8:
            # Perturbed best solution
            pts = _project_to_sphere(best_pts + 0.15 * rng.normal(size=(N, 3)))
        else:
            # Clustered starts
            n_clusters = 3 + (restart % 4)
            centers = rng.normal(size=(n_clusters, 3))
            centers = _project_to_sphere(centers)
            pts = []
            for i in range(N):
                center = centers[i % n_clusters]
                noise = rng.normal(size=3) * 0.2
                pt = center + noise
                pt = pt / np.linalg.norm(pt)
                pts.append(pt)
            pts = np.array(pts)
        
        # Optimize: Riemannian descent then L-BFGS polish
        try:
            opt_pts, opt_energy = _riemannian_descent(pts, max_iter=2000, lr_init=0.005, deadline=deadline - 15)
            if time.time() < deadline - 10:
                opt_pts, opt_energy = _lbfgs_polish(opt_pts, max_iter=300, ftol=1e-13, gtol=1e-11, deadline=deadline - 5)
            
            if opt_energy < best_energy:
                best_energy = opt_energy
                best_pts = opt_pts.copy()
        except Exception:
            continue
    
    # Phase 3: Simulated annealing refinement on best solution
    if best_pts is not None and time.time() < deadline - 20:
        try:
            sa_pts, sa_energy = _simulated_annealing(best_pts, t_start=0.06, t_end=0.0003, 
                                                     max_iter=100, deadline=deadline - 10)
            if sa_energy < best_energy:
                best_energy = sa_energy
                best_pts = sa_pts.copy()
            
            # Re-optimize from SA result
            if time.time() < deadline - 10:
                re_opt_pts, re_opt_energy = _lbfgs_polish(sa_pts, max_iter=300, 
                                                          ftol=1e-13, gtol=1e-11, deadline=deadline - 5)
                if re_opt_energy < best_energy:
                    best_energy = re_opt_energy
                    best_pts = re_opt_pts.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision polish
    if best_pts is not None and time.time() < deadline - 5:
        try:
            final_pts, final_energy = _high_precision_polish(best_pts, deadline=deadline)
            if final_energy < best_energy:
                best_energy = final_energy
                best_pts = final_pts.copy()
        except Exception:
            pass
    
    # Fallback
    if best_pts is None:
        best_pts = _fibonacci_sphere(N, seed=0)
    
    # Final validation
    best_pts = _project_to_sphere(best_pts)
    if not np.all(np.isfinite(best_pts)) or np.any(np.linalg.norm(best_pts, axis=1) < 1e-9):
        best_pts = _fibonacci_sphere(N, seed=0)
    
    # Ensure no coincident points
    diff = best_pts[:, np.newaxis, :] - best_pts[np.newaxis, :, :]
    dist_sq = np.einsum('ijk,ijk->ij', diff, diff)
    np.fill_diagonal(dist_sq, 1.0)
    if np.min(dist_sq) < 1e-18:
        best_pts += 1e-8 * rng.normal(size=(N, 3))
        best_pts = _project_to_sphere(best_pts)
    
    return best_pts
# EVOLVE-BLOCK-END
