"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time
import math

def _fibonacci_sphere():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(pts):
    """Compute logarithmic energy and gradient for points on unit sphere."""
    # pts: (N, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=-1)
    np.fill_diagonal(dists, 1.0)
    
    # Energy = sum_{i<j} log(1/d_ij) = -sum_{i<j} log(d_ij)
    upper = np.triu_indices(N, k=1)
    energy = -np.sum(np.log(dists[upper]))
    
    # Gradient: dE/dx_i = -sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    inv_d2 = np.zeros_like(dists)
    mask = dists > 1e-15
    inv_d2[mask] = 1.0 / (dists[mask] ** 2)
    grad = -np.sum(diff * inv_d2[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-15)
    return pts / norms

def _optimize_lbfgs(pts, deadline):
    """Use L-BFGS-B with custom gradient, optimizing on R^3 with projection."""
    from scipy.optimize import minimize
    
    def objective_and_grad(x_flat):
        pts = x_flat.reshape(N, 3)
        pts = _project_to_sphere(pts)
        e, g = _energy_and_grad(pts)
        # Project gradient to tangent space for the manifold optimization
        g_tangent = g - np.sum(g * pts, axis=1, keepdims=True) * pts
        return e, g_tangent.flatten()
    
    x0 = pts.flatten()
    result = minimize(
        objective_and_grad,
        x0,
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxls': 50}
    )
    
    best_pts = _project_to_sphere(result.x.reshape(N, 3))
    return best_pts

def _gradient_descent_polish(pts, deadline, lr_init=0.01, max_iter=10000):
    """Polish with adaptive gradient descent on the sphere."""
    best_pts = pts.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    lr = lr_init
    n_iter = 0
    
    while n_iter < max_iter and time.time() < deadline:
        e, g = _energy_and_grad(best_pts)
        if e < best_energy:
            best_energy = e
        
        # Project gradient to tangent space
        g_tangent = g - np.sum(g * best_pts, axis=1, keepdims=True) * best_pts
        
        # Update
        new_pts = _project_to_sphere(best_pts - lr * g_tangent)
        
        e_new, _ = _energy_and_grad(new_pts)
        if e_new < e:
            best_pts = new_pts
            if n_iter % 100 == 0 and lr < lr_init:
                lr *= 1.05
        else:
            lr *= 0.5
            if lr < 1e-10:
                break
        
        n_iter += 1
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Leave 5s safety margin
    
    # Start with Fibonacci sphere
    x0 = _fibonacci_sphere()
    best_pts = x0.copy()
    best_energy, _ = _energy_and_grad(best_pts)
    
    # Phase 1: L-BFGS-B optimization from Fibonacci
    if time.time() < deadline - 60:
        pts_lbfgs = _optimize_lbfgs(best_pts, deadline - 30)
        e_lbfgs, _ = _energy_and_grad(pts_lbfgs)
        if e_lbfgs < best_energy:
            best_pts = pts_lbfgs
            best_energy = e_lbfgs
    
    # Phase 2: Try a perturbed starting point to escape local minima
    if time.time() < deadline - 40:
        rng = np.random.default_rng(42)
        x0_perturbed = _fibonacci_sphere() + 0.05 * rng.standard_normal((N, 3))
        x0_perturbed = _project_to_sphere(x0_perturbed)
        pts_pert = _optimize_lbfgs(x0_perturbed, deadline - 20)
        e_pert, _ = _energy_and_grad(pts_pert)
        if e_pert < best_energy:
            best_pts = pts_pert
            best_energy = e_pert
    
    # Phase 3: Gradient descent polish with adaptive learning rate
    if time.time() < deadline - 10:
        pts_polish = _gradient_descent_polish(best_pts, deadline - 5, lr_init=0.005, max_iter=80000)
        e_polish, _ = _energy_and_grad(pts_polish)
        if e_polish < best_energy:
            best_pts = pts_polish
            best_energy = e_polish
    
    # Phase 4: Final high-precision polish with very small learning rate
    if time.time() < deadline - 1:
        pts_final = _gradient_descent_polish(best_pts, deadline, lr_init=0.0005, max_iter=200000)
        e_final, _ = _energy_and_grad(pts_final)
        if e_final < best_energy:
            best_pts = pts_final
            best_energy = e_final
    
    return best_pts
# EVOLVE-BLOCK-END
