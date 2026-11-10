"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize

def _energy_and_grad(x):
    """
    Compute logarithmic energy and gradient for N points on unit sphere.
    x: (N, 3) array of points on unit sphere.
    Returns (energy, grad) where grad is (N, 3).
    """
    dots = x @ x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(N, k=1)
    energy = -0.5 * np.sum(np.log(dist_sq[iu]))
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    S = np.sum(inv_dist_sq, axis=1)
    W = inv_dist_sq @ x
    grad = -x * S[:, np.newaxis] + W
    
    return energy, grad

def _energy(x):
    e, _ = _energy_and_grad(x)
    return e

def _project_and_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return x / norms

def _fmin_on_sphere(x0, maxiter=10000, tol=1e-14, gtol=1e-10):
    """
    Minimize energy on the sphere using L-BFGS-B with sphere projection.
    """
    x = x0.copy()
    x = _project_and_normalize(x)
    
    def objective_and_grad(x_flat):
        x_2d = x_flat.reshape(N, 3)
        x_2d = _project_and_normalize(x_2d)
        e, g = _energy_and_grad(x_2d)
        g_tangent = g - (g * x_2d).sum(axis=1, keepdims=True) * x_2d
        return e, g_tangent.flatten()
    
    res = minimize(
        objective_and_grad,
        x.flatten(),
        jac=True,
        method='L-BFGS-B',
        options={'maxiter': maxiter, 'ftol': tol, 'gtol': gtol, 'maxls': 50}
    )
    
    x_opt = _project_and_normalize(res.x.reshape(N, 3))
    return x_opt

def _simulated_annealing(x0, deadline, n_iterations=8000):
    """
    Simulated annealing with per-point adaptive step sizes and periodic re-optimization.
    """
    x = x0.copy()
    x = _project_and_normalize(x)
    current_e = _energy(x)
    best_x = x.copy()
    best_e = current_e
    
    T0 = 0.05
    T_min = 1e-8
    alpha = 0.9995
    
    # Per-point step sizes
    step_sizes = np.ones(N) * 0.03
    
    for it in range(n_iterations):
        if time.time() > deadline:
            break
        
        T = T0 * (alpha ** it)
        if T < T_min:
            break
        
        # Perturb each point with its own step size
        perturbation = np.random.randn(N, 3) * step_sizes[:, np.newaxis]
        x_new = _project_and_normalize(x + perturbation)
        
        e_new = _energy(x_new)
        delta_e = e_new - current_e
        
        accepted = False
        if delta_e < 0 or (T > 0 and np.random.rand() < np.exp(-delta_e / T)):
            x = x_new
            current_e = e_new
            accepted = True
            
            if current_e < best_e:
                best_e = current_e
                best_x = x.copy()
        
        # Adapt step sizes: increase for accepted, decrease for rejected
        if accepted:
            step_sizes *= 1.02
        else:
            step_sizes *= 0.98
        # Clamp step sizes
        step_sizes = np.clip(step_sizes, 0.001, 0.2)
        
        # Periodic re-optimization to escape local minima
        if it % 500 == 499:
            try:
                x_refined = _fmin_on_sphere(x, maxiter=2000, tol=1e-12, gtol=1e-8)
                e_refined = _energy(x_refined)
                if e_refined < current_e:
                    x = x_refined
                    current_e = e_refined
                    if current_e < best_e:
                        best_e = current_e
                        best_x = x.copy()
            except Exception:
                pass
    
    return best_x, best_e

def _fibonacci_spiral():
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    np.random.seed(12345)
    
    # Generate diverse starting points
    x0 = _fibonacci_spiral()
    
    starts = [x0.copy()]
    
    # Perturbed Fibonacci spirals with various scales
    for pert_scale in [0.02, 0.05, 0.1, 0.15, 0.25]:
        x_pert = x0 + np.random.randn(N, 3) * pert_scale
        x_pert = _project_and_normalize(x_pert)
        starts.append(x_pert)
    
    # Random points
    for _ in range(15):
        x_rand = np.random.randn(N, 3)
        x_rand /= np.linalg.norm(x_rand, axis=1, keepdims=True)
        starts.append(x_rand)
    
    # Ring-based initialization: points on rings at different latitudes
    n_rings = 10
    pts_per_ring = N // n_rings
    ring_pts = []
    for r in range(n_rings):
        z = 1.0 - 2.0 * (r + 0.5) / n_rings
        r_horiz = np.sqrt(1.0 - z * z)
        for k in range(pts_per_ring):
            theta = 2.0 * np.pi * k / pts_per_ring + np.pi * r / n_rings
            ring_pts.append([r_horiz * np.cos(theta), r_horiz * np.sin(theta), z])
    ring_pts = np.array(ring_pts[:N])
    starts.append(ring_pts)
    
    # Optimized ring with perturbation
    ring_pert = ring_pts + np.random.randn(N, 3) * 0.1
    ring_pert = _project_and_normalize(ring_pert)
    starts.append(ring_pert)
    
    best_x = x0.copy()
    best_e = _energy(best_x)
    
    # Optimize each starting point
    for x_init in starts:
        if time.time() > deadline - 40.0:
            break
        
        try:
            x_opt = _fmin_on_sphere(x_init, maxiter=8000, tol=1e-12, gtol=1e-9)
            e_opt = _energy(x_opt)
            
            if e_opt < best_e:
                best_e = e_opt
                best_x = x_opt.copy()
        except Exception:
            continue
    
    # Simulated annealing to escape local minima
    if time.time() < deadline - 30.0:
        try:
            sa_x, sa_e = _simulated_annealing(best_x, deadline - 20.0, n_iterations=6000)
            if sa_e < best_e:
                best_e = sa_e
                best_x = sa_x.copy()
        except Exception:
            pass
    
    # Second round of simulated annealing from best
    if time.time() < deadline - 20.0:
        try:
            sa_x2, sa_e2 = _simulated_annealing(best_x, deadline - 15.0, n_iterations=4000)
            if sa_e2 < best_e:
                best_e = sa_e2
                best_x = sa_x2.copy()
        except Exception:
            pass
    
    # Final high-precision polish
    if time.time() < deadline - 5.0:
        try:
            x_final = _fmin_on_sphere(best_x, maxiter=30000, tol=1e-16, gtol=1e-14)
            e_final = _energy(x_final)
            if e_final < best_e:
                best_e = e_final
                best_x = x_final
        except Exception:
            pass
        
        # Another polish pass
        if time.time() < deadline - 2.0:
            try:
                x_final2 = _fmin_on_sphere(best_x, maxiter=20000, tol=1e-16, gtol=1e-15)
                e_final2 = _energy(x_final2)
                if e_final2 < best_e:
                    best_e = e_final2
                    best_x = x_final2
            except Exception:
                pass
    
    # Ensure points are on unit sphere and valid
    best_x = _project_and_normalize(best_x)
    
    # Check for very close points
    dots = best_x @ best_x.T
    dist_sq = 2.0 - 2.0 * dots
    np.fill_diagonal(dist_sq, 1.0)
    min_dist_sq = np.min(dist_sq)
    
    if min_dist_sq < 1e-18:  # distance < 1e-9
        return x0
    
    return best_x
# EVOLVE-BLOCK-END
