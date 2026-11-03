"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 150

# EVOLVE-BLOCK-START
import numpy as np
import math
import time

def _fibonacci_sphere(n):
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _energy_and_grad(pts):
    """Compute log-energy and tangent-projected gradient for points on sphere.
    E = sum_{i<j} log(1/|x_i - x_j|) = -sum_{i<j} log|d_ij|
    grad_i (unconstrained) = sum_{j != i} (x_j - x_i) / |x_i - x_j|^2
    Project to tangent: g_i -= (g_i . x_i) x_i
    """
    n = len(pts)
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]  # (n,n,3)
    dist = np.sqrt(np.sum(diff * diff, axis=2))  # (n,n)
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist, 1.0)
    
    # Energy: -sum_{i<j} log(dist_ij)
    # Use upper triangle
    iu = np.triu_indices(n, k=1)
    e = -math.fsum(math.log(dist[iu[0][k], iu[1][k]]) for k in range(len(iu[0])))
    
    # Gradient: for each i, sum over j != i of (x_j - x_i) / dist_ij^2
    # diff[i,j] = x_i - x_j, so (x_j - x_i) = -diff[i,j]
    # grad_i = sum_j (x_j - x_i) / dist_ij^2 = -sum_j diff[i,j] / dist_ij^2
    inv_dist2 = 1.0 / (dist * dist)
    np.fill_diagonal(inv_dist2, 0.0)
    # grad[i] = sum_j (-diff[i,j]) * inv_dist2[i,j] over j
    # = -sum_j diff[i,j] * inv_dist2[i,j]
    grad = -np.sum(diff * inv_dist2[:, :, np.newaxis], axis=1)
    
    # Project to tangent space
    dot = np.sum(grad * pts, axis=1)
    grad = grad - dot[:, np.newaxis] * pts
    
    return e, grad

def _normalize(pts):
    norms = np.sqrt(np.sum(pts * pts, axis=1, keepdims=True))
    norms = np.where(norms < 1e-12, 1.0, norms)
    return pts / norms

def construct():
    t_start = time.time()
    deadline = t_start + 140.0  # safety margin
    
    # Initial configuration
    pts = _fibonacci_sphere(N)
    pts = _normalize(pts)
    
    e_best, _ = _energy_and_grad(pts)
    pts_best = pts.copy()
    
    # Gradient descent with adaptive step size and simulated annealing
    # Phase 1: Pure gradient descent to converge
    # Phase 2: Simulated annealing to escape local minima
    
    # Precompute some constants
    rng = np.random.RandomState(42)
    
    # Phase 1: Gradient descent
    step_size = 0.01
    min_step = 1e-8
    max_iter = 5000
    
    for it in range(max_iter):
        if time.time() > deadline:
            break
        
        e, grad = _energy_and_grad(pts)
        
        # Adaptive step: use line search
        # Try step_size, if energy increases, reduce
        best_e = e
        best_pts = pts
        ss = step_size
        for _ in range(20):
            new_pts = _normalize(pts - ss * grad)
            new_e, _ = _energy_and_grad(new_pts)
            if new_e < best_e:
                best_e = new_e
                best_pts = new_pts
            else:
                ss *= 0.5
                if ss < min_step:
                    break
        pts = best_pts
        step_size = ss * 1.1 if best_e < e else ss * 0.5
        
        if best_e < e_best:
            e_best = best_e
            pts_best = pts.copy()
        
        # Check convergence
        if step_size < min_step:
            break
    
    # Phase 2: Simulated annealing with random perturbations
    T = 0.1  # initial temperature
    T_min = 1e-6
    cooling = 0.999
    n_sa_iters = 100000
    
    for it in range(n_sa_iters):
        if time.time() > deadline:
            break
        
        e, grad = _energy_and_grad(pts)
        
        # Gradient step
        new_pts = _normalize(pts - 0.005 * grad)
        new_e, _ = _energy_and_grad(new_pts)
        
        # Random perturbation
        if it % 50 == 0:
            # Pick a random point and perturb it
            idx = rng.randint(N)
            # Random direction on tangent plane
            rand_vec = rng.randn(3)
            tangent = rand_vec - np.dot(rand_vec, pts[idx]) * pts[idx]
            norm_t = np.sqrt(np.dot(tangent, tangent))
            if norm_t > 1e-12:
                tangent /= norm_t
                angle = rng.randn() * 0.05 * (T + 0.01)
                cos_a = np.cos(angle)
                sin_a = np.sin(angle)
                pts[idx] = cos_a * pts[idx] + sin_a * tangent
                pts[idx] /= np.sqrt(np.dot(pts[idx], pts[idx]))
        
        # Metropolis acceptance
        delta_e = new_e - e
        if delta_e < 0 or rng.rand() < math.exp(-delta_e / max(T, 1e-10)):
            pts = new_pts
            e = new_e
        
        if e < e_best:
            e_best = e
            pts_best = pts.copy()
        
        T = max(T * cooling, T_min)
        
        # If T is very low, do more gradient steps
        if T < T_min * 10:
            # Fine gradient descent
            for _ in range(10):
                if time.time() > deadline:
                    break
                e2, g2 = _energy_and_grad(pts)
                pts = _normalize(pts - 0.001 * g2)
                e3, _ = _energy_and_grad(pts)
                if e3 < e_best:
                    e_best = e3
                    pts_best = pts.copy()
    
    # Final polishing: many small gradient steps
    for _ in range(200):
        if time.time() > deadline:
            break
        e, grad = _energy_and_grad(pts_best)
        pts_best = _normalize(pts_best - 0.0001 * grad)
    
    return _normalize(pts_best)

# EVOLVE-BLOCK-END
