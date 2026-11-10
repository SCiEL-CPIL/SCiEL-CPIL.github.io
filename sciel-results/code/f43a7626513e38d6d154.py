"""
Minimize logarithmic energy of 100 points on the sphere using
Riemannian Conjugate Gradient (RCG) and Basin Hopping.
"""
import time
import math
import numpy as np

N = 100
TIME_LIMIT = 145.0  # Safety margin before 150s kill

# EVOLVE-BLOCK-START

def _energy(pts):
    """
    Compute logarithmic energy: sum_{i<j} log(1/||x_i - x_j||)
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    
    # Avoid log(0) on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[iu])
    
    # Energy is sum log(1/d) = -sum log(d)
    energy = -np.sum(np.log(dists))
    return energy

def _energy_grad(pts):
    """
    Compute energy and Euclidean gradient.
    grad_i = sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=-1)
    np.fill_diagonal(dist_sq, 1.0)
    
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    grad = np.sum(diff * inv_dist_sq[..., np.newaxis], axis=1)
    
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[iu])
    energy = -np.sum(np.log(dists))
    
    return energy, grad

def _riemannian_grad(pts, grad_euclidean):
    """
    Project Euclidean gradient onto tangent space.
    grad_tan_i = grad_euclidean_i - <grad_euclidean_i, x_i> x_i
    """
    dot = np.sum(pts * grad_euclidean, axis=1, keepdims=True)
    return grad_euclidean - dot * pts

def _retract_sphere(pts, vecs):
    """
    Retract points from tangent space back to sphere.
    p_new = sin(||v||)/||v|| * v + cos(||v||) * p
    """
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    
    sin_norms = np.sin(safe_norms)
    cos_norms = np.cos(safe_norms)
    
    new_pts = (sin_norms / safe_norms) * vecs + cos_norms * pts
    # Normalize to fix numerical drift
    return new_pts / np.linalg.norm(new_pts, axis=1, keepdims=True)

def _parallel_transport(pts, vecs, new_pts):
    """
    Parallel transport vectors from tangent space at pts to new_pts.
    Simple approximation: project back to new tangent space.
    For RCG, this is sufficient for convergence.
    """
    dot = np.sum(new_pts * vecs, axis=1, keepdims=True)
    return vecs - dot * new_pts

def _fibonacci_sphere(n):
    """Generate Fibonacci lattice points on sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _random_sphere(n, seed=0):
    """Generate random points on sphere."""
    rng = np.random.RandomState(seed)
    vecs = rng.randn(n, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms

def _rcg_minimize(pts, time_budget, tol=1e-8, max_iter=1000):
    """
    Riemannian Conjugate Gradient minimization.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy, grad_euc = _energy_grad(current_pts)
    current_grad = _riemannian_grad(current_pts, grad_euc)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    # Initial search direction
    direction = -current_grad.copy()
    
    for k in range(max_iter):
        if time.time() - start_time > time_budget:
            break
            
        grad_norm = np.linalg.norm(current_grad)
        if grad_norm < tol:
            break
            
        # Line search (Backtracking Armijo)
        alpha = 1.0
        c1 = 1e-4
        rho = 0.5
        
        # Compute gradient dot direction
        grad_dot_dir = np.sum(current_grad * direction)
        
        new_pts = current_pts
        new_energy = current_energy
        
        for _ in range(20):
            new_pts = _retract_sphere(current_pts, alpha * direction)
            new_energy, new_grad_euc = _energy_grad(new_pts)
            new_grad = _riemannian_grad(new_pts, new_grad_euc)
            
            # Armijo condition
            if new_energy <= current_energy + c1 * alpha * grad_dot_dir:
                break
            alpha *= rho
        
        if new_energy < best_energy:
            best_energy = new_energy
            best_pts = new_pts.copy()
            
        # Check for convergence or failure
        if alpha < 1e-12:
            break
            
        # Update direction
        # Fletcher-Reeves or Hestenes-Stiefel
        # Hestenes-Stiefel is often better for non-quadratic
        # beta = <new_grad - current_grad, direction> / <direction, direction>
        # But we need to transport gradients.
        
        # Simple approach: Restart if non-descent
        if np.sum(new_grad * direction) > 0:
            direction = -new_grad
        else:
            # Hestenes-Stiefel beta
            diff_grad = new_grad - current_grad
            num = np.sum(diff_grad * direction)
            den = np.sum(direction * direction)
            if den < 1e-12:
                beta = 0
            else:
                beta = num / den
            # Clip beta to avoid explosion
            beta = max(0, min(beta, 1.0))
            direction = -new_grad + beta * direction
            
        current_pts = new_pts
        current_energy = new_energy
        current_grad = new_grad
        
    return best_pts, best_energy

def _basin_hopping(pts, time_budget):
    """
    Basin Hopping: Random kick -> RCG Minimize.
    """
    start_time = time.time()
    current_pts = pts.copy()
    current_energy = _energy(current_pts)
    
    best_pts = current_pts.copy()
    best_energy = current_energy
    
    rng = np.random.RandomState(123)
    
    # Kick size: start large, maybe decay?
    # For log energy, the landscape is complex.
    # A kick that moves points by ~1 radian is significant.
    kick_scale = 1.0
    
    while time.time() - start_time < time_budget:
        # Generate random tangent vector
        kick = rng.randn(N, 3) * kick_scale
        # Project to tangent space of current_pts
        kick = kick - np.sum(kick * current_pts, axis=1, keepdims=True) * current_pts
        
        # Apply kick
        kicked_pts = _retract_sphere(current_pts, kick)
        
        # Minimize from kicked point
        # Allocate small budget per minimization?
        # Or just run RCG for a fixed number of iterations or until convergence?
        # RCG is fast. Let's allow it to run until convergence or a small time slice.
        # Since we don't know how many kicks we can afford, let's limit RCG iterations per kick.
        
        # Adaptive time budget for RCG:
        # If we've done many kicks, reduce time per kick?
        # Or just fix max_iter for RCG inside basin hopping.
        # Let's use a time-sliced RCG.
        # Estimate time per RCG step.
        # N=100, energy+grad is ~1ms. 1000 iters = 1s.
        # So we can do ~100 kicks in 100s?
        
        # Let's just run RCG with a time budget of 0.5s per kick.
        # If it finishes early, great.
        
        sub_budget = 0.5
        refined_pts, refined_energy = _rcg_minimize(kicked_pts, sub_budget, max_iter=200)
        
        if refined_energy < best_energy:
            best_energy = refined_energy
            best_pts = refined_pts.copy()
            
        # Update current state to the best found so far?
        # Standard Basin Hopping: current = best
        current_pts = best_pts
        current_energy = best_energy
        
        # Optional: decay kick scale
        # kick_scale *= 0.99
        
    return best_pts, best_energy

def construct():
    start_time = time.time()
    deadline = start_time + TIME_LIMIT

    best_pts = None
    best_energy = float('inf')
    
    # 1. Initializer: Fibonacci
    pts0 = _fibonacci_sphere(N)
    e0 = _energy(pts0)
    best_energy = e0
    best_pts = pts0.copy()
    
    # 2. Initial Polish: RCG from Fibonacci
    # Give it 10% of time
    polish_time = 0.1 * TIME_LIMIT
    if time.time() < deadline - polish_time:
        pts_polished, e_polished = _rcg_minimize(pts0, polish_time)
        if e_polished < best_energy:
            best_energy = e_polished
            best_pts = pts_polished.copy()
    else:
        pts_polished = best_pts
    
    # 3. Basin Hopping
    # Allocate remaining time (minus safety) to BH
    bh_time = deadline - time.time() - 2.0 # 2s safety
    if bh_time > 5.0:
        pts_bh, e_bh = _basin_hopping(pts_polished, bh_time)
        if e_bh < best_energy:
            best_energy = e_bh
            best_pts = pts_bh.copy()
            
    # 4. Final Polish: RCG from best found
    # If we have time left (BH might finish early or we skipped it)
    final_time = deadline - time.time() - 1.0
    if final_time > 1.0:
        pts_final, e_final = _rcg_minimize(best_pts, final_time)
        if e_final < best_energy:
            best_energy = e_final
            best_pts = pts_final.copy()

    # Final check: ensure valid points
    if best_pts is None or np.any(~np.isfinite(best_pts)):
        best_pts = _fibonacci_sphere(N)
        
    # Normalize
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts

# EVOLVE-BLOCK-END
