# EVOLVE-BLOCK-START
import numpy as np
import math
import time

N = 150

def _energy_and_grad(pts):
    """
    Compute logarithmic energy and gradient.
    Energy: sum_{i<j} log(1 / ||x_i - x_j||)
    Gradient: sum_{j != i} (x_i - x_j) / ||x_i - x_j||^2
    """
    n = len(pts)
    # Compute pairwise squared distances
    # diff shape (n, n, 3)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Mask diagonal to avoid division by zero
    # We set diagonal to 1.0 for division, then zero out later
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy calculation
    # We only need upper triangle
    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dist_sq[iu])
    # Use np.sum for speed in optimization loop; evaluator uses fsum
    energy = -np.sum(np.log(dists))
    
    # Gradient calculation
    # inv_dist_sq shape (n, n)
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad_i = sum_j (x_i - x_j) / dist_sq_ij
    # diff * inv_dist_sq[:,:,None] -> (n, n, 3)
    # Sum over axis 1 (j)
    grad = np.sum(diff * inv_dist_sq[:, :, np.newaxis], axis=1)
    
    return energy, grad

def _project_to_sphere(pts):
    """Project points onto the unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _halton_init(n, offset=0):
    """Generate n points using the Halton sequence in 3D with an offset."""
    def halton_sequence(n, base, start=1):
        seq = []
        for i in range(start, start + n):
            result = 0.0
            f = 1.0 / base
            num = i
            while num > 0:
                num, rem = divmod(num, base)
                result += rem * f
                f /= base
            seq.append(result)
        return np.array(seq)
    
    bases = [2, 3, 5]
    h1 = halton_sequence(n, bases[0], start=1 + offset)
    h2 = halton_sequence(n, bases[1], start=1 + offset)
    h3 = halton_sequence(n, bases[2], start=1 + offset)
    
    u = h1
    v = h2
    
    theta = 2.0 * np.pi * u
    phi = np.arccos(1.0 - 2.0 * v)
    
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    pts = np.column_stack([x, y, z])
    return pts

def _cartesian_gradient_step(pts, max_iter=50, lr_init=0.01):
    """Projected gradient descent in Cartesian space."""
    x = pts.copy()
    lr = lr_init
    
    for it in range(max_iter):
        energy, grad_e = _energy_and_grad(x)
        
        # Project gradient to tangent space
        inner = np.sum(grad_e * x, axis=1)
        grad_t = grad_e - inner[:, np.newaxis] * x
        
        grad_norm = np.linalg.norm(grad_t)
        if grad_norm < 1e-12:
            break
        
        # Armijo backtracking line search
        t = lr
        c1 = 1e-4
        rho = 0.5
        improved = False
        
        for _ in range(20):
            x_trial = x - t * grad_t
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            
            if e_trial < energy - c1 * t * grad_norm**2:
                x = x_trial
                energy = e_trial
                improved = True
                break
            else:
                t *= rho
        
        if not improved:
            break
            
        # Update learning rate
        if improved:
            lr = min(lr * 1.5, 0.1)
        else:
            lr *= 0.5
            
    return x, energy

def _riemannian_cg(pts, max_iter=100, tol=1e-10):
    """Riemannian conjugate gradient on the sphere."""
    x = pts.copy()
    
    energy, grad_e = _energy_and_grad(x)
    inner = np.sum(grad_e * x, axis=1)
    grad_r = grad_e - inner[:, np.newaxis] * x
    
    p = -grad_r.copy()
    grad_r_prev = grad_r.copy()
    
    for it in range(max_iter):
        grad_norm = np.linalg.norm(grad_r)
        if grad_norm < tol:
            break
        
        # Direction p is in tangent space
        # Exponential map: x_new = x * cos(t * ||p||) + p/||p|| * sin(t * ||p||)
        p_norm = np.linalg.norm(p, axis=1, keepdims=True)
        p_norm[p_norm < 1e-12] = 1e-12
        p_unit = p / p_norm
        
        # Line search on manifold
        t = 1.0
        c1 = 1e-4
        rho = 0.5
        improved = False
        
        # Initial step size heuristic
        t = min(t, 1.0 / (grad_norm * 10)) # rough scale
        
        for _ in range(20):
            x_trial = x * np.cos(t * p_norm) + p_unit * np.sin(t * p_norm)
            x_trial = _project_to_sphere(x_trial)
            e_trial, _ = _energy_and_grad(x_trial)
            
            # Armijo condition
            # grad_r is tangent, p is tangent. <grad_r, p> is the directional derivative
            dir_deriv = np.sum(grad_r * p)
            
            if e_trial < energy - c1 * t * dir_deriv:
                x = x_trial
                energy = e_trial
                improved = True
                break
            else:
                t *= rho
        
        if not improved:
            break
            
        # Compute new gradient
        _, grad_e_new = _energy_and_grad(x)
        inner_new = np.sum(grad_e_new * x, axis=1)
        grad_r_new = grad_e_new - inner_new[:, np.newaxis] * x
        
        # Polak-Ribiere beta
        beta_num = np.sum(grad_r_new * (grad_r_new - grad_r_prev))
        beta_den = np.sum(grad_r_prev * grad_r_prev)
        beta = max(0.0, beta_num / beta_den) if beta_den > 1e-16 else 0.0
        
        p = -grad_r_new + beta * p
        
        # Project p to tangent space
        inner_p = np.sum(p * x, axis=1)
        p = p - inner_p[:, np.newaxis] * x
        
        grad_r_prev = grad_r_new
        
    return x, energy

def construct():
    best_pts = None
    best_energy = float('inf')
    
    # Try multiple Halton offsets
    for offset in [0, 1, 2, 3, 4, 5]:
        pts = _halton_init(N, offset=offset)
        pts = _project_to_sphere(pts)
        
        # Phase 1: Cartesian Gradient
        pts, e = _cartesian_gradient_step(pts, max_iter=50, lr_init=0.01)
        
        # Phase 2: Riemannian CG
        pts, e = _riemannian_cg(pts, max_iter=100, tol=1e-10)
        
        # Phase 3: Cartesian Gradient (finer)
        pts, e = _cartesian_gradient_step(pts, max_iter=50, lr_init=0.005)
        
        # Phase 4: Riemannian CG (polish)
        pts, e = _riemannian_cg(pts, max_iter=100, tol=1e-11)
        
        pts = _project_to_sphere(pts)
        
        if np.all(np.isfinite(pts)) and e < best_energy:
            # Check minimum distance
            min_d = float('inf')
            # Vectorized min distance check for speed
            diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
            dists = np.linalg.norm(diff, axis=2)
            np.fill_diagonal(dists, 1.0)
            min_d = np.min(dists)
            
            if min_d > 1e-9:
                best_energy = e
                best_pts = pts.copy()
                
    if best_pts is None:
        pts = _halton_init(N, offset=0)
        pts = _project_to_sphere(pts)
        pts, _ = _cartesian_gradient_step(pts, max_iter=50, lr_init=0.01)
        pts, _ = _riemannian_cg(pts, max_iter=100, tol=1e-10)
        best_pts = pts
        
    return best_pts
# EVOLVE-BLOCK-END
