import numpy as np
from scipy.optimize import minimize
import time
import math

N = 150

def _energy_and_grad(x_flat):
    """
    Fast calculation of energy and gradient on the sphere.
    Returns energy and flattened gradient projected to tangent space.
    """
    x = x_flat.reshape(N, 3)
    
    # Normalize to ensure points are on sphere (handles small drifts)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    # Avoid division by zero
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / safe_norms
    
    # Pairwise squared distances: |x_i - x_j|^2 = 2 - 2 x_i . x_j
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    
    # Set diagonal to a large number so it doesn't affect sums/grads improperly
    # We will mask them out later, but filling with 1e30 avoids log(0) or div(0) issues
    # in vectorized ops if not careful. 
    np.fill_diagonal(dist_sq, 1e30)
    
    # Ensure no negative zeros or tiny negatives due to floating point
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    # Log energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Using log(d) = 0.5 * log(d^2)
    log_dist_sq = np.log(dist_sq)
    
    # Mask for upper triangle (i < j)
    iu = np.triu_indices(N, k=1)
    
    # Energy is - sum log(d_ij) = -0.5 * sum log(d_ij^2)
    e = -0.5 * math.fsum(log_dist_sq[iu].tolist())
    
    # Gradient calculation
    # dE/dx_i = sum_{j!=i} (x_i - x_j) / |x_i - x_j|^2
    # Let's compute weights W_ij = 1 / |x_i - x_j|^2
    weights = 1.0 / dist_sq
    np.fill_diagonal(weights, 0.0)
    
    # sum_j W_ij x_j
    w_x = weights @ x
    # sum_j W_ij
    w_sum = np.sum(weights, axis=1)
    
    # Raw gradient in R3
    grad_raw = -x * w_sum[:, np.newaxis] + w_x
    
    # Project to tangent space: g_t = g_raw - (g_raw . x) x
    g_dot_x = np.sum(grad_raw * x, axis=1, keepdims=True)
    grad_tangent = grad_raw - g_dot_x * x
    
    return e, grad_tangent.flatten()

def _energy_only(x):
    """Compute energy only for quick checks."""
    x = x.reshape(N, 3)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    safe_norms = np.where(norms < 1e-12, 1.0, norms)
    x = x / safe_norms
    
    dot = x @ x.T
    dist_sq = 2.0 - 2.0 * dot
    np.fill_diagonal(dist_sq, 1e30)
    dist_sq = np.maximum(dist_sq, 1e-30)
    
    log_dist_sq = np.log(dist_sq)
    iu = np.triu_indices(N, k=1)
    
    return -0.5 * math.fsum(log_dist_sq[iu].tolist())

def _bb_step(x, g, prev_x=None, prev_g=None, alpha_init=1.0):
    """
    Barzilai-Borwein step size calculation.
    """
    if prev_x is None or prev_g is None:
        return alpha_init
    
    s = x - prev_x
    y = g - prev_g
    
    ys = np.dot(y, s)
    if abs(ys) < 1e-14:
        return alpha_init
        
    ss = np.dot(s, s)
    yy = np.dot(y, y)
    
    if ss < 1e-14 or yy < 1e-14:
        return alpha_init
        
    # BB steps: a = <s,s>/<s,y> or a = <s,y>/<y,y>
    # We use a weighted average or alternate. For robustness, let's pick based on norm
    a1 = ss / ys
    a2 = ys / yy
    
    # Clip to reasonable range to prevent divergence
    a = np.clip(a1, 0.01, 10.0)
    # Actually, alternating BB steps are often better. Let's just use one for simplicity 
    # or check which one is "safer" (smaller gradient norm reduction prediction is hard here).
    # Let's stick to a1 (s-type) which is generally more stable for descent.
    return a

def _polish_bb(x, max_iter=500, tol=1e-14):
    """
    Polish using Gradient Descent with Barzilai-Borwein step sizes.
    """
    x = x.copy()
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    x = x / np.where(norms < 1e-12, 1.0, norms)
    
    curr_e = _energy_only(x)
    
    g_curr = _energy_and_grad(x.flatten())[1].reshape(N, 3)
    
    prev_x = x
    prev_g = g_curr
    
    for i in range(max_iter):
        g_norm = np.linalg.norm(g_curr)
        if g_norm < tol:
            break
            
        alpha = _bb_step(x, g_curr, prev_x, prev_g)
        
        # Backtracking to ensure sufficient decrease
        best_alpha = alpha
        best_x = x
        best_e = curr_e
        
        for k in range(10):
            x_new = x - best_alpha * g_curr
            # Renormalize
            norms_new = np.linalg.norm(x_new, axis=1, keepdims=True)
            x_new = x_new / np.where(norms_new < 1e-12, 1.0, norms_new)
            
            e_new = _energy_only(x_new)
            
            if e_new < curr_e - 1e-4 * best_alpha * g_norm**2:
                best_x = x_new
                best_e = e_new
                break
            best_alpha *= 0.5
        
        if best_e >= curr_e:
            # If no improvement, try a smaller step or break
            if best_alpha < 1e-8:
                break
            # Force a small step if stuck
            x_new = x - 1e-4 * g_curr
            norms_new = np.linalg.norm(x_new, axis=1, keepdims=True)
            x_new = x_new / np.where(norms_new < 1e-12, 1.0, norms_new)
            e_new = _energy_only(x_new)
            if e_new < curr_e:
                best_x = x_new
                best_e = e_new
            else:
                break
                
        # Update state
        prev_x = x
        prev_g = g_curr
        
        x = best_x
        curr_e = best_e
        
        g_curr = _energy_and_grad(x.flatten())[1].reshape(N, 3)
        
    return x

def _spectral_init(rng, n_harmonics=4, noise_scale=0.05):
    """
    Generate points using a spectral basis deformation of a base Fibonacci grid.
    This creates low-frequency, globally coherent deformations that may avoid
    shallow local minima better than pure noise.
    """
    # Base Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_base = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Simple spectral deformation using sine/cosine of spherical coordinates
    # theta = arccos(z), phi is already defined
    theta = np.arccos(np.clip(z, -1, 1))
    
    x_def = x_base.copy()
    
    for l in range(1, n_harmonics + 1):
        # Add random low-frequency perturbations
        a = rng.normal(0, noise_scale / l, size=3)
        b = rng.normal(0, noise_scale / l, size=3)
        
        # Perturb in local tangent directions
        # This is a crude approximation of spherical harmonics, but captures the spirit
        # of low-frequency global modes
        pert = a * np.sin(l * theta)[:, np.newaxis] + b * np.cos(l * phi)[:, np.newaxis]
        x_def += pert
        
    # Normalize
    norms = np.linalg.norm(x_def, axis=1, keepdims=True)
    x_def = x_def / np.where(norms < 1e-12, 1.0, norms)
    
    return x_def

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # 2s safety margin
    
    def time_left():
        return deadline - time.time()

    rng = np.random.default_rng(42)
    
    best_x = None
    best_energy = float('inf')
    
    # 1. Generate diverse initializations
    inits = []
    
    # A. Fibonacci Spirals with various noise levels
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    x_fib_base = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    for noise_scale in [0.0, 0.005, 0.02, 0.05, 0.1, 0.3]:
        noise = rng.normal(0, noise_scale, (N, 3))
        x = x_fib_base + noise
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        inits.append(x)
        
    # B. Random Initializations
    for _ in range(15):
        x = rng.normal(size=(N, 3))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        inits.append(x)
        
    # C. "Clustered" or "Structured" Random (Mixture of Gaussians on sphere)
    for _ in range(5):
        # Pick 5 centers
        centers = rng.normal(size=(5, 3))
        centers /= np.linalg.norm(centers, axis=1, keepdims=True)
        x = np.empty((N, 3))
        for c_idx, center in enumerate(centers):
            idxs = range(c_idx * 30, c_idx * 30 + 30)
            pts = center + rng.normal(0, 0.3, (30, 3))
            pts /= np.linalg.norm(pts, axis=1, keepdims=True)
            x[idxs] = pts
        inits.append(x)
        
    # D. Spectral Deformations (New Structural Idea)
    for n_h in [2, 3, 4, 5]:
        for ns in [0.02, 0.05, 0.1]:
            x_spec = _spectral_init(rng, n_harmonics=n_h, noise_scale=ns)
            inits.append(x_spec)
    
    # Sort inits by initial energy? Optional, but good to start with promising ones.
    # Calculating energy for all is fast.
    init_energies = [_energy_only(x) for x in inits]
    sorted_indices = np.argsort(init_energies)
    
    # Process top 25 initializations
    num_trials = min(25, len(inits))
    
    for idx in sorted_indices[:num_trials]:
        if time_left() < 10.0:
            break
            
        x0 = inits[idx]
        
        try:
            # Phase 1: L-BFGS-B
            # Use a moderate maxiter, rely on polish for fine tuning
            result = minimize(
                _energy_and_grad, 
                x0.flatten(), 
                jac=True, 
                method='L-BFGS-B', 
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 20}
            )
            
            x_opt = result.x.reshape(N, 3)
            
            # Phase 2: BB Polish
            if time_left() > 5.0:
                x_opt = _polish_bb(x_opt, max_iter=200, tol=1e-13)
                
            e_opt = _energy_only(x_opt)
            
            if e_opt < best_energy:
                best_energy = e_opt
                best_x = x_opt.copy()
                
        except Exception as e:
            # Skip failed runs
            continue
            
    # If no valid result (shouldn't happen), return a valid default
    if best_x is None:
        # Return Fibonacci spiral
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_x = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        
    # Final normalization and sanity check
    norms = np.linalg.norm(best_x, axis=1, keepdims=True)
    best_x = best_x / np.where(norms < 1e-12, 1.0, norms)
    best_x = np.where(np.isfinite(best_x), best_x, 0.0)
    
    # Ensure no points are too close (distance > 1e-9)
    # If there are collisions, the energy is +inf or very large. 
    # The optimizer should prevent this. 
    # Just in case, we don't do explicit separation here as it might break the gradient flow 
    # and the evaluator checks for < 1e-9. Our energy function penalizes small distances heavily.
    
    return best_x

if __name__ == "__main__":
    # Self-test
    pts = construct()
    e = _energy_only(pts)
    print(f"Final Energy: {e}")
    print(f"Shape: {pts.shape}")
